"""Option 5 — distil a trained reranker (laya, 322M) into e5-small (118M) for ~5x faster band scoring.

Student target = alpha * label + (1 - alpha) * teacher probability (BCE accepts soft targets), trained
with reranker.train, so the output is an ordinary e5 reranker dir (meta kind=e5) that run_v2 --rerank
reads unchanged. Only worth running once laya has WON the A/B: it keeps laya's quality at e5's speed
(test band ~6 min instead of ~28 on a 3050).

Leakage: the student learns from the teacher, so its entities.txt is the UNION of its own pairs'
entities and the teacher's; run_v2 then refuses a GBDT sample that overlaps either. Use pairs the
teacher did not train on (e.g. `ft_data.py band --seed 8`) — scores on its own training pairs are
over-confident and teach the student less.

    python src/gpu/distill.py --teacher models/rr_laya --pairs <INTERIM/ft_band_...parquet> --out models/rr_e5_kd
"""
import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu import ft_data as FD  # noqa: E402
from gpu import reranker as RR  # noqa: E402
from gpu import reranker_module  # noqa: E402


def soft_targets(y, p, alpha):
    """alpha * y + (1 - alpha) * p, as float32. alpha = 1 is plain supervised training."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (alpha * np.asarray(y, dtype=np.float64) + (1 - alpha) * np.asarray(p, dtype=np.float64)).astype(np.float32)


def union_entities(*lists):
    return sorted(set().union(*map(set, lists)))


def unseen_by_teacher(s1_ids, teacher_entities):
    """Mask of pairs whose S1 entity the teacher never trained on: its scores on its own training
    pairs are memorised, over-confident targets that would also inflate the student's metrics."""
    seen = set(teacher_entities)
    return np.fromiter((s not in seen for s in s1_ids), bool, len(s1_ids))


def distill(teacher, pairs_paths, out, alpha=0.5, epochs=1, bs=64, augment=0.0, run_dir=None):
    d = FD.load_pairs(pairs_paths)
    teacher = Path(teacher)
    t_ent = teacher / "entities.txt"
    if t_ent.exists():
        keep = unseen_by_teacher(d["s1_id"].astype(str).to_numpy(), t_ent.read_text(encoding="utf-8").split())
        if not keep.any():
            raise SystemExit(f"distill: every pair's entity is in {t_ent}; build pairs the teacher did not see "
                             f"(e.g. `ft_data.py band --seed 8`)")
        if not keep.all():
            print(f"distill: dropped {int((~keep).sum()):,} of {len(d):,} pairs whose entity the teacher "
                  f"trained on", flush=True)
        d = d[keep].reset_index(drop=True)
    else:
        warnings.warn(f"teacher {teacher} has no entities.txt: cannot exclude its training pairs, and the "
                      f"student's leak guard covers only its own pairs")
    tm = reranker_module(teacher)
    print(f"distill: scoring {len(d):,} pairs with teacher {teacher} ({tm.__name__})", flush=True)
    p = tm.score(teacher, d["a"].to_numpy(), d["b"].to_numpy())
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()                   # the teacher leaves the GPU before the student arrives
    except Exception:
        pass
    y = d["y"].to_numpy().astype(np.float32)
    t = soft_targets(y, p, alpha)
    meta = RR.train(d["a"], d["b"], t, out, epochs=epochs, bs=bs, run_dir=run_dir, groups=d["s1_id"],
                    y_eval=y, augment=augment,
                    extra_meta=dict(teacher=str(teacher), alpha=alpha, pair_files=[str(x) for x in pairs_paths]))
    out = Path(out)
    if t_ent.exists():
        own = (out / "entities.txt").read_text(encoding="utf-8").split()
        merged = union_entities(own, t_ent.read_text(encoding="utf-8").split())
        (out / "entities.txt").write_text("\n".join(merged), encoding="utf-8")
        print(f"distill: entities.txt = own {len(own):,} U teacher -> {len(merged):,}", flush=True)
    (out / "meta.json").write_text(json.dumps({**meta, "kind": "e5"}, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="trained reranker dir (laya or e5)")
    ap.add_argument("--pairs", nargs="+", required=True, help="training-pair parquet(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the true label vs the teacher")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--augment", type=float, default=0.0)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    distill(a.teacher, a.pairs, a.out, a.alpha, a.epochs, a.bs, a.augment, a.run_dir)
