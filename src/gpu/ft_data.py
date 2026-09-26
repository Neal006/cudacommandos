"""Training data for the band reranker: band-matched pairs (option 1) and in-data augmentation (option 2).

Option 1 — band-matched pairs. reranker.make_training_pairs trains on the highest BLOCKING-similarity
negatives, but run_v2 only ever shows the reranker pairs whose STAGE-1 probability is in the band.
`band_pairs` closes that gap: it scores a pool of train entities that are OUTSIDE the GBDT run's sample
with that run's own stage-1 models (runs/<id>/model.pkl, out of sample, exactly like test) and keeps
the pairs in the band, plus a small fraction outside it so the score still orders easy pairs.

Option 2 — augmentation, Ditto-style but from our own rows only (DATA_SECURITY §1 forbids external
data): script views (native only / transliteration only), dropped name or address tokens, shuffled
address, stripped legal form. Applied to TRAIN rows after the valid split, never to valid rows.

    python src/gpu/ft_data.py band --run runs/<id> --entities 150000     # -> INTERIM/ft_band_*.parquet
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from normalize import LEGAL_SUFFIXES  # noqa: E402

PAIR_COLS = ["s1_id", "a", "b", "y"]          # what every trainer needs; band pairs add cand_id, p1


# ------------------------------------------------------------------ text (reranker.serialize format)
def parse(s):
    """'name: N[ | T] addr: A' -> (N, T, A). T is '' when serialize() added no transliteration."""
    body = s[len("name: "):] if s.startswith("name: ") else s
    name, _, addr = body.partition(" addr: ")
    name, sep, tl = name.rpartition(" | ")
    return (name, tl, addr) if sep else (tl, "", addr)


def render(name, tl, addr):
    return "name: " + name + (" | " + tl if tl else "") + " addr: " + addr


def _native_only(s, rng):
    n, _, a = parse(s)
    return render(n, "", a)


def _latin_only(s, rng):
    n, t, a = parse(s)
    return render(t, "", a) if t else s


def _drop_token(tokens, rng):
    if len(tokens) < 2:
        return tokens
    i = rng.integers(len(tokens))
    return tokens[:i] + tokens[i + 1:]


def _drop_name_token(s, rng):
    n, t, a = parse(s)
    return render(" ".join(_drop_token(n.split(), rng)), t, a)


def _drop_addr_token(s, rng):
    n, t, a = parse(s)
    return render(n, t, " ".join(_drop_token(a.split(), rng)))


def _shuffle_addr(s, rng):
    n, t, a = parse(s)
    tok = a.split()
    return render(n, t, " ".join(tok[i] for i in rng.permutation(len(tok))))


def _strip_legal(s, rng):
    n, t, a = parse(s)
    kept = [w for w in n.split() if w not in LEGAL_SUFFIXES]
    return render(" ".join(kept) if kept else n, t, a)


AUG_OPS = {"native_only": _native_only, "latin_only": _latin_only, "drop_name_token": _drop_name_token,
           "drop_addr_token": _drop_addr_token, "shuffle_addr": _shuffle_addr, "strip_legal": _strip_legal}


def augment_train(a, b, y, groups, tr_idx, rate, rng):
    """Append round(rate * len(tr_idx)) augmented copies of train rows (label and group kept; one random
    op on one random side) and return (a, b, y, groups, tr_idx) as NEW arrays. Valid rows never change."""
    n_new = int(round(rate * len(tr_idx)))
    if n_new <= 0:
        return a, b, y, groups, tr_idx
    src = rng.choice(tr_idx, n_new, replace=n_new > len(tr_idx))
    ops = list(AUG_OPS.values())
    na, nb = a[src].copy(), b[src].copy()
    for j in range(n_new):
        op = ops[rng.integers(len(ops))]
        if rng.random() < 0.5:
            na[j] = op(na[j], rng)
        else:
            nb[j] = op(nb[j], rng)
    g = None if groups is None else np.concatenate([np.asarray(groups, dtype=object), np.asarray(groups, dtype=object)[src]])
    return (np.concatenate([a, na]), np.concatenate([b, nb]), np.concatenate([y, y[src]]), g,
            np.concatenate([tr_idx, np.arange(len(a), len(a) + n_new)]))


# ------------------------------------------------------------------ band-matched pairs
def select_band(p1, band, outside_frac, rng):
    """Sorted row indices: every pair with band[0] <= p1 <= band[1], plus `outside_frac` of the rest."""
    inb = (p1 >= band[0]) & (p1 <= band[1])
    out = np.where(~inb)[0]
    extra = out[rng.random(len(out)) < outside_frac] if outside_frac > 0 else out[:0]
    return np.sort(np.concatenate([np.where(inb)[0], extra]))


def band_pairs(run_dir, n_entities, band=(0.05, 0.95), outside_frac=0.05, seed=7):
    """Band-matched training pairs for entities outside the GBDT run in `run_dir`. Cached in INTERIM.
    The training band is wider than run_v2's 0.2-0.8 serving band: at ~1.2% of pairs the serving band
    alone gives too few rows, and 0.05-0.95 is still the stage-1-hard distribution."""
    import blocking
    import data as D
    import ingest
    import run_v2
    from gpu.reranker import serialize
    from normalize import add_blocking_columns

    run_dir = Path(run_dir)
    ex = set(pd.read_csv(run_dir / "folds.tsv", sep="\t", dtype=str)["s1_id"])
    path = C.INTERIM / (f"ft_band_{run_dir.name}_n{n_entities}_s{seed}_b{band[0]}-{band[1]}"
                        f"_o{outside_frac}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    bundle = pickle.loads((run_dir / "model.pkl").read_bytes())
    s1 = D.read_source(C.TRAIN_S1)
    s1 = s1[~s1[C.ID].isin(ex)]
    s1 = s1.sample(n=min(n_entities, len(s1)), random_state=seed).reset_index(drop=True)
    s1 = add_blocking_columns(s1)
    s2, s3 = ingest.blocking_frame([C.TRAIN_S2]), ingest.blocking_frame([C.TRAIN_S3])
    pairs = blocking.candidates_to_frame(blocking.generate_candidates(s1, s2, s3)).reset_index(drop=True)
    del s2, s3
    X, L, R = run_v2.featurize(pairs, "train", run_v2.stats_for("train"), extra=True)
    missing = [c for c in bundle["feat1"] if c not in X.columns]
    if missing:
        raise ValueError(f"{run_dir}: stage-1 features {missing[:5]} not produced by this code version")
    p1 = run_v2.predict(bundle["models1"], X[bundle["feat1"]].to_numpy())
    del X
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID])
    y = np.fromiter((c in truth[s] for s, c in zip(pairs["s1_id"], pairs["cand_id"])), bool, len(pairs))
    keep = select_band(p1, band, outside_frac, np.random.default_rng(seed))
    kp = pairs.iloc[keep]
    out = pd.DataFrame({"s1_id": kp["s1_id"].to_numpy(), "cand_id": kp["cand_id"].to_numpy(),
                        "a": serialize(L, kp["s1_id"]), "b": serialize(R, kp["cand_id"]),
                        "y": y[keep].astype(int), "p1": p1[keep]})
    out.to_parquet(path, index=False)
    print(f"band pairs -> {path.name}: {len(out):,} of {len(pairs):,} pairs, {int(out['y'].sum()):,} positives, "
          f"{((out['p1'] >= band[0]) & (out['p1'] <= band[1])).mean():.0%} in band", flush=True)
    return out


def load_pairs(paths):
    """Concatenate training-pair parquets (make_training_pairs / band_pairs), de-duplicated per pair."""
    frames = [pd.read_parquet(p) for p in paths]
    for p, f in zip(paths, frames):
        if missing := [c for c in PAIR_COLS if c not in f.columns]:
            raise ValueError(f"{p}: missing columns {missing}; expected at least {PAIR_COLS}")
    d = pd.concat(frames, ignore_index=True)
    key = ["s1_id", "cand_id"] if "cand_id" in d.columns and d["cand_id"].notna().all() else ["s1_id", "a", "b"]
    return d.drop_duplicates(key).reset_index(drop=True)


# ------------------------------------------------------------------ shared held-out evaluation
def eval_leak(eval_entities, trained_entities):
    """How many evaluation entities a model trained on (must be 0 for an honest comparison)."""
    return len(set(eval_entities) & set(trained_entities))


def evaluate(model_dirs, pairs_path, band=(0.2, 0.8)):
    """Score every model dir (e5 or laya) on ONE frozen pair file, so options are compared on the same
    rows. Each model's own valid split is a different draw and is not comparable across runs."""
    from sklearn.metrics import log_loss, roc_auc_score
    from gpu import reranker_module
    d = load_pairs([pairs_path])
    y = d["y"].to_numpy()
    inb = ((d["p1"] >= band[0]) & (d["p1"] <= band[1])).to_numpy() if "p1" in d.columns else np.ones(len(d), bool)
    rows = []
    for m in map(Path, model_dirs):
        ents = (m / "entities.txt").read_text(encoding="utf-8").split() if (m / "entities.txt").exists() else []
        if leak := eval_leak(d["s1_id"].astype(str), ents):
            raise SystemExit(f"{m} trained on {leak:,} of the evaluation entities; build the eval file with "
                             f"another --seed (ft_data.py band --seed 9)")
        p = reranker_module(m).score(m, d["a"].to_numpy(), d["b"].to_numpy())
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append(dict(model=m.name, auc=roc_auc_score(y, p), logloss=log_loss(y, pc, labels=[0, 1]),
                         auc_band=roc_auc_score(y[inb], p[inb]) if 0 < y[inb].mean() < 1 else float("nan"),
                         n=len(d), n_band=int(inb.sum())))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("band", help="band-matched pairs from a finished GBDT run (runs/<id>)")
    bp.add_argument("--run", required=True, help="run dir with folds.tsv and model.pkl")
    bp.add_argument("--entities", type=int, default=150000)
    bp.add_argument("--band", type=float, nargs=2, default=(0.05, 0.95))
    bp.add_argument("--outside", type=float, default=0.05, help="fraction of out-of-band pairs kept")
    bp.add_argument("--seed", type=int, default=7)
    ev = sub.add_parser("eval", help="score reranker dirs on one frozen held-out pair file")
    ev.add_argument("--pairs", required=True, help="held-out pairs, e.g. `band --seed 9` output")
    ev.add_argument("--models", nargs="+", required=True)
    ev.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8), help="serving band for auc_band")
    a = ap.parse_args()
    if a.cmd == "band":
        band_pairs(a.run, a.entities, tuple(a.band), a.outside, a.seed)
    else:
        evaluate(a.models, a.pairs, tuple(a.band))
