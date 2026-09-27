"""Blend cached per-pair test scores from several runs, then decide and write.

The `lgb-xgb-cat-blend` branch ensembles at TRAINING time: three libraries x
five folds, which triples training and has never been run on real data. This
ensembles at PREDICTION time instead, over score arrays we already have on
disk. It costs about two minutes rather than four hours, and every input has
already been validated by a full scoring run.

The members are genuinely decorrelated -- they differ in training sample
(30k vs 150k), in whether stage 2 saw contention-corrected claim features,
and in boosting rounds -- which is the property that makes averaging pay.
What they share is the candidate set: every cached array scores the same
51,974,499 pairs of the same parquet in the same order, so element i means
the same pair in all of them. The row-count check below is what enforces that.

    python src/ensemble.py --scores a.npy b.npy --cfg soft:expected_f:0.1
    python src/ensemble.py --scores a.npy:2 b.npy:1 --method logit

METHODS
  mean   plain average of calibrated probabilities. Fine here because every
         member is isotonic-calibrated, so they are already on one scale.
  logit  average in log-odds space. Treats 0.99 vs 0.999 as a real difference
         where `mean` barely separates them -- usually the better choice when
         members disagree in the tails, which is exactly where our decisions
         are made.
  rank   average of within-entity ranks. Throws away magnitude, so the
         expected-F decoder loses the probabilities it needs; offered only
         for diagnosis, not for a submission.

A WARNING WORTH READING
Averaging a weaker member INTO a stronger one can lower the score. Our 30k
model scored 0.951 and the 150k model should be better, so an equal-weight
blend of the two is not obviously an improvement -- weight accordingly, or
leave the weak member out. This script prints the pairwise agreement between
members and how far the blend moves from member 1, so that the decision to
spend a submission slot is made on numbers rather than on the hope that
ensembles always help.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import data as D  # noqa: E402
import decide  # noqa: E402
from redecide import parse_cfg  # noqa: E402


def log(m):
    print(m, flush=True)


def _logit(p, eps=1e-6):
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main(a):
    specs = []
    for s in a.scores:
        # "path:weight", but a Windows path starts "D:\..." so splitting on the
        # first colon eats the drive letter. Split from the right and only take
        # the tail as a weight when it actually parses as a number.
        head, sep, tail = s.rpartition(":")
        if sep and head:
            try:
                specs.append((Path(head), float(tail)))
                continue
            except ValueError:
                pass
        specs.append((Path(s), 1.0))

    arrays, n = [], None
    for path, w in specs:
        if not path.exists():
            raise SystemExit(f"missing {path}")
        v = np.load(path)
        if n is None:
            n = len(v)
        elif len(v) != n:
            raise SystemExit(f"{path} has {len(v):,} rows, expected {n:,} -- "
                             "members must score the same candidate frame")
        arrays.append(v.astype(np.float32))
        log(f"  {path.name:<28} {len(v):,} rows  weight {w}  mean {v.mean():.4f}")
    w = np.array([x[1] for x in specs], dtype=np.float64)
    w = w / w.sum()

    # How much do the members actually disagree? An ensemble of near-identical
    # members cannot help, and it is cheaper to find that out here than by
    # spending a submission slot.
    if len(arrays) > 1:
        log("\npairwise agreement (Pearson on logits, sampled):")
        idx = np.random.default_rng(0).choice(n, size=min(2_000_000, n), replace=False)
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                r = np.corrcoef(_logit(arrays[i][idx]), _logit(arrays[j][idx]))[0, 1]
                log(f"  member {i+1} vs {j+1}: r = {r:.4f}"
                    + ("   (near-identical; blending will change little)" if r > 0.99 else ""))

    if a.method == "mean":
        p = np.zeros(n, dtype=np.float64)
        for wi, v in zip(w, arrays):
            p += wi * v
    elif a.method == "logit":
        z = np.zeros(n, dtype=np.float64)
        for wi, v in zip(w, arrays):
            z += wi * _logit(v)
        p = _sigmoid(z)
    elif a.method == "rank":
        raise SystemExit("rank averaging destroys the probabilities expected_f needs; "
                         "use it for diagnosis only")
    else:
        raise SystemExit(f"unknown method {a.method}")
    log(f"\nblended by {a.method}: mean {p.mean():.4f}")

    moved = float(np.abs(p - arrays[0].astype(np.float64)).mean())
    log(f"mean |blend - member 1| = {moved:.5f}"
        + ("   (tiny; a slot spent here likely measures noise)" if moved < 1e-3 else ""))

    if a.dry_run:
        log("\n--dry-run: not writing outputs")
        return

    cands = C.INTERIM / f"cands_test_k30_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}_ctry1_nall.parquet"
    pairs = pl.read_parquet(cands, columns=["s1_id", "cand_id"]).to_pandas()
    if len(pairs) != n:
        raise SystemExit(f"candidate frame {len(pairs):,} != scores {n:,}")

    _, cfg = parse_cfg(a.cfg)
    log(f"decision {cfg}")
    df = pairs.assign(p=p)
    tsel = decide.apply(df, cfg)
    del df

    s1 = pl.read_csv(C.TEST_S1, separator="\t", infer_schema=False,
                     columns=[C.ID, C.COUNTRY]).to_pandas()
    test_ids = s1[C.ID].astype(str).tolist()
    country = pd.Series(s1[C.COUNTRY].astype(str).to_numpy(), index=test_ids)

    cand_sets = pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
    del pairs
    matches = tsel.groupby("s1_id")["cand_id"].apply(set).to_dict()
    out = D.write_outputs(test_ids, {s: matches.get(s, set()) for s in test_ids},
                          {s: cand_sets.get(s, set()) for s in test_ids})
    for k, v in out.items():
        log(f"  {k}: {v}")

    cnt = tsel.groupby("s1_id").size().reindex(test_ids, fill_value=0)
    by = pd.DataFrame({"n": cnt.to_numpy(), "c": country.reindex(test_ids).to_numpy()})
    log(f"\n{'country':<10} {'links/ent':>10} {'singleton%':>11}")
    for c, g in by.groupby("c"):
        log(f"{c:<10} {g['n'].mean():>10.3f} {(g['n'] == 0).mean():>10.2%}")
    log(f"{'ALL':<10} {by['n'].mean():>10.3f} {(by['n'] == 0).mean():>10.2%}")
    if out["entities_with_matches_outside_candidates"]:
        raise SystemExit("matches outside candidates -- would be rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="cached .npy score files, optionally path:weight")
    ap.add_argument("--method", default="logit", choices=["mean", "logit", "rank"])
    ap.add_argument("--cfg", default="soft:expected_f:0.1", help="assign:select:value")
    ap.add_argument("--dry-run", action="store_true",
                    help="report agreement and blend shift without writing outputs")
    main(ap.parse_args())
