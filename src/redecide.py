"""Re-run the decision layer over cached test scores, optionally per country.

Scoring the test set costs ~2 hours. The decision layer that turns those scores
into a submission costs about a minute. Before score caching those were welded
together, so trying a different threshold meant a full rescore and nobody tried
one. This separates them: load `interim/testp_<key>.npy`, apply a decision
config, write the TSVs.

    python src/redecide.py --scores interim/testp_<key>.npy --cfg soft:expected_f:0.1
    python src/redecide.py --scores ... --cfg soft:expected_f:0.1 --cfg France=soft:expected_f:0.4

**Why per country.** `select_expected_f` keeps the top-k maximizing expected
F0.5, and `miss` is its estimate of how many true links blocking never offered
for that entity. A higher `miss` makes the empty prediction less attractive, so
more links are kept. That parameter was fitted on US and India out-of-fold and
is applied verbatim to France, which has no labels at all.

If France's scores are systematically lower than US/India for the same true
matches -- the domain-shift case `diagnose_country.py` tests for -- then a
`miss` tuned on US/India makes France predict too few links. Under F0.5 that is
expensive in a way that is easy to underestimate: an entity that has matches
and is predicted empty scores **0**, not a partial credit. Raising France's
`miss` alone is the cheapest available correction.

**What this cannot do.** There are no French labels, so a France-specific
setting cannot be validated offline -- only justified from the label-free
distributions and then confirmed on the leaderboard. Treat any per-country
override as an experiment that costs a submission slot, not as a tuned result.
The partition (`assign`) is always applied globally before any per-country
split, because it is a constraint across entities rather than a per-country
choice.
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


def log(m):
    print(m, flush=True)


def parse_cfg(s: str):
    """'soft:expected_f:0.1' or 'hard:threshold:0.7' -> cfg dict (optionally 'Country=...')."""
    country = None
    if "=" in s:
        country, s = s.split("=", 1)
    parts = s.split(":")
    if len(parts) != 3:
        raise SystemExit(f"bad --cfg {s!r}; want assign:select:value")
    a, sel, v = parts
    if sel not in ("expected_f", "threshold"):
        raise SystemExit(f"bad select {sel!r}")
    cfg = {"assign": a, "select": sel,
           "miss": float(v) if sel == "expected_f" else None,
           "thr": float(v) if sel == "threshold" else None}
    return country, cfg


def main(a):
    cands = C.INTERIM / f"cands_test_k30_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}_ctry1_nall.parquet"
    pairs = pl.read_parquet(cands, columns=["s1_id", "cand_id"]).to_pandas()
    p = np.load(a.scores)
    if len(p) != len(pairs):
        raise SystemExit(f"scores {len(p):,} != pairs {len(pairs):,}")
    log(f"{len(pairs):,} pairs, scores from {a.scores}")

    default, per_country = None, {}
    for spec in a.cfg:
        c, cfg = parse_cfg(spec)
        if c is None:
            default = cfg
        else:
            per_country[c] = cfg
    if default is None:
        raise SystemExit("need one --cfg without a Country= prefix as the default")
    log(f"default {default}" + (f"  overrides {per_country}" if per_country else ""))

    df = pairs.assign(p=p.astype(np.float64))

    # The partition is a constraint between entities, not a per-country knob,
    # so it is resolved once over everything before any split.
    df = decide.assign(df, default["assign"])

    s1 = pl.read_csv(C.TEST_S1, separator="\t", infer_schema=False,
                     columns=[C.ID, C.COUNTRY]).to_pandas()
    test_ids = s1[C.ID].astype(str).tolist()
    country = pd.Series(s1[C.COUNTRY].astype(str).to_numpy(), index=test_ids)

    if per_country:
        cmap = df["s1_id"].map(country)
        parts = []
        for c, cfg in per_country.items():
            m = cmap == c
            if not m.any():
                log(f"  WARNING: no test entities with country {c!r} -- override unused")
                continue
            parts.append(decide.apply(df[m].drop(columns=[]), {**cfg, "assign": "none"}))
            log(f"  {c}: {int(m.sum()):,} pairs with {cfg}")
        rest = df[~cmap.isin(list(per_country))]
        parts.append(decide.apply(rest, {**default, "assign": "none"}))
        log(f"  default: {len(rest):,} pairs")
        tsel = pd.concat(parts, ignore_index=True)
    else:
        tsel = decide.apply(df, {**default, "assign": "none"})

    cand_sets = pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
    del pairs, df
    matches = tsel.groupby("s1_id")["cand_id"].apply(set).to_dict()
    out = D.write_outputs(test_ids, {s: matches.get(s, set()) for s in test_ids},
                          {s: cand_sets.get(s, set()) for s in test_ids})
    for k, v in out.items():
        log(f"  {k}: {v}")

    n = tsel.groupby("s1_id").size().reindex(test_ids, fill_value=0)
    by = pd.DataFrame({"n": n.to_numpy(), "c": country.reindex(test_ids).to_numpy()})
    log(f"\n{'country':<10} {'links/ent':>10} {'singleton%':>11}")
    for c, g in by.groupby("c"):
        log(f"{c:<10} {g['n'].mean():>10.3f} {(g['n'] == 0).mean():>10.2%}")
    log(f"{'ALL':<10} {by['n'].mean():>10.3f} {(by['n'] == 0).mean():>10.2%}")
    if out["entities_with_matches_outside_candidates"]:
        raise SystemExit("matches outside candidates -- would be rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="interim/testp_<key>.npy")
    ap.add_argument("--cfg", action="append", required=True,
                    help="assign:select:value, or Country=assign:select:value; repeatable")
    main(ap.parse_args())
