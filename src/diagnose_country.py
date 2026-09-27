"""Label-free per-country diagnosis of the test set.

France is 15% of test, has zero training labels, and every offline number we
have is blind to it. Submission 001 scored LB 0.943 against OOF 0.9532, and
solving for France under the observed country shares puts it near 0.90 -- six
points below US. That is our largest known loss and it has never been looked
at directly.

It can be looked at WITHOUT labels. Two questions, separately answerable:

  1. Is BLOCKING failing France?  -> orphan rate, candidates per entity, and
     the distribution of the best candidate's blocking similarity. These come
     from the candidate cache alone and need no model.

  2. Is the MATCHER failing France? -> predicted links per entity, predicted
     singleton rate, and the distribution of the model's score for the best
     candidate. These need the cached test scores, not labels.

If France looks like the US on (1) but not on (2), blocking is fine and the
matcher does not transfer -- the fix is France-aware training (pseudo-labels,
reweighting), not retrieval. If France is already behind on (1), no amount of
matcher work will recover it and the fix is a France-specific blocking pass.
Those two conclusions point at completely different days of work, which is
why this runs before either.

    python src/diagnose_country.py                       # blocking only
    python src/diagnose_country.py --scores <p.npy>      # + matcher view
    python src/diagnose_country.py --matching <tsv>      # + shipped output

Memory: the candidate frame is 52M rows, so every aggregate here is done with
polars streaming rather than by materializing it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def log(m):
    print(m, flush=True)


def country_map() -> pl.DataFrame:
    """entity_id -> country for every test Source-1 entity."""
    return pl.read_csv(C.TEST_S1, separator="\t", infer_schema=False,
                       columns=[C.ID, C.COUNTRY]).rename({C.ID: "s1_id"})


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main(a):
    cands = C.INTERIM / f"cands_test_k30_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}_ctry1_nall.parquet"
    if not cands.exists():
        raise SystemExit(f"candidate cache not found: {cands}")

    cm = country_map()
    n_by_country = cm.group_by(C.COUNTRY).len().sort("len", descending=True)
    total = cm.height
    log(f"test Source-1 entities: {total:,}")
    for r in n_by_country.iter_rows(named=True):
        log(f"  {r[C.COUNTRY]:<10} {r['len']:>9,}  {r['len']/total:6.2%}")

    # ---- 1. blocking view: per entity, how many candidates and how good is the best
    log("\n=== 1. BLOCKING (no model involved) ===")
    per_entity = (
        pl.scan_parquet(cands)
        .group_by("s1_id")
        .agg(pl.len().alias("n_cand"), pl.col("block_sim").max().alias("top_sim"))
        .collect(engine="streaming")
    )
    # entities absent from the cache got zero candidates
    j = cm.join(per_entity, on="s1_id", how="left").with_columns(
        pl.col("n_cand").fill_null(0), pl.col("top_sim").fill_null(0.0))

    log(f"{'country':<10} {'entities':>9} {'orphan%':>8} {'cand/ent':>9} "
        f"{'top_sim p10':>12} {'p50':>7} {'p90':>7}")
    for c in [r[C.COUNTRY] for r in n_by_country.iter_rows(named=True)]:
        g = j.filter(pl.col(C.COUNTRY) == c)
        ts = g["top_sim"].to_numpy()
        log(f"{c:<10} {g.height:>9,} {(g['n_cand'] == 0).mean():>7.3%} "
            f"{g['n_cand'].mean():>9.2f} {pct(ts,10):>12.4f} {pct(ts,50):>7.4f} {pct(ts,90):>7.4f}")

    log("\nRead: a country with a LOWER top_sim distribution is one whose true")
    log("matches blocking struggles to surface -- retrieval problem, not matcher.")

    # ---- 2. matcher view: needs the cached per-pair scores
    if a.scores:
        log("\n=== 2. MATCHER (cached test scores, still no labels) ===")
        p = np.load(a.scores)
        pairs = pl.read_parquet(cands, columns=["s1_id"])
        if len(p) != pairs.height:
            raise SystemExit(f"scores {len(p):,} != pairs {pairs.height:,}")
        top = (pairs.with_columns(pl.Series("p", p.astype(np.float32)))
               .group_by("s1_id")
               .agg(pl.col("p").max().alias("top_p"),
                    (pl.col("p") >= 0.5).sum().alias("n_conf")))
        jm = cm.join(top, on="s1_id", how="left").with_columns(
            pl.col("top_p").fill_null(0.0), pl.col("n_conf").fill_null(0))
        log(f"{'country':<10} {'top_p p10':>10} {'p50':>7} {'p90':>7} "
            f"{'conf/ent':>9} {'no-conf%':>9}")
        for c in [r[C.COUNTRY] for r in n_by_country.iter_rows(named=True)]:
            g = jm.filter(pl.col(C.COUNTRY) == c)
            tp = g["top_p"].to_numpy()
            log(f"{c:<10} {pct(tp,10):>10.4f} {pct(tp,50):>7.4f} {pct(tp,90):>7.4f} "
                f"{g['n_conf'].mean():>9.2f} {(g['n_conf'] == 0).mean():>8.2%}")
        log("\nRead: if a country's top_sim (section 1) matches the others but its")
        log("top_p is lower, blocking found the right record and the MATCHER did")
        log("not recognise it. That is the domain-shift case.")

    # ---- 3. what we actually shipped
    if a.matching:
        log("\n=== 3. SHIPPED OUTPUT ===")
        m = pl.read_csv(a.matching, separator="\t", infer_schema=False).rename(
            {C.GT_S1: "s1_id", C.GT_MATCH: "ids"})
        m = m.with_columns(
            pl.when(pl.col("ids").is_null() | (pl.col("ids").str.strip_chars() == ""))
              .then(0).otherwise(pl.col("ids").str.count_matches(",") + 1).alias("n"))
        jo = cm.join(m.select("s1_id", "n"), on="s1_id", how="left").with_columns(
            pl.col("n").fill_null(0))
        log(f"{'country':<10} {'links/ent':>10} {'singleton%':>11}")
        for c in [r[C.COUNTRY] for r in n_by_country.iter_rows(named=True)]:
            g = jo.filter(pl.col(C.COUNTRY) == c)
            log(f"{c:<10} {g['n'].mean():>10.3f} {(g['n'] == 0).mean():>10.2%}")
        log("\nTrain truth averages 3.46 links/entity. A country far BELOW the")
        log("others here is one we are under-predicting on -- and under F0.5 an")
        log("empty prediction on an entity that has matches scores 0, not 0.5.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=None, help="cached per-pair scores .npy (interim/testp_*.npy)")
    ap.add_argument("--matching", default=None, help="a matching_results.tsv to profile")
    main(ap.parse_args())
