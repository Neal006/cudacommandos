"""Write output/matching_results.tsv and output/candidate_pairs.tsv, then run the official validator.

  python write_output.py                        # default decision params (work/decision_params.json)
  python write_output.py --params my.json --tag fr_b05   # leaderboard probe variant -> *_fr_b05.tsv
"""
import argparse
import json
import subprocess
import sys

import polars as pl

from config import DATA, OUT, VALIDATOR, wpath
from decide import decide, load_params
from io_utils import load_records


def _lists(pairs: pl.DataFrame, ids: pl.DataFrame, s1_all: pl.DataFrame, col: str) -> pl.DataFrame:
    named = (pairs.join(ids.rename({"idx": "cand", "entity_id": "cid"}), on="cand")
                  .sort("s1", "cid").group_by("s1").agg(pl.col("cid").str.join(",").alias(col)))
    return (s1_all.join(named, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
                  .select(pl.col("entity_id").alias("source1_entity_id"), col))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="decision params json (default work/decision_params.json)")
    ap.add_argument("--tag", default="", help="suffix for leaderboard probe variants")
    ap.add_argument("--score-col", default="p2")
    args = ap.parse_args(argv)

    params = json.load(open(args.params)) if args.params else load_params()
    rec = load_records("test", ["idx", "entity_id", "src", "country"])
    ids = rec.select("idx", "entity_id")
    s1_all = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "entity_id")

    sc = pl.read_parquet(wpath("s2_test.parquet" if args.score_col == "p2" else "s1_test.parquet"))
    sc = sc.select("s1", "cand", pl.col(args.score_col).alias("p")).join(
        rec.select(pl.col("idx").alias("s1"), "country"), on="s1")
    matches = decide(sc, params)
    cands = pl.read_parquet(wpath("pruned_test.parquet"), columns=["s1", "cand"])
    assert matches.join(cands, on=["s1", "cand"], how="anti").height == 0, "match outside candidate set"

    suffix = f"_{args.tag}" if args.tag else ""
    # each run writes its own candidate file: the pruned set (and so the candidates) can differ between runs
    m_path, c_path = OUT / f"matching_results{suffix}.tsv", OUT / f"candidate_pairs{suffix}.tsv"
    _lists(matches, ids, s1_all, "matched_entity_ids").write_csv(m_path, separator="\t", quote_style="never")
    _lists(cands, ids, s1_all, "candidate_entity_ids").write_csv(c_path, separator="\t", quote_style="never")
    n_match = matches.height
    empty = s1_all.height - matches["s1"].n_unique()
    print(f"wrote {m_path.name}: {n_match:,} matches, {empty:,} empty S1 of {s1_all.height:,}")
    by_c = matches.join(rec.select(pl.col("idx").alias("s1"), "country"), on="s1").group_by("country").len()
    print("   matches by country:", {r["country"]: r["len"] for r in by_c.to_dicts()})
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(m_path), "--candidate", str(c_path),
                        "--test-dir", str(DATA / "test")], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
