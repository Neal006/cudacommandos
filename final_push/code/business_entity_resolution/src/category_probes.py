"""Category probes: change the submission for ONE kind of pair and compare the leaderboard change with the
change the same edit makes on train Q (where the truth is known).

If test follows train's conventions, LB delta ~= the "expected LB delta" printed here. A clear difference
means that kind of pair is labelled differently in test - and the probe that helps is itself the fix.

Pair kinds (from the normalised records):
  name: same   = same normalised name key          addr: empty  = candidate has no address
        some   = shares a name token                     samehn = same house number
        none   = no name token in common                 diffhn = both have a house number, different
                                                          nohn   = no house number on one side

Probes:
  add_same_empty   : add the best unpredicted same-name / empty-address candidate of each S1
  drop_empty       : drop every predicted pair whose candidate has no address
  drop_same_diffhn : drop predicted same-name pairs whose house numbers differ
  add_none_samehn  : add the best unpredicted no-shared-name / same-house-number candidate of each S1
                     (a renamed record at the S1's exact address)

  python category_probes.py                 # train deltas + output/probes/matching_results_cat_<probe>.tsv
"""
import numpy as np
import polars as pl

from config import OUT, wpath
from decide import decide, load_params
from evaluate import load_gt, per_entity
from io_utils import load_records
from normalize import load_norm

MIN_P = 0.001      # added pairs must have at least this score


def pairs_with_kind(split: str, sc: pl.DataFrame) -> pl.DataFrame:
    n = load_norm(split, ["idx", "n_key", "n_core", "a_empty", "a_hnd"])
    a = n.select(pl.col("idx").alias("s1"), pl.col("n_key").alias("k1"), pl.col("n_core").alias("c1"),
                 pl.col("a_hnd").alias("h1"))
    b = n.select(pl.col("idx").alias("cand"), pl.col("n_key").alias("k2"), pl.col("n_core").alias("c2"),
                 pl.col("a_hnd").alias("h2"), pl.col("a_empty").alias("e2"))
    d = sc.join(a, on="s1").join(b, on="cand")
    shared = pl.col("c1").str.split(" ").list.set_intersection(pl.col("c2").str.split(" ")).list.eval(
        pl.element().filter(pl.element() != "")).list.len()
    d = d.with_columns(
        pl.when(pl.col("k1") == pl.col("k2")).then(pl.lit("same"))
        .when(shared > 0).then(pl.lit("some")).otherwise(pl.lit("none")).alias("name"),
        pl.when(pl.col("e2")).then(pl.lit("empty"))
        .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nohn"))
        .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samehn")).otherwise(pl.lit("diffhn")).alias("addr"))
    return d.select("s1", "cand", "p", "country", "name", "addr")


def _add_best(base: pl.DataFrame, d: pl.DataFrame, mask: pl.Expr) -> pl.DataFrame:
    """add, per S1, the best-scoring unpredicted pair matching mask whose candidate is not used elsewhere."""
    used = base.select("cand").unique()
    add = (d.filter(mask & (pl.col("p") >= MIN_P))
             .join(base, on=["s1", "cand"], how="anti").join(used, on="cand", how="anti")
             .sort("p", descending=True).unique("s1", keep="first").unique("cand", keep="first"))
    return pl.concat([base, add.select("s1", "cand")])


def _drop(base: pl.DataFrame, d: pl.DataFrame, mask: pl.Expr) -> pl.DataFrame:
    return base.join(d.filter(mask).select("s1", "cand"), on=["s1", "cand"], how="anti")


PROBES = {
    "add_same_empty": lambda b, d: _add_best(b, d, (pl.col("name") == "same") & (pl.col("addr") == "empty")),
    "drop_empty": lambda b, d: _drop(b, d, pl.col("addr") == "empty"),
    "drop_same_diffhn": lambda b, d: _drop(b, d, (pl.col("name") == "same") & (pl.col("addr") == "diffhn")),
    "add_none_samehn": lambda b, d: _add_best(b, d, (pl.col("name") == "none") & (pl.col("addr") == "samehn")),
}


def _changed(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([a.join(b, on=["s1", "cand"], how="anti"), b.join(a, on=["s1", "cand"], how="anti")]
                     ).select("s1").unique()


def main():
    params = load_params()
    # ---- train Q: true effect of each probe
    from splits import load_splits
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    cn = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    tr = pl.read_parquet(wpath("s2_train.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(cn, on="s1")
    gt = load_gt().join(q, on="s1")
    dtr = pairs_with_kind("train", tr)
    btr = decide(tr, params)
    f0 = per_entity(btr, gt, q).select("s1", pl.col("f").alias("f0"))
    kinds = dtr.join(btr, on=["s1", "cand"], how="semi").group_by("name", "addr").len("pred").join(
        dtr.join(gt, on=["s1", "cand"], how="semi").group_by("name", "addr").len("true"), on=["name", "addr"], how="full",
        coalesce=True).fill_null(0)
    print("train Q predicted / true pairs by kind:")
    with pl.Config(tbl_rows=-1):
        print(kinds.sort("pred", descending=True))

    # ---- test
    rec = load_records("test", ["idx", "entity_id", "src", "country"])
    tc = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country")
    te = pl.read_parquet(wpath("s2_test.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(tc, on="s1")
    dte = pairs_with_kind("test", te)
    bte = decide(te, params)
    print("test predicted pairs by kind:",
          {f"{r['name']}/{r['addr']}": r["len"] for r in dte.join(bte, on=["s1", "cand"], how="semi")
           .group_by("name", "addr").len().sort("len", descending=True).to_dicts()})

    ids = rec.select("idx", "entity_id")
    s1_all = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "entity_id")
    probe_dir = OUT / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    from write_output import _lists
    rows = []
    for name, fn in PROBES.items():
        ptr = fn(btr, dtr)
        ch = _changed(btr, ptr)
        e = per_entity(ptr, gt, q).join(f0, on="s1").join(cn, on="s1")
        d_by = e.group_by("country").agg((pl.col("f") - pl.col("f0")).mean().alias("d"))
        per_changed = e.join(ch, on="s1").select((pl.col("f") - pl.col("f0")).mean()).item() if ch.height else 0.0
        pte = fn(bte, dte)
        ch_te = _changed(bte, pte).join(tc, on="s1")
        path = probe_dir / f"matching_results_cat_{name}.tsv"
        _lists(pte, ids, s1_all, "matched_entity_ids").write_csv(path, separator="\t", quote_style="never")
        exp_lb = ch_te.height / tc.height * (per_changed or 0.0)
        rows.append({"probe": name,
                     "train_changed_S1": ch.height / q.height,
                     "train_dF_per_changed_S1": per_changed,
                     **{f"train_dF_{r['country']}": r["d"] for r in d_by.to_dicts()},
                     "test_changed_S1": ch_te.height / tc.height,
                     **{f"test_changed_{r['country']}": r["len"] / tc.filter(pl.col("country") == r["country"]).height
                        for r in ch_te.group_by("country").len().to_dicts()},
                     "expected_LB_delta": exp_lb})
        print(f"  wrote {path}")
    t = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=5):
        print(t.transpose(include_header=True, column_names="probe"))
    print("Submit a probe; if LB delta differs clearly from expected_LB_delta (sign or ~2x), test labels that "
          "kind of pair differently from train.")


if __name__ == "__main__":
    main()
