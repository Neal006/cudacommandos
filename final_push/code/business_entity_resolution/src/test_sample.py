"""Local test-set evaluation without leaderboard submissions.

  expected : label-free. Expected macro F0.5 per country from the calibrated p2 scores, on train Q (where the
             true score is known, so it checks the estimate) and on test. If test's expected score is much
             higher than the leaderboard, p2 is over-confident on test (e.g. more lookalike distractors).
  shift    : label-free. Train Q vs test per country: pool size, predicted matches per S1 and per source,
             duplicate S1 names, contested candidates, score buckets (with train precision per bucket).
  sample   : draws a uniform random sample of test S1 entities per country and writes a *blind* labelling
             sheet (no scores, no predictions, candidates shuffled) with each S1's top candidates.
  eval     : scores any number of submission files on the labelled sample: macro F0.5 per country with
             bootstrap CIs, precision / recall / singleton rate, and paired differences vs the first file.

  python test_sample.py expected
  python test_sample.py shift
  python test_sample.py sample --n 300                     # -> work/test_sample/sheet.tsv
  python test_sample.py sample --n 100 --split train        # -> sheet_train.tsv, ground truth filled in
  #   fill the `label` column: 1 = same business, 0 = different, ? = unsure (rows with no candidate: leave blank)
  python test_sample.py eval work/test_sample/sheet.tsv ../../../output/matching_results.tsv \
                        ../../../output/matching_results_probe1.tsv

Recall and the singleton rate are measured against the candidates on the sheet: a true match that
blocking / pruning never proposed is invisible here (train blocking recall is ~99.9%, so this is small).
With n=300 per country the per-country CI is roughly +-0.01; paired differences between two submissions
are much tighter because both are scored on the same entities.
"""
import argparse
import sys

import numpy as np
import polars as pl
from numba import njit, prange

from config import SEED, wpath
from decide import decide, load_params
from evaluate import per_entity
from io_utils import load_records

SHEET_DIR = wpath("test_sample")


# ---------------------------------------------------------------------------------------------- expected
@njit(parallel=True, cache=True)
def _expected_f(offsets, p, sel, n_samples, seed):
    """per group: E[F0.5] of the selected rows when each row is a true match independently with prob p."""
    ng = len(offsets) - 1
    out = np.ones(ng)
    for g in prange(ng):
        a, b = offsets[g], offsets[g + 1]
        np.random.seed(seed + g)
        k = 0
        for i in range(a, b):
            k += sel[i]
        acc = 0.0
        for _ in range(n_samples):
            t, tp = 0, 0
            for i in range(a, b):
                if np.random.random() < p[i]:
                    t += 1
                    tp += sel[i]
            acc += 1.0 if t == 0 and k == 0 else 1.25 * tp / (0.25 * t + k)
        out[g] = acc / n_samples
    return out


def expected_by_country(sc: pl.DataFrame, pred: pl.DataFrame, s1_country: pl.DataFrame, n_samples=512):
    """sc: s1, cand, p (calibrated); pred: selected (s1, cand); s1_country: every S1 of the population."""
    d = (sc.filter(pl.col("p") >= 1e-3)
           .join(pred.with_columns(pl.lit(1, pl.Int8).alias("sel")), on=["s1", "cand"], how="left")
           .with_columns(pl.col("sel").fill_null(0)).sort("s1"))
    # predicted pairs below the floor still count as predictions
    d = pl.concat([d, pred.join(d, on=["s1", "cand"], how="anti")
                   .with_columns(pl.lit(0.0, d["p"].dtype).alias("p"), pl.lit(1, pl.Int8).alias("sel"))
                   .select(d.columns)]).sort("s1")
    s1 = d["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]]) if len(s1) else np.zeros(0, np.int64)
    offsets = np.r_[starts, len(s1)].astype(np.int64)
    ef = _expected_f(offsets, d["p"].to_numpy().astype(np.float64), d["sel"].to_numpy().astype(np.int64),
                     n_samples, SEED)
    e = pl.DataFrame({"s1": s1[starts], "ef": ef}).cast({"s1": s1_country["s1"].dtype})
    # S1s without any scored candidate: predicted empty, assumed truly empty
    e = s1_country.join(e, on="s1", how="left").with_columns(pl.col("ef").fill_null(1.0))
    return e.group_by("country").agg(pl.col("ef").mean(), pl.len()).sort("country"), e["ef"].mean()


def cmd_expected(args):
    from evaluate import load_gt, macro_f05
    from normalize import load_norm
    from splits import load_splits
    params = load_params()

    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    cn = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    qc = q.join(cn, on="s1")
    tr = pl.read_parquet(wpath("s2_train.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(cn, on="s1")
    pred = decide(tr, params)
    by, tot = expected_by_country(tr.select("s1", "cand", "p"), pred, qc)
    gtq = load_gt().join(q, on="s1")
    act = per_entity(pred, gtq, q).join(cn, on="s1").group_by("country").agg(pl.col("f").mean()).sort("country")
    print(f"train Q  expected {tot:.5f}   actual {macro_f05(pred, gtq, q):.5f}")
    print("   expected by country:", {r["country"]: round(r["ef"], 5) for r in by.to_dicts()})
    print("   actual   by country:", {r["country"]: round(r["f"], 5) for r in act.to_dicts()})

    rec = load_records("test", ["idx", "src", "country"])
    tc = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country")
    te = pl.read_parquet(wpath("s2_test.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(tc, on="s1")
    pred = decide(te, params)
    by, tot = expected_by_country(te.select("s1", "cand", "p"), pred, tc)
    print(f"test     expected {tot:.5f}   (compare with the leaderboard)")
    print("   expected by country:", {r["country"]: round(r["ef"], 5) for r in by.to_dicts()})
    emp = tc.join(pred.select("s1").unique().with_columns(pl.lit(True).alias("m")), on="s1", how="left")
    print("   predicted-empty share:", {r["country"]: round(r["e"], 4) for r in emp.group_by("country").agg(
        pl.col("m").is_null().mean().alias("e")).sort("country").to_dicts()})


# ---------------------------------------------------------------------------------------------- shift
def _shift_stats(split: str, sc: pl.DataFrame, pred: pl.DataFrame, s1c: pl.DataFrame) -> pl.DataFrame:
    """one row per country: data-structure and score-structure statistics of one split."""
    from normalize import load_norm
    norm = load_norm(split, ["idx", "src", "country", "n_key", "a_city", "a_hnd"])
    pool = norm.filter(pl.col("src") != 1).group_by("country").len("n_pool")
    s1n = norm.join(s1c.select(pl.col("s1").alias("idx")), on="idx")
    # S1 entities sharing a normalised name with another S1 (chains / branches) in the same country or city
    s1n = s1n.with_columns(
        (pl.len().over("country", "n_key") > 1).alias("dup_name"),
        (pl.len().over("country", "n_key", "a_city") > 1).alias("dup_name_city"))
    src = norm.select(pl.col("idx").alias("cand"), "src")
    top = sc.group_by("s1").agg(pl.col("p").max().alias("top"), (pl.col("p") > 0.5).sum().alias("n05"))
    cp = sc.filter(pl.col("p") > 0.1).group_by("cand").agg(
        pl.col("p").max().alias("c1"), pl.col("p").top_k(2).min().alias("c2"), pl.len().alias("nc"))
    pr = pred.join(src, on="cand").join(sc.select("s1", "cand", "p"), on=["s1", "cand"], how="left")
    per = (s1c.join(top, on="s1", how="left")
              .join(pr.group_by("s1").agg(pl.len().alias("np"), (pl.col("src") == 2).sum().alias("np2"),
                                          (pl.col("src") == 3).sum().alias("np3"),
                                          (pl.col("p") < 0.9).sum().alias("np_lt09")), on="s1", how="left")
              .join(s1n.select(pl.col("idx").alias("s1"), "dup_name", "dup_name_city"), on="s1", how="left")
              .with_columns(pl.col("np", "np2", "np3", "np_lt09", "n05").fill_null(0), pl.col("top").fill_null(0.0)))
    contested = (pr.join(cp, on="cand").with_columns(((pl.col("nc") > 1) & (pl.col("c2") > 0.1)).alias("_ct"))
                   .join(s1c, on="s1").group_by("country").agg(pl.col("_ct").mean().alias("pred_contested")))
    out = per.group_by("country").agg(
        pl.len().alias("n_s1"),
        pl.col("np").mean().alias("pred_per_s1"), pl.col("np2").mean().alias("pred_s2"),
        pl.col("np3").mean().alias("pred_s3"), (pl.col("np2") > 1).mean().alias("multi_s2"),
        (pl.col("np3") > 1).mean().alias("multi_s3"), (pl.col("np") == 0).mean().alias("pred_empty"),
        (pl.col("np_lt09").sum() / pl.col("np").sum()).alias("pred_p_lt_0.9"),
        pl.col("top").is_between(0.1, 0.9).mean().alias("top_uncertain"),
        pl.col("n05").mean().alias("pairs_p05"),
        pl.col("dup_name").mean().alias("s1_dup_name"), pl.col("dup_name_city").mean().alias("s1_dup_name_city"),
    ).join(contested, on="country", how="left").join(pool, on="country", how="left")
    s1_all = norm.filter(pl.col("src") == 1).group_by("country").len("n_s1_all")
    return (out.join(s1_all, on="country").with_columns((pl.col("n_pool") / pl.col("n_s1_all")).alias("pool_per_s1"))
               .drop("n_pool", "n_s1_all").with_columns(pl.lit(split).alias("split")))


def cmd_shift(args):
    """train Q vs test, per country: where does test look different from what the model was validated on?"""
    from evaluate import load_gt
    from normalize import load_norm
    from splits import load_splits
    params = load_params()
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    cn = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    qc = q.join(cn, on="s1")
    tr = pl.read_parquet(wpath("s2_train.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(cn, on="s1")
    ptr = decide(tr, params)
    tn = load_norm("test", ["idx", "src", "country"])
    tc = tn.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1"), "country")
    te = pl.read_parquet(wpath("s2_test.parquet"), columns=["s1", "cand", "p2"]).rename({"p2": "p"}).join(tc, on="s1")
    pte = decide(te, params)
    t = pl.concat([_shift_stats("train", tr, ptr, qc), _shift_stats("test", te, pte, tc)])
    cols = [c for c in t.columns if c not in ("split", "country")]
    t = t.sort("country", "split").select("country", "split", *cols).with_columns(pl.col(cols).cast(pl.Float64).round(4))
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=4):
        print(t.transpose(include_header=True, column_names=[f"{r['country']}/{r['split']}" for r in t.to_dicts()])
               .filter(~pl.col("column").is_in(["country", "split"])))
    # train reference: how precise are the predictions in each bucket that can shift?
    gt = load_gt().join(q, on="s1").with_columns(pl.lit(1, pl.Int8).alias("y"))
    ev = ptr.join(tr, on=["s1", "cand"]).join(gt, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0))
    b = ev.with_columns(pl.col("p").cut([0.5, 0.7, 0.9, 0.97, 0.99]).alias("p_bucket")).group_by("p_bucket").agg(
        pl.len().alias("pred_pairs"), pl.col("y").mean().alias("train_precision")).sort("p_bucket")
    te_b = pte.join(te, on=["s1", "cand"]).with_columns(pl.col("p").cut([0.5, 0.7, 0.9, 0.97, 0.99]).alias("p_bucket")
                                                        ).group_by("p_bucket").len("test_pred_pairs")
    print(b.join(te_b, on="p_bucket", how="left").with_columns(
        (pl.col("pred_pairs") / pl.col("pred_pairs").sum()).alias("train_share"),
        (pl.col("test_pred_pairs") / pl.col("test_pred_pairs").sum()).alias("test_share")))


# ---------------------------------------------------------------------------------------------- sample
def cmd_sample(args):
    """test: blind sheet to label. train (--split train): same format from the query set Q, with the ground
    truth in `label` plus true matches the sheet does not show (shown=0) - to learn the labelling conventions
    and to measure a labeller's accuracy before trusting their test labels."""
    split = args.split
    rec = load_records(split, ["idx", "entity_id", "src", "name", "addr", "country"])
    s1 = rec.filter(pl.col("src") == 1)
    if split == "train":
        from splits import load_splits
        s1 = s1.join(load_splits().filter(~pl.col("hidden")).select("idx"), on="idx")
    rng = np.random.default_rng(args.seed)
    pick = (s1.with_columns(pl.Series("_r", rng.random(s1.height)))
              .filter(pl.col("_r").rank("ordinal").over("country") <= args.n)
              .select(pl.col("idx").alias("s1"), pl.col("entity_id").alias("s1_id"), "country",
                      pl.col("name").alias("s1_name"), pl.col("addr").alias("s1_addr")))
    sc = pl.read_parquet(wpath(f"s2_{split}.parquet"), columns=["s1", "cand", "p2"]).join(pick.select("s1"), on="s1")
    # top candidates: rank <= 3 always, then anything with p2 >= min_p, at most max_c per S1
    sc = (sc.with_columns(pl.col("p2").rank("ordinal", descending=True).over("s1").alias("_rk"))
            .filter((pl.col("_rk") <= 3) | (pl.col("p2") >= args.min_p)).filter(pl.col("_rk") <= args.max_c)
            .select("s1", "cand", pl.lit(1, pl.Int8).alias("shown")))
    label = pl.lit("")
    if split == "train":
        from evaluate import load_gt
        gt = load_gt().join(pick.select("s1"), on="s1").with_columns(pl.lit("1").alias("_y"))
        sc = pl.concat([sc, gt.join(sc, on=["s1", "cand"], how="anti").select(
            "s1", "cand", pl.lit(0, pl.Int8).alias("shown"))])
        sc = sc.join(gt, on=["s1", "cand"], how="left")
        label = pl.col("_y").fill_null("0")
    cands = rec.select(pl.col("idx").alias("cand"), pl.col("entity_id").alias("cand_id"),
                       pl.col("src").alias("source"), pl.col("name").alias("cand_name"),
                       pl.col("addr").alias("cand_addr"))
    rows = sc.join(cands, on="cand").with_columns(pl.Series("_o", rng.random(sc.height)))
    cols = ["s1_id", "country", "s1_name", "s1_addr", "cand_id", "source", "cand_name", "cand_addr"]
    sheet = (pick.join(rows, on="s1", how="left").sort("country", "s1_id", "_o")
                 .with_columns(label.alias("label")))
    if split == "train":
        sheet = sheet.with_columns(pl.when(pl.col("cand_id").is_null()).then(pl.lit("")).otherwise(pl.col("label"))
                                   .alias("label")).select(*cols, "label", "shown")
    else:
        sheet = sheet.select(*cols, "label")
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    path = SHEET_DIR / ("sheet.tsv" if split == "test" else "sheet_train.tsv")
    sheet.write_csv(path, separator="\t", quote_style="never")
    n_rows = sheet.filter(pl.col("cand_id").is_not_null()).height
    print(f"wrote {path}: {pick.height:,} S1 ({args.n} per country), {n_rows:,} pairs "
          f"({n_rows / pick.height:.1f} per S1); {pick.height - sheet['s1_id'].filter(sheet['cand_id'].is_not_null()).n_unique()} S1 have no candidate")


# ---------------------------------------------------------------------------------------------- eval
def _read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def _bootstrap(f: np.ndarray, n=2000, seed=SEED):
    if len(f) == 0:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(f, (n, len(f))).mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def cmd_eval(args):
    lab = _read_tsv(args.sheet).with_columns(pl.col("label").fill_null("").str.strip_chars())
    s1s = lab.select(pl.col("s1_id").alias("s1"), "country").unique()
    pairs = lab.filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != ""))
    todo = pairs.filter(~pl.col("label").is_in(["0", "1"]))
    if todo.height:
        print(f"!! {todo.height:,} pairs are unlabelled or '?': they count as non-matches "
              f"({todo['s1_id'].n_unique()} S1 affected)")
    gt = pairs.filter(pl.col("label") == "1").select(pl.col("s1_id").alias("s1"), pl.col("cand_id").alias("cand"))
    known = pairs.select(pl.col("s1_id").alias("s1"), pl.col("cand_id").alias("cand"))

    rec = load_records("test", ["src", "country"]).filter(pl.col("src") == 1)
    w = {r["country"]: r["len"] / rec.height for r in rec.group_by("country").len().to_dicts()}
    countries = sorted(s1s["country"].unique().to_list())
    nt = s1s.join(gt.group_by("s1").len("nt"), on="s1", how="left")
    print(f"{s1s.height:,} S1 in the sample; labelled singleton rate:",
          {c: round(nt.filter(pl.col("country") == c)["nt"].is_null().mean(), 4) for c in countries},
          " (train: 0.0558)")

    base = None
    for path in args.submissions:
        sub = (_read_tsv(path).join(s1s, left_on="source1_entity_id", right_on="s1", how="semi")
                 .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
                 .explode("matched_entity_ids", empty_as_null=True).filter(pl.col("matched_entity_ids") != "")
                 .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("cand")))
        off = sub.join(known, on=["s1", "cand"], how="anti").height
        e = per_entity(sub, gt, s1s).join(s1s, on="s1").sort("s1")
        print(f"\n== {path}" + (f"   ({off} predicted pairs not on the sheet: counted as wrong)" if off else ""))
        tot = 0.0
        for c in countries:
            ec = e.filter(pl.col("country") == c)
            lo, hi = _bootstrap(ec["f"].to_numpy())
            tp, npred, ntrue = ec["tp"].sum(), ec["np"].sum(), ec["nt"].sum()
            print(f"   {c:8s} F0.5 {ec['f'].mean():.4f} [{lo:.4f}, {hi:.4f}]   P {tp / max(npred, 1):.4f} "
                  f"R {tp / max(ntrue, 1):.4f}   false merges on singletons: "
                  f"{ec.filter((pl.col('nt') == 0) & (pl.col('np') > 0)).height}/{ec.filter(pl.col('nt') == 0).height}")
            tot += w.get(c, 0.0) * ec["f"].mean()
        print(f"   overall (weighted by test country share) {tot:.4f}")
        if base is None:
            base = e
            continue
        d = e.join(base.select("s1", pl.col("f").alias("f0")), on="s1")
        for c in countries:
            dc = d.filter(pl.col("country") == c)
            diff = (dc["f"] - dc["f0"]).to_numpy()
            lo, hi = _bootstrap(diff)
            print(f"   vs first  {c:8s} {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"({int((diff != 0).sum())} S1 changed)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("expected")
    sp.add_parser("shift")
    s = sp.add_parser("sample")
    s.add_argument("--n", type=int, default=300, help="S1 entities per country")
    s.add_argument("--max-c", type=int, default=10, help="max candidates shown per S1")
    s.add_argument("--min-p", type=float, default=0.01, help="also show candidates with p2 >= this")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--split", default="test", choices=["test", "train"],
                   help="train: labelled calibration sheet from the query set (work/test_sample/sheet_train.tsv)")
    e = sp.add_parser("eval")
    e.add_argument("sheet")
    e.add_argument("submissions", nargs="+")
    a = ap.parse_args(argv)
    {"expected": cmd_expected, "shift": cmd_shift, "sample": cmd_sample, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
