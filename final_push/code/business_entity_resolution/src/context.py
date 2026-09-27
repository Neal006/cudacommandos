"""Stage F: context features from stage-1 scores (+ cross-encoder if available) -> stage-2 LightGBM
-> isotonic calibration -> decision tuning on OOF -> test predictions.

Context features:
  within-S1   : rank, gap to best, share, expected cluster size, same-source counts
  competition : the same candidate's best score for any *other* S1 (one-to-one structure)
  agreement   : similarity of this candidate to the S1's other top-3 candidates, weighted by their p1
                (S2 <-> S3 agreement rescues records whose house number disagrees only with S1)
  twin        : a same-name candidate with a different house number scores higher for this S1
  CE          : cross-encoder logit, its rank / gap / competition margin (NaN outside the CE band)
"""
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.isotonic import IsotonicRegression

from config import wpath
from decide import save_params, tune, decide
from evaluate import load_gt, report
from features import Ctx, _cp, _setfeats
from gbdt import cv_train
from splits import load_splits
from stage1 import feature_cols


def _within(df: pl.DataFrame, p: str, pre: str) -> pl.DataFrame:
    g = pl.col(p)
    return df.with_columns(
        g.rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias(f"{pre}_rank"),
        (g - g.max().over("s1")).alias(f"{pre}_gap"),
        (g / g.sum().over("s1")).alias(f"{pre}_share"),
        g.sum().over("s1").alias(f"{pre}_sum"),
        (g > 0.5).sum().over("s1").cast(pl.Float32).alias(f"{pre}_n05"),
        # competition across S1s for the same candidate
        g.rank("ordinal", descending=True).over("cand").cast(pl.Float32).alias(f"{pre}_crank"),
        g.top_k(2).over("cand", mapping_strategy="join").alias("_t"),
        (g > 0.5).sum().over("cand").cast(pl.Float32).alias(f"{pre}_cn05"),
    ).with_columns(
        pl.when(pl.col(f"{pre}_crank") == 1).then(pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0))
        .otherwise(pl.col("_t").list.get(0)).alias(f"{pre}_other"),
    ).with_columns((g - pl.col(f"{pre}_other")).alias(f"{pre}_margin")).drop("_t")


def agreement(ctx: Ctx, df: pl.DataFrame, chunk=4_000_000) -> pl.DataFrame:
    top = (df.filter(pl.col("p1_rank") <= 3)
             .select("s1", pl.col("cand").alias("cj"), pl.col("p1").alias("pj"), pl.col("src_b").alias("srcj")))
    pairs = (df.select("s1", "cand", "src_b").join(top, on="s1").filter(pl.col("cand") != pl.col("cj")))
    parts = []
    for i in range(0, pairs.height, chunk):
        c = pairs.slice(i, chunk)
        ia, ib = c["cand"].to_numpy().astype(np.int64), c["cj"].to_numpy().astype(np.int64)
        at = np.nan_to_num(_setfeats(ctx, "at", ia, ib, "at", False)["at_jacc"])
        nm = _cp(ctx.col("n_core", ia).to_list(), ctx.col("n_core", ib).to_list(), fuzz.token_set_ratio) / 100
        st = _cp(ctx.col("a_street", ia).to_list(), ctx.col("a_street", ib).to_list(), fuzz.token_set_ratio) / 100
        ha, hb = ctx.col("a_hnd", ia), ctx.col("a_hnd", ib)
        hn = ((ha == hb) & (ha != "")).to_numpy().astype(np.float32)
        parts.append(c.with_columns(pl.Series("at", at), pl.Series("nm", nm), pl.Series("st", st),
                                    pl.Series("hn", hn)))
    pairs = pl.concat(parts)
    pj, opp = pl.col("pj"), pl.col("srcj") != pl.col("src_b")
    agg = pairs.group_by("s1", "cand").agg(
        (pj * pl.col("at")).max().alias("ag_at"), (pj * pl.col("nm")).max().alias("ag_nm"),
        (pj * pl.col("st")).max().alias("ag_st"), (pj * pl.col("hn")).max().alias("ag_hn"),
        (pj * pl.col("at")).filter(opp).max().alias("ag_opp_at"),
        (pj * pl.col("hn")).filter(opp).max().alias("ag_opp_hn"),
        pl.col("at").filter(pj > 0.5).mean().alias("ag_at_mean05"),
    )
    return df.join(agg, on=["s1", "cand"], how="left")


def twins(ctx: Ctx, df: pl.DataFrame) -> pl.DataFrame:
    ib = df["cand"].to_numpy().astype(np.int64)
    d = df.with_columns(ctx.col("n_key", ib).alias("_k"), ctx.col("a_hnd", ib).alias("_h"))
    top_h = pl.col("_h").sort_by("p1", descending=True).first().over("s1", "_k")
    top_p = pl.col("p1").max().over("s1", "_k")
    d = d.with_columns(
        pl.len().over("s1", "_k").cast(pl.Float32).alias("tw_n"),
        ((top_p > pl.col("p1")) & (top_h != pl.col("_h")) & (top_h != "") & (pl.col("_h") != ""))
        .cast(pl.Float32).alias("tw_flag"),
        (top_p - pl.col("p1")).alias("tw_gap"),
    )
    return d.drop("_k", "_h")


def add_ce(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """Cross-encoder logits: work/ce_{split}.parquet (and optional extra models ce2_{split}, ce3_{split} ...);
    several models' raw logits are averaged (no per-split normalisation: it would shift test vs train)."""
    sel = os.environ.get("ER_CE")
    if sel is not None:
        paths = [wpath(f"{x}_{split}.parquet") for x in sel.split(",") if x]
    else:
        paths = sorted(p for p in wpath("").glob(f"ce*_{split}.parquet") if p.name.split("_")[0].startswith("ce"))
    if not paths:
        return df
    parts = []
    for k, path in enumerate(paths):
        c = pl.read_parquet(path)
        parts.append(c.select("pair_id", pl.col("ce").alias(f"ce{k}")))
    ce = parts[0]
    for c in parts[1:]:
        ce = ce.join(c, on="pair_id", how="full", coalesce=True)
    ce = ce.select("pair_id", pl.mean_horizontal([f"ce{k}" for k in range(len(parts))]).alias("ce"))
    print(f"  cross-encoder scores for {split}: {[p.name for p in paths]}")
    ce = ce.with_columns((pl.col("pair_id") // (1 << 32)).cast(pl.Int32).alias("s1"),
                         (pl.col("pair_id") % (1 << 32)).cast(pl.Int32).alias("cand")).drop("pair_id")
    df = df.join(ce, on=["s1", "cand"], how="left")
    g = pl.col("ce")
    return df.with_columns(
        g.rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("ce_rank"),
        (g - g.max().over("s1")).alias("ce_gap"),
        (g - g.filter(g.is_not_null()).max().over("cand")).alias("ce_cgap"),
    )


def add_pseudo(tr: pl.DataFrame, te: pl.DataFrame, path: str, weight=0.5, neg_rate=0.05, seed=2026):
    """Self-training rows from confident test decisions of a previous run (fold=-1: always training data)."""
    prev = pl.read_parquet(path).select("s1", "cand", pl.col("p2").alias("prev"))
    prev = prev.with_columns(
        pl.col("prev").rank("ordinal", descending=True).over("cand").alias("_r"),
        pl.col("prev").top_k(2).over("cand", mapping_strategy="join").list.get(1, null_on_oob=True)
        .fill_null(0.0).alias("_second"))
    pos = prev.filter((pl.col("prev") > 0.995) & (pl.col("_r") == 1) & (pl.col("prev") - pl.col("_second") > 0.5))
    neg = prev.filter(pl.col("prev") < 0.005).sample(fraction=neg_rate, seed=seed)
    lab = pl.concat([pos.select("s1", "cand", pl.lit(1, pl.Int8).alias("y")),
                     neg.select("s1", "cand", pl.lit(0, pl.Int8).alias("y"))])
    # each sampled negative stands for 1/neg_rate negatives
    rows = te.join(lab, on=["s1", "cand"]).with_columns(
        pl.lit(-1, pl.Int8).alias("fold"),
        pl.when(pl.col("y") == 0).then(weight / neg_rate).otherwise(weight).cast(pl.Float32).alias("w"))
    print(f"  pseudo-labels from {path}: {pos.height:,} pos, {neg.height:,} neg (weight {weight})")
    tr = tr.with_columns(pl.lit(1.0, pl.Float32).alias("w"), pl.col("fold").cast(pl.Int8), pl.col("y").cast(pl.Int8))
    return pl.concat([tr, rows.select(tr.columns)], how="vertical_relaxed")


def context_features(split: str) -> pl.DataFrame:
    t0 = time.time()
    df = pl.read_parquet(wpath(f"s1_{split}.parquet"))
    ctx = Ctx(split)
    df = _within(df, "p1", "p1")
    df = df.with_columns(((pl.col("p1") > 0.5).sum().over("s1", "src_b")).cast(pl.Float32).alias("p1_n05_src"),
                         (pl.col("p1") / (1 - pl.col("p1").clip(0, 0.999999))).log().alias("p1_logit"))
    df = agreement(ctx, df)
    df = twins(ctx, df)
    df = add_ce(df, split)
    print(f"  context {split}: {df.height:,} rows, {time.time() - t0:.0f}s", flush=True)
    return df


def main():
    t0 = time.time()
    tr = context_features("train")
    te = context_features("test")
    feats = feature_cols(te) + ["p1"]
    print(f"stage-2: {len(feats)} features")
    n_tr = tr.height
    pseudo_src = os.environ.get("ER_PSEUDO")      # path to a previous s2_test.parquet -> self-training rows
    if pseudo_src:
        tr = add_pseudo(tr, te, pseudo_src)
    algos = os.environ.get("ER_STAGE2", "lgb").split("+")
    oof, preds = np.zeros(tr.height, np.float32), {"test": np.zeros(te.height, np.float32)}
    for algo in algos:
        o, pr, _, imp = cv_train(tr, feats, neg_sample=(pl.col("p1") < 0.005, 0.25),
                                 predict={"test": te.select(feats)}, log="stage2", algo=algo,
                                 weight_col="w" if "w" in tr.columns else None)
        oof += o / len(algos)
        preds["test"] += pr["test"] / len(algos)
    print(imp.head(25))
    imp.write_csv(wpath("imp_stage2.csv"))
    oof, tr = oof[:n_tr], tr.slice(0, n_tr)            # pseudo rows are training-only, never evaluated
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, tr["y"].to_numpy())
    tr = tr.select("s1", "cand", "y", "fold", "p1").with_columns(pl.Series("p2", iso.predict(oof).astype(np.float32)))
    te = te.select("s1", "cand", "p1").with_columns(pl.Series("p2", iso.predict(preds["test"]).astype(np.float32)))
    tr.write_parquet(wpath("s2_train.parquet"))
    te.write_parquet(wpath("s2_test.parquet"))
    tune_and_report()
    print(f"stage-2 done in {time.time() - t0:.0f}s")


def tune_and_report(col="p2"):
    from normalize import load_norm
    tr = pl.read_parquet(wpath("s2_train.parquet"))
    gt = load_gt()
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    gtq = gt.join(q, on="s1")
    country = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    d = tr.select("s1", "cand", pl.col(col).alias("p")).join(country, on="s1")
    params, table = tune(d, gtq, q)
    table.write_csv(wpath(f"decision_grid_{col}.csv"))
    save_params(params)
    print("best decision params:", params)
    report(decide(d, params), gtq, q, f"Q OOF ({col})")


if __name__ == "__main__":
    main()
