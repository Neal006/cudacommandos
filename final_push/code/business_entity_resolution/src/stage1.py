"""Stage E: full pairwise features on the pruned set + stage-1 LightGBM (5-fold OOF on Q).

Writes full_{split}.parquet: s1, cand, all features, p1 (train: Q rows only).
"""
import time

import polars as pl

from config import wpath
from features import Ctx, features_chunked
from gbdt import cv_train
from prune import label
from splits import load_splits

META = {"s1", "cand", "y", "hidden", "fold", "p1", "p2"}


def full(split: str) -> pl.DataFrame:
    path = wpath(f"feat_{split}.parquet")
    if path.exists():
        return pl.read_parquet(path)
    p = pl.read_parquet(wpath(f"pruned_{split}.parquet"))
    if split == "train":  # stage-1 is trained / evaluated on Q only
        q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
        p = p.join(q, on="s1")
    f = features_chunked(Ctx(split), p, "full")
    f.write_parquet(path)
    return f


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def main():
    t0 = time.time()
    tr = label(full("train")).join(load_splits().rename({"idx": "s1"}), on="s1")
    te = full("test")
    feats = feature_cols(te)
    print(f"stage-1: {tr.height:,} train rows ({tr['y'].sum():,} pos), {te.height:,} test rows, {len(feats)} feats")
    oof, preds, _, imp = cv_train(tr, feats, neg_sample=(pl.col("p_prune") < 0.02, 0.25),
                                  predict={"test": te.select(feats)}, log="stage1")
    print(imp.head(25))
    imp.write_csv(wpath("imp_stage1.csv"))
    tr.drop("hidden").with_columns(pl.Series("p1", oof)).write_parquet(wpath("s1_train.parquet"))
    te.with_columns(pl.Series("p1", preds["test"])).write_parquet(wpath("s1_test.parquet"))
    print(f"stage-1 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
