"""Stage-2 experiments on identical folds: which cross-encoder files, which GBDT(s).

  ER_CE=ce3 python exp_stage2.py            -> held-out F0.5 for lgb, xgb and their average
Each variant is isotonic-calibrated on its own OOF and decided with the same fixed rule, so numbers are
comparable; the winning setting is then produced for real with `ER_CE=... ER_STAGE2=... run_pipeline --only context`.
"""
import json
import os
import time

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from config import wpath
from context import context_features
from decide import decide
from evaluate import load_gt, macro_f05
from gbdt import cv_train
from normalize import load_norm
from splits import load_splits
from stage1 import feature_cols

RULE = {"default": {"method": "ef", "a": 1.5, "b": -0.5, "delta": 0.1}}


def main():
    t0 = time.time()
    name = os.environ.get("ER_CE", "all")
    tr = context_features("train")
    feats = feature_cols(tr) + ["p1"]
    gt = load_gt()
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    gtq = gt.join(q, on="s1")
    country = load_norm("train", ["idx", "country"]).rename({"idx": "s1"})
    base = tr.select("s1", "cand").join(country, on="s1")
    oofs, res = {}, {}
    algos = os.environ.get("ER_EXP_ALGOS", "lgb").split(",")      # "lgb,xgb" to also try XGBoost
    for algo in algos:
        oofs[algo], _, _, _ = cv_train(tr, feats, neg_sample=(pl.col("p1") < 0.005, 0.25), log="exp", algo=algo)
    if len(algos) > 1:
        oofs["avg"] = sum(oofs[a] for a in algos) / len(algos)
    y = tr["y"].to_numpy()
    for k, o in oofs.items():
        p = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o, y).predict(o)
        d = base.with_columns(pl.Series("p", p.astype(np.float32))).filter(pl.col("p") >= 1e-3)
        res[k] = macro_f05(decide(d, RULE), gtq, q)
        print(f"  CE={name:10s} stage2={k:4s}: F0.5 {res[k]:.5f}", flush=True)
    with open(wpath("exp_stage2.jsonl"), "a") as f:
        f.write(json.dumps({"ce": name, **res}) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
