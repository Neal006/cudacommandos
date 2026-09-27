"""Adversarial validation: can a model tell train (Q, out-of-fold) pairs from test pairs?
AUC ~0.5 = same distribution; high AUC + top features = what shifted. US + India only (France is test-only)."""
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from config import wpath
from normalize import load_norm
from stage1 import feature_cols

N = 1_000_000
rng = np.random.default_rng(0)


def load(split):
    f = pl.read_parquet(wpath(f"feat_{split}.parquet"))
    c = load_norm(split, ["idx", "country"]).rename({"idx": "s1"})
    f = f.join(c, on="s1").filter(pl.col("country") != "France")
    return f.sample(min(N, f.height), seed=1)


tr, te = load("train"), load("test")
feats = [c for c in feature_cols(te) if c != "country"]
X = np.vstack([tr.select(feats).to_numpy(), te.select(feats).to_numpy()]).astype(np.float32)
y = np.r_[np.zeros(tr.height), np.ones(te.height)]
idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
a, b = idx[:cut], idx[cut:]
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, num_threads=4, verbose=-1),
              lgb.Dataset(X[a], y[a]), 200)
print(f"adversarial AUC (all features): {roc_auc_score(y[b], m.predict(X[b])):.4f}")
imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
tot = sum(g for g, _ in imp)
for g, f in imp[:12]:
    print(f"  {f:22s} {g / tot:.3f}   train mean {tr[f].mean():9.4f}   test mean {te[f].mean():9.4f}")
