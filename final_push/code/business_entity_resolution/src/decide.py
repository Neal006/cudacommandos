"""Stage G: one-to-one assignment + per-S1 expected-F0.5 subset selection.

Probabilities are adjusted p' = sigmoid(a * logit(p) + b) before the decision (a, b tuned on OOF),
per country if a params dict is supplied (leaderboard protocol, plan §H.2).
"""
import itertools
import json

import numpy as np
import polars as pl
from numba import njit, prange

from config import DECIDE_MAX_M, MC_SAMPLES, SEED, wpath


@njit(parallel=True, cache=True)
def _best_k(offsets, p, n_samples, seed):
    """offsets: group boundaries over p (each group sorted desc). returns chosen k per group."""
    ng = len(offsets) - 1
    ks = np.zeros(ng, np.int32)
    for g in prange(ng):
        a, b = offsets[g], offsets[g + 1]
        m = b - a
        if m == 0:
            continue
        np.random.seed(seed + g)
        ef = np.zeros(m + 1)
        y = np.zeros(m, np.int32)
        for s in range(n_samples):
            t = 0
            for i in range(m):
                y[i] = 1 if np.random.random() < p[a + i] else 0
                t += y[i]
            if t == 0:
                ef[0] += 1.0
                continue
            tp = 0
            for k in range(1, m + 1):
                tp += y[k - 1]
                ef[k] += 1.25 * tp / (0.25 * t + k)
        best, bk = -1.0, 0
        for k in range(m + 1):
            if ef[k] > best + 1e-12:
                best, bk = ef[k], k
        ks[g] = bk
    return ks


def adjust(p: pl.Expr, a: float, b: float) -> pl.Expr:
    pc = p.clip(1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + (-(a * (pc / (1 - pc)).log() + b)).exp())


def one_to_one(df: pl.DataFrame, delta=0.0) -> pl.DataFrame:
    """keep each candidate only for its best S1 (optionally only if it wins by >= delta)."""
    r = pl.col("p").rank("ordinal", descending=True).over("cand")
    if delta <= 0:
        return df.filter(r == 1)
    df = df.with_columns(r.alias("_r"))
    second = df.filter(pl.col("_r") == 2).select("cand", pl.col("p").alias("_p2"))
    return (df.filter(pl.col("_r") == 1).join(second, on="cand", how="left")
              .filter(pl.col("p") - pl.col("_p2").fill_null(0.0) >= delta).drop("_r", "_p2"))


def expected_f(df: pl.DataFrame, max_m=DECIDE_MAX_M, n_samples=MC_SAMPLES) -> pl.DataFrame:
    """df: s1, cand, p (after one-to-one). returns selected (s1, cand)."""
    d = (df.sort(["s1", "p"], descending=[False, True])
           .with_columns(pl.int_range(pl.len()).over("s1").alias("_i")).filter(pl.col("_i") < max_m))
    s1 = d["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    offsets = np.r_[starts, len(s1)].astype(np.int64)
    ks = _best_k(offsets, d["p"].to_numpy().astype(np.float64), n_samples, SEED)
    kk = np.repeat(ks, np.diff(offsets))
    return d.filter(pl.Series(d["_i"].to_numpy() < kk)).select("s1", "cand")


def threshold(df: pl.DataFrame, t: float) -> pl.DataFrame:
    return df.filter(pl.col("p") >= t).select("s1", "cand")


def decide(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """df: s1, cand, p, country. params: {'default': {...}, 'France': {...}, ...} with keys
    method ('ef'|'thr'), a, b, delta, t."""
    out = []
    df = df.with_columns(pl.col("p").alias("p_raw"))
    base = params.get("default", params)
    # one-to-one must see all S1s together; country only changes the adjustment and the rule
    adj = []
    for c in df["country"].unique().to_list():
        pc = dict(base, **params.get(c, {}))
        adj.append(df.filter(pl.col("country") == c).with_columns(adjust(pl.col("p_raw"), pc["a"], pc["b"]).alias("p")))
    d = one_to_one(pl.concat(adj), base.get("delta", 0.0))
    for c in d["country"].unique().to_list():
        pc = dict(base, **params.get(c, {}))
        dc = d.filter(pl.col("country") == c)
        out.append(expected_f(dc) if pc.get("method", "ef") == "ef" else threshold(dc, pc["t"]))
    return pl.concat(out) if out else pl.DataFrame(schema={"s1": pl.Int32, "cand": pl.Int32})


def tune(df: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.DataFrame, fast=False):
    """grid-search the decision rule on OOF predictions; returns best params and a results table."""
    from evaluate import macro_f05
    df = df.filter(pl.col("p") >= 1e-3)
    res = []
    grid_a = [0.8, 1.0, 1.25, 1.5] if not fast else [1.0]
    grid_b = [-1.0, -0.5, 0.0, 0.5, 1.0] if not fast else [0.0]
    for a, b in itertools.product(grid_a, grid_b):
        prm = {"method": "ef", "a": a, "b": b, "delta": 0.0}
        f = macro_f05(decide(df, prm), gt, s1_ids)
        res.append({**prm, "t": None, "f": f})
        print(f"   ef a={a} b={b}: {f:.5f}", flush=True)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        prm = {"method": "thr", "a": 1.0, "b": 0.0, "delta": 0.0, "t": t}
        f = macro_f05(decide(df, prm), gt, s1_ids)
        res.append({**prm, "f": f})
        print(f"   thr t={t}: {f:.5f}", flush=True)
    best = max(res, key=lambda r: r["f"])
    for dl in [0.05, 0.1, 0.2]:
        prm = dict(best, delta=dl)
        f = macro_f05(decide(df, prm), gt, s1_ids)
        res.append({**prm, "f": f})
        print(f"   {best['method']} delta={dl}: {f:.5f}", flush=True)
    best = max(res, key=lambda r: r["f"])
    best = {k: v for k, v in best.items() if k != "f" and v is not None}
    return {"default": best}, pl.DataFrame(res)


def save_params(params, name="decision_params.json"):
    with open(wpath(name), "w") as f:
        json.dump(params, f, indent=2)


def load_params(name="decision_params.json"):
    with open(wpath(name)) as f:
        return json.load(f)
