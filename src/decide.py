"""Decision layer: calibrate -> assign -> select, tuned on out-of-fold scores.

Why each step (docs/master-plan/LLD.md §7):
  calibrate  expected-F needs probabilities; isotonic, cross-fitted on OOF so the
             OOF score we report is not flattered by the calibrator
  assign     ground truth is a partition — no S2/S3 record belongs to two S1 —
             so a record claimed by several S1 goes to its best claimant ('hard'),
             or has its probability shared by claim mass ('soft')
  select     per entity, keep the top-k that maximizes expected F0.5, where k=0
             (predict nothing) is worth P(no true match); or a global threshold
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

B2 = 0.25


def fit_calibrator(scores, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(scores, y)


def crossfit_calibrate(scores, y, folds):
    """Calibrated OOF probabilities: fold k is mapped by a calibrator fit on the other folds."""
    out = np.empty(len(scores), dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = fit_calibrator(scores[~m], y[~m]).predict(scores[m])
    return out


def assign(df: pd.DataFrame, mode: str = "hard") -> pd.DataFrame:
    """df: [s1_id, cand_id, p]. Returns df with p adjusted for the partition constraint."""
    if mode == "none":
        return df
    if mode == "hard":
        # deterministic tie-break on s1_id so reruns agree
        order = df.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True], kind="mergesort")
        keep = ~order["cand_id"].duplicated()
        return df.loc[order.index[keep.to_numpy()]].sort_index()
    if mode == "soft":
        tot = df.groupby("cand_id")["p"].transform("sum")
        return df.assign(p=df["p"] * df["p"] / tot.where(tot > 0, 1.0))
    raise ValueError(mode)


def select_threshold(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    return df[df["p"] >= thr]


def select_expected_f(df: pd.DataFrame, miss: float = 0.0) -> pd.DataFrame:
    """Per entity, keep the top-k maximizing E[F0.5] ≈ 1.25·Σ_{i≤k} p_i / (0.25·E|T| + k),
    with E|T| = Σp + miss; k=0 is worth Π(1-p)·exp(-miss) (true singleton).
    `miss` = expected true links blocking never offered (per entity)."""
    d = df.sort_values(["s1_id", "p"], ascending=[True, False], kind="mergesort")
    g = d.groupby("s1_id", sort=False)["p"]
    k = g.cumcount().to_numpy() + 1
    cum = g.cumsum().to_numpy()
    et = g.transform("sum").to_numpy() + miss
    v = (1 + B2) * cum / (B2 * et + k)
    log1m = np.log1p(-np.clip(d["p"].to_numpy(), 0, 1 - 1e-12))
    v0 = np.exp(pd.Series(log1m, index=d.index).groupby(d["s1_id"]).transform("sum").to_numpy() - miss)
    vmax = pd.Series(v, index=d.index).groupby(d["s1_id"]).transform("max").to_numpy()
    # first k reaching the entity's best value; keep ranks up to it, unless empty wins
    best_k = pd.Series(np.where(v >= vmax, k, np.inf), index=d.index).groupby(d["s1_id"]).transform("min").to_numpy()
    keep = (k <= best_k) & (vmax > v0)
    return d[keep]


def select_expected_f_exact(df: pd.DataFrame, miss: float = 0.0, batch: int = 200_000) -> pd.DataFrame:
    """select_expected_f with the Jensen gap removed from the main term.

    The plain decoder scores top-k by E[TP]/(0.25·E|T| + k) -- a ratio of
    expectations. F is concave in TP, so that over-values uncertain cut-offs,
    and the error is largest for entities with 1-4 true links, which is most
    of them. Here TP_k = Σ_{i<=k} Bernoulli(p_i) keeps its exact distribution
    (a Poisson-binomial, built by a DP over the ranked list); only the links
    outside the top-k enter by their mean:

        E[F_k] = Σ_a P(TP_k = a) · 1.25·a / (0.25·(a + Σ_{i>k} p_i + miss) + k)

    k=0 keeps the plain decoder's value, P(no true link). Vectorized over
    entities padded to the longest candidate list (K=30), in batches.
    """
    d = df.sort_values(["s1_id", "p"], ascending=[True, False], kind="mergesort")
    codes, starts = np.unique(d["s1_id"].to_numpy(), return_index=True)
    order = np.argsort(starts)
    starts = starts[order]
    lens = np.diff(np.r_[starts, len(d)])
    rank = np.arange(len(d)) - np.repeat(starts, lens)
    kmax = int(lens.max()) if len(lens) else 0
    p_all = np.clip(d["p"].to_numpy(dtype=np.float64), 0, 1)
    keep_k = np.zeros(len(starts), dtype=np.int64)

    a_vals = np.arange(kmax + 1, dtype=np.float64)
    for lo in range(0, len(starts), batch):
        hi = min(lo + batch, len(starts))
        ents = np.arange(lo, hi)
        P = np.zeros((hi - lo, kmax))
        rows = np.repeat(ents - lo, lens[lo:hi])
        sl = slice(starts[lo], starts[hi - 1] + lens[hi - 1])
        P[rows, rank[sl]] = p_all[sl]
        total = P.sum(1)
        v0 = np.exp(np.log1p(-np.minimum(P, 1 - 1e-12)).sum(1) - miss)
        best_v, best_k = v0, np.zeros(hi - lo, dtype=np.int64)
        pmf = np.zeros((hi - lo, kmax + 1))
        pmf[:, 0] = 1.0
        cum = np.zeros(hi - lo)
        for k in range(1, kmax + 1):
            pk = P[:, k - 1:k]
            pmf[:, 1:] = pmf[:, 1:] * (1 - pk) + pmf[:, :-1] * pk
            pmf[:, 0] *= (1 - pk[:, 0])
            cum += P[:, k - 1]
            rest = (total - cum + miss)[:, None]
            f = (1 + B2) * a_vals[None, :] / (B2 * (a_vals[None, :] + rest) + k)
            v = (pmf * f).sum(1)
            better = (v > best_v) & (k <= lens[lo:hi])
            best_v = np.where(better, v, best_v)
            best_k = np.where(better, k, best_k)
        keep_k[lo:hi] = best_k
    return d[rank < np.repeat(keep_k, lens)]


def macro_f05(selected: pd.DataFrame, truth_count: pd.Series, y_col="y") -> float:
    """Vectorized macro F0.5 over ALL entities in truth_count (index s1_id -> #true links).

    selected must carry the 0/1 label column; equals metrics.macro_f_beta (tests/test_decide.py).
    """
    tp = selected.groupby("s1_id")[y_col].sum()
    npred = selected.groupby("s1_id").size()
    t = truth_count.astype(float)
    tp = tp.reindex(t.index, fill_value=0).astype(float)
    p = npred.reindex(t.index, fill_value=0).astype(float)
    f = np.where((t == 0) & (p == 0), 1.0,
                 np.where((t == 0) | (p == 0), 0.0, (1 + B2) * tp / (B2 * t + p).clip(lower=1e-12)))
    return float(np.mean(f))


def apply(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run one decision config over [s1_id, cand_id, p(, y)] and return the selected rows."""
    d = assign(df, cfg["assign"])
    if cfg["select"] == "threshold":
        return select_threshold(d, cfg["thr"])
    if cfg["select"] == "expected_f_exact":
        return select_expected_f_exact(d, cfg["miss"])
    return select_expected_f(d, cfg["miss"])


def tune(df: pd.DataFrame, truth_count: pd.Series,
         thr_grid=np.round(np.arange(0.05, 0.96, 0.01), 2), miss_grid=(0.0, 0.05, 0.1, 0.2, 0.4),
         modes=("none", "hard", "soft"), exact=False):
    """Sweep every (assign, select) config on OOF. Returns (best_cfg, table sorted by score).

    `modes` limits the assignment step. hard/soft resolve competition between
    the entities present in `df`, so they are only honest when `df` holds every
    claimant a record will face at inference -- a sample or holdout does not.
    """
    rows = []
    for mode in modes:
        d = assign(df, mode)
        for thr in thr_grid:
            rows.append({"assign": mode, "select": "threshold", "thr": float(thr), "miss": None,
                         "f05": macro_f05(select_threshold(d, thr), truth_count)})
        for miss in miss_grid:
            rows.append({"assign": mode, "select": "expected_f", "thr": None, "miss": float(miss),
                         "f05": macro_f05(select_expected_f(d, miss), truth_count)})
            if exact:
                rows.append({"assign": mode, "select": "expected_f_exact", "thr": None,
                             "miss": float(miss),
                             "f05": macro_f05(select_expected_f_exact(d, miss), truth_count)})
    table = pd.DataFrame(rows).sort_values("f05", ascending=False, kind="mergesort").reset_index(drop=True)
    return table.iloc[0].to_dict(), table
