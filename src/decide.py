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
    return select_expected_f(d, cfg["miss"])


def tune(df: pd.DataFrame, truth_count: pd.Series,
         thr_grid=np.round(np.arange(0.05, 0.96, 0.01), 2), miss_grid=(0.0, 0.05, 0.1, 0.2, 0.4)):
    """Sweep every (assign, select) config on OOF. Returns (best_cfg, table sorted by score)."""
    rows = []
    for mode in ("none", "hard", "soft"):
        d = assign(df, mode)
        for thr in thr_grid:
            rows.append({"assign": mode, "select": "threshold", "thr": float(thr), "miss": None,
                         "f05": macro_f05(select_threshold(d, thr), truth_count)})
        for miss in miss_grid:
            rows.append({"assign": mode, "select": "expected_f", "thr": None, "miss": float(miss),
                         "f05": macro_f05(select_expected_f(d, miss), truth_count)})
    table = pd.DataFrame(rows).sort_values("f05", ascending=False, kind="mergesort").reset_index(drop=True)
    return table.iloc[0].to_dict(), table
