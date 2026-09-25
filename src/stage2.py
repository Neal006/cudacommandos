"""Stage-2 context features, computed from stage-1 scores `p1`.

Leakage rule: on train, p1 MUST be out-of-fold (each pair scored by a model that
never saw its entity) and stage 2 must reuse the stage-1 folds. On test, p1 is
the mean of the fold models. See DATA_SECURITY_AND_LEAKAGE.md L4.

  entity shape  p1_rank, p1_gap, p1_second, n_strong, sum_p1
                -> how this candidate sits among the entity's options
  competition   claim_rank, claim_gap, n_claims, n_strong_claims
                -> ground truth is a partition: a record strongly claimed by a
                   different S1 is unlikely to be ours
  peers         peer1_sim, peer2_sim
                -> true matches are duplicates of each other (a source holds up to
                   5-6 copies of one business), so a candidate that resembles the
                   entity's top candidates is more likely right
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

STRONG = 0.5


def _quick_sim(R, a_idx, b_idx):
    """Mean of name/address token-set similarity between two candidate records."""
    n = process.cpdist(R["_core_name"].to_numpy()[a_idx], R["_core_name"].to_numpy()[b_idx],
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64)
    a = process.cpdist(R["_core_addr"].to_numpy()[a_idx], R["_core_addr"].to_numpy()[b_idx],
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64)
    return (n + a) / 200.0


def build(pairs: pd.DataFrame, p1: np.ndarray, R: pd.DataFrame) -> pd.DataFrame:
    """pairs: [s1_id, cand_id] aligned with p1; R: features_v2.record_table of candidates."""
    d = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "cand_id": pairs["cand_id"].to_numpy(),
                      "p1": np.asarray(p1, dtype=np.float64)})
    ge = d.groupby("s1_id", sort=False)["p1"]
    gc = d.groupby("cand_id", sort=False)["p1"]
    f = pd.DataFrame(index=d.index)
    f["p1"] = d["p1"]
    f["p1_rank"] = ge.rank(ascending=False, method="first")
    f["p1_gap"] = ge.transform("max") - d["p1"]
    f["n_strong"] = (d["p1"] >= STRONG).groupby(d["s1_id"]).transform("sum")
    f["sum_p1"] = ge.transform("sum")
    f["claim_rank"] = gc.rank(ascending=False, method="min")
    f["claim_gap"] = gc.transform("max") - d["p1"]
    f["n_claims"] = gc.transform("size")
    f["n_strong_claims"] = (d["p1"] >= STRONG).groupby(d["cand_id"]).transform("sum")

    # one sort gives every per-entity order statistic without per-group Python
    ri = R.index.get_indexer(d["cand_id"].astype(str))
    order = d.assign(ri=ri).sort_values(["s1_id", "p1"], ascending=[True, False], kind="mergesort")
    cc = order.groupby("s1_id", sort=False).cumcount().to_numpy()
    first = order[cc == 0].set_index("s1_id")
    second = order[cc == 1].set_index("s1_id")
    f["p1_second"] = d["s1_id"].map(second["p1"]).fillna(0.0).to_numpy()
    t1 = d["s1_id"].map(first["ri"]).to_numpy()
    t2 = d["s1_id"].map(second["ri"]).fillna(-1).astype(np.int64).to_numpy()
    for name, t in (("peer1_sim", t1), ("peer2_sim", t2)):
        sim = np.full(len(d), -1.0)
        ok = (t >= 0) & (t != ri)          # comparing a record with itself says nothing
        sim[ok] = _quick_sim(R, ri[ok], t[ok].astype(np.int64))
        f[name] = sim
    return f
