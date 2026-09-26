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


# The four competition columns are the only ones in build() that group by
# cand_id. Everything else groups by s1_id, or compares a candidate with the
# top two of its OWN entity -- all of which stay correct inside a chunk that
# never splits an entity. These four do not: one candidate record can be
# claimed by Source-1 entities that land in different chunks, so computing
# them per chunk would silently give different values than training saw.
CLAIM_COLS = ("claim_rank", "claim_gap", "n_claims", "n_strong_claims")


def build_claims(pairs: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """The cand_id-grouped columns of build(), over the WHOLE frame.

    Split out so the test path can compute them once globally and then build
    the rest chunk by chunk. Needs no record table, which is the point: R for
    all 9.4M test candidates is ~8 GB of Python strings, and this lets the
    global pass avoid it entirely.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    d = pd.DataFrame({"cand_id": pairs["cand_id"].to_numpy(), "p1": p1})
    gc_ = d.groupby("cand_id", sort=False)["p1"]
    f = pd.DataFrame(index=d.index)
    f["claim_rank"] = gc_.rank(ascending=False, method="min")
    f["claim_gap"] = gc_.transform("max") - d["p1"]
    f["n_claims"] = gc_.transform("size")
    f["n_strong_claims"] = (d["p1"] >= STRONG).groupby(d["cand_id"]).transform("sum")
    return f


def build(pairs: pd.DataFrame, p1: np.ndarray, R: pd.DataFrame,
          claims=None) -> pd.DataFrame:
    """pairs: [s1_id, cand_id] aligned with p1; R: features_v2.record_table of candidates.

    `claims`: precomputed CLAIM_COLS for exactly these rows, from build_claims()
    over the full frame. Pass it when `pairs` is a chunk; leave it None and the
    columns are computed here, which is correct only when `pairs` is everything.
    """
    d = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "cand_id": pairs["cand_id"].to_numpy(),
                      "p1": np.asarray(p1, dtype=np.float64)})
    ge = d.groupby("s1_id", sort=False)["p1"]
    f = pd.DataFrame(index=d.index)
    f["p1"] = d["p1"]
    f["p1_rank"] = ge.rank(ascending=False, method="first")
    f["p1_gap"] = ge.transform("max") - d["p1"]
    f["n_strong"] = (d["p1"] >= STRONG).groupby(d["s1_id"]).transform("sum")
    f["sum_p1"] = ge.transform("sum")
    if claims is None:
        gc_ = d.groupby("cand_id", sort=False)["p1"]
        f["claim_rank"] = gc_.rank(ascending=False, method="min")
        f["claim_gap"] = gc_.transform("max") - d["p1"]
        f["n_claims"] = gc_.transform("size")
        f["n_strong_claims"] = (d["p1"] >= STRONG).groupby(d["cand_id"]).transform("sum")
    else:
        if len(claims) != len(d):
            raise ValueError(f"claims has {len(claims)} rows, pairs has {len(d)}")
        for c in CLAIM_COLS:
            f[c] = np.asarray(claims[c], dtype=np.float64)

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
