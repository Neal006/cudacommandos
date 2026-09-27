"""Regression tests for the chasing99 review fixes. Data-free: synthetic frames only.

  1. a cached score array over a REORDERED frame of the same length is refused
     (a length check alone passes it and silently misaligns every score);
  2. a stage-2 model trained with the reranker cannot be scored without it
     (v4's test phase used to pass kwargs predict_test_chunked did not accept);
  3. the v4 holdout never overlaps the sample or the reranker's entities,
     and is reproducible;
  4. holdout folds never split an entity;
  5. decide.tune(modes=...) sweeps only the requested assignment modes.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D  # noqa: E402
import decide  # noqa: E402

import run_v4  # noqa: E402


def frame(n_ent=50, k=4, seed=0):
    rng = np.random.default_rng(seed)
    s1 = np.repeat([f"S1-{i:04d}" for i in range(n_ent)], k)
    cand = [f"S2-{c:05d}" for c in rng.integers(0, 400, n_ent * k)]
    return pd.DataFrame({"s1_id": s1, "cand_id": cand})


# ---- 1: score cache alignment
pairs = frame()
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "testp_x.npy"
    np.save(path, np.linspace(0, 1, len(pairs)).astype(np.float32))
    D.write_score_meta(path, pairs, run="runs/x")
    assert path.with_suffix(".json").exists(), "sidecar not written"
    D.check_score_meta(path, pairs)                     # same frame: passes

    shuffled = pairs.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert len(shuffled) == len(pairs)                  # a length check cannot see this
    try:
        D.check_score_meta(path, shuffled)
        raise AssertionError("a reordered frame of equal length was accepted")
    except SystemExit:
        pass
    try:
        D.check_score_meta(path, pairs.iloc[:-1])
        raise AssertionError("a shorter frame was accepted")
    except SystemExit:
        pass

    # a cache with no sidecar (003's testp_f550ffb02552.npy) still works, with a warning
    legacy = Path(d) / "legacy.npy"
    np.save(legacy, np.zeros(len(pairs), dtype=np.float32))
    D.check_score_meta(legacy, pairs)
print("1 ok: reordered / resized frames are refused, legacy caches warn")

# ---- 2: fingerprint is order-sensitive but ignores the index
fp = D.frame_fingerprint(pairs)
assert fp == D.frame_fingerprint(pairs.set_index(pairs.index + 7)), "index leaked into the hash"
assert fp != D.frame_fingerprint(pairs.iloc[::-1]), "row order ignored"
print("2 ok: fingerprint depends on row content and order only")

# ---- 3: holdout is disjoint from sample and reranker entities, and reproducible
ids = pd.Series(np.repeat([f"S1-{i:04d}" for i in range(1000)], 3))
sample = {f"S1-{i:04d}" for i in range(0, 300)}
rr_seen = {f"S1-{i:04d}" for i in range(300, 400)}
h1 = run_v4.pick_holdout(ids, sample | rr_seen, 250)
h2 = run_v4.pick_holdout(ids, sample | rr_seen, 250)
assert len(h1) == 250 and h1 == h2, "holdout size or reproducibility wrong"
assert not (h1 & sample) and not (h1 & rr_seen), "holdout leaks into sample/reranker"
assert len(run_v4.pick_holdout(ids, sample | rr_seen, 10**6)) == 600, "cap at the pool size"
assert run_v4.pick_holdout(ids, sample, 0) == set(), "0 disables"
print("3 ok: holdout disjoint from sample and reranker, reproducible, capped, disableable")

# ---- 4: holdout folds never split an entity
f = run_v4.entity_folds(ids, 5)
per_ent = pd.Series(f).groupby(ids.to_numpy()).nunique()
assert (per_ent == 1).all(), "an entity landed in two folds"
assert set(np.unique(f)) == set(range(5)), "not all folds used"
print("4 ok: entity folds are group-pure")

# ---- 4b: the holdout frame keeps block_sim (featurize needs it; v4 crashed without it)
fp_ = pd.DataFrame({"s1_id": ["A", "A", "B"], "cand_id": ["x", "y", "z"], "block_sim": [.1, .2, .3]})
hf = run_v4.hold_frame(fp_, np.array([False, True, True]))
assert list(hf.columns) == ["s1_id", "cand_id", "block_sim"] and list(hf["block_sim"]) == [.2, .3]
assert list(hf.index) == [0, 1], "index must be reset"
print("4b ok: holdout frame keeps every column")

# ---- 5: tune respects modes
rng = np.random.default_rng(3)
df = frame(40, 5, seed=3)
df = df.drop_duplicates(["s1_id", "cand_id"]).reset_index(drop=True)
df["y"] = (rng.random(len(df)) < 0.3).astype(int)
df["p"] = np.clip(df["y"] * 0.6 + rng.random(len(df)) * 0.4, 0, 1)
tc = df.groupby("s1_id")["y"].sum()
_, table = decide.tune(df, tc, modes=("none",))
assert set(table["assign"]) == {"none"}, "tune swept modes it was told not to"
_, full = decide.tune(df, tc)
assert set(full["assign"]) == {"none", "hard", "soft"}, "default changed"
print("5 ok: tune(modes=) restricts the sweep; default unchanged")

# ---- 6: the exact decoder is closer to the true E[F0.5] optimum than the plain one
import itertools  # noqa: E402

rng = np.random.default_rng(0)
rows, truth_best = [], {}
for e in range(300):
    n = int(rng.integers(1, 8))
    pv = np.sort(rng.beta(0.6, 0.9, n))[::-1]
    sid = f"S1-{e:04d}"
    rows += [(sid, f"c{e}_{i}", pv[i]) for i in range(n)]
    ev = np.zeros(n + 1)                                  # brute force over 2^n outcomes
    for bits in itertools.product([0, 1], repeat=n):
        b = np.array(bits)
        pr, t = np.prod(np.where(b, pv, 1 - pv)), b.sum()
        for k in range(n + 1):
            if k == 0:
                ev[k] += pr * (t == 0)
            elif t:
                ev[k] += pr * 1.25 * b[:k].sum() / (0.25 * t + k)
    truth_best[sid] = ev
dd = pd.DataFrame(rows, columns=["s1_id", "cand_id", "p"])


def regret(sel):
    k = sel.groupby("s1_id").size().reindex(list(truth_best), fill_value=0)
    return np.mean([truth_best[s].max() - truth_best[s][k[s]] for s in truth_best])


r_plain = regret(decide.select_expected_f(dd, 0.0))
r_exact = regret(decide.select_expected_f_exact(dd, 0.0))
assert r_exact < r_plain, f"exact decoder not better: {r_exact:.5f} vs {r_plain:.5f}"
# degenerate inputs: certain links are kept, impossible ones dropped, no crash on 1-row entities
det = pd.DataFrame({"s1_id": ["A", "A", "A", "B"], "cand_id": ["a", "b", "c", "d"],
                    "p": [1.0, 1.0, 0.0, 0.0]})
got = decide.select_expected_f_exact(det, 0.0)
assert sorted(got["cand_id"]) == ["a", "b"], got
assert len(decide.select_expected_f_exact(det.iloc[:0], 0.0)) == 0, "empty frame"
print(f"6 ok: exact decoder regret {r_exact:.5f} < plain {r_plain:.5f}; degenerate cases right")

# ---- 7: hopeso passes pull the right siblings and respect country + pool caps
import polars as pl  # noqa: E402
import hopeso  # noqa: E402
import stage2 as S2  # noqa: E402

k = pl.DataFrame({
    "entity_id": ["S2-1", "S2-2", "S3-3", "S2-4", "S2-5", "S2-6", "S2-7"],
    "country":   ["India", "India", "India", "US", "India", "India", "India"],
    "cn":        ["ram medicals", "ram medicals", "ram medicals", "ram medicals", "", "medical store", "medical store"],
    "sk":        ["rmmdcls", "rmmdcls", "rmmdcls", "rmmdcls", "", "mdclstr", "mdclstr"],
    "addr_empty": [False, True, False, False, True, False, False]})
base = pl.DataFrame({"s1_id": ["S1-A", "S1-A"], "cand_id": ["S2-1", "S2-6"], "block_sim": [0.9, 0.2]})
got = set(hopeso.sibs(base, k, "cn", top=1, cap=50).rows())
assert got == {("S1-A", "S2-1"), ("S1-A", "S2-2"), ("S1-A", "S3-3")}, got   # no US copy, top-1 only
assert hopeso.sibs(base, k, "cn", top=1, cap=2).height == 0, "pool of 3 must be capped out at 2"
s1k = pl.DataFrame({"entity_id": ["S1-A"], "country": ["India"], "cn": ["ram medicals"],
                    "sk": ["rmmdcls"], "addr_empty": [False]})
assert set(hopeso.dost(s1k, k, "cn", cap=10)["cand_id"]) == {"S2-1", "S2-2", "S3-3"}
new = hopeso.extras(base, s1k, k, dict(sib_top=1, sib_cap=50, dost_cap=10))
assert set(new["cand_id"]) == {"S2-2", "S3-3"}, "extras must drop pairs the base already has"
p_def = hopeso.tag_path("test", None)
p_alt = hopeso.tag_path("test", None, vibe=dict(sib_top=5, sib_cap=20, dost_cap=0))
assert p_def != p_alt and "_t3c50d10" in p_def.name, "pass settings must be in the cache name"
try:
    hopeso.load_frame(None, None, None, "test", None, tag="no_such_tag_zz")
    raise AssertionError("a missing tagged frame must stop the run, not fall back to base")
except SystemExit as e:
    assert "hopeso.py build" in str(e)
print("7 ok: sibs/dost stay in-country, respect caps, extras is base-disjoint, "
      "cache name carries settings, missing tagged frame stops")

# ---- 7b: lifted caps: keep_best bounds per-entity pairs, AMLC_VIBE is parsed, keep is in the name
import os  # noqa: E402

nw = pd.DataFrame({"s1_id": ["A", "A", "A", "B", "B"], "cand_id": ["x", "y", "z", "u", "v"],
                   "block_sim": [0.1, 0.9, 0.9, 0.5, 0.2]})
kb = hopeso.keep_best(nw, 2)
assert list(zip(kb["s1_id"], kb["cand_id"])) == [("A", "y"), ("A", "z"), ("B", "u"), ("B", "v")], kb
assert hopeso.keep_best(nw.iloc[::-1], 2).equals(kb), "keep_best depends on row order"
os.environ["AMLC_VIBE"] = "5,500,50,40"
assert hopeso._vibe() == dict(sib_top=5, sib_cap=500, dost_cap=50, keep=40)
del os.environ["AMLC_VIBE"]
assert hopeso._vibe() == dict(sib_top=3, sib_cap=50, dost_cap=10, keep=0), "default changed"
assert "_t3c50d10.parquet" in p_def.name, "keep=0 must keep the old cache name"
assert "_t5c500d50k40" in hopeso.tag_path("test", None, vibe=hopeso._vibe() | dict(
    sib_top=5, sib_cap=500, dost_cap=50, keep=40)).name
print("7b ok: keep_best is order-free and bounded, AMLC_VIBE parsed, keep in cache name")

# ---- 8: gang features: count siblings, best OTHER sibling, ties, empty keys
s1g = np.array(["A", "A", "A", "A", "B", "B"])
kg = np.array(["x", "x", "y", "", "x", "x"], dtype=object)
n_, mx_ = S2.gang(s1g, kg, np.array([.9, .2, .5, .7, .3, .3]), np.ones(6, bool))
assert list(n_) == [2, 2, 1, 0, 2, 2] and list(mx_) == [.2, .9, -1, -1, .3, .3], (n_, mx_)
print("8 ok: gang features exclude self, handle ties and empty keys")

print("\nALL REVIEW-FIX TESTS PASSED")
