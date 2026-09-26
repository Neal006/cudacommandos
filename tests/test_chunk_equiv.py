"""Chunked test scoring must equal the unchunked path, exactly.

Uses the real 2k train candidate cache so the frame has the same shape as
production: many entities, ~30 candidates each, candidates shared between
entities (which is what makes the claim_* columns global).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import config as C, stage2 as S2, features_v2 as F2, features as F, data as D
from run_v2 import _entity_chunks

pairs = pd.read_parquet(C.INTERIM / "cands_train_k30_df0.01_mdf3_ctry1_n2000.parquet")
pairs = pairs.sort_values("s1_id", kind="mergesort").reset_index(drop=True)
print(f"pairs {len(pairs):,}  entities {pairs.s1_id.nunique():,}  cands {pairs.cand_id.nunique():,}")

rng = np.random.default_rng(0)
p1 = rng.random(len(pairs))

p1p, p2p, p3p = D.source_paths("train")
L = F2.record_table(F2.load_records([p1p], set(pairs["s1_id"])))
R = F2.record_table(F2.load_records([p2p, p3p], set(pairs["cand_id"])))

# --- 1. claims split out == claims computed inline
full = S2.build(pairs, p1, R)
claims = S2.build_claims(pairs, p1)
for c in S2.CLAIM_COLS:
    assert np.allclose(full[c].to_numpy(), claims[c].to_numpy()), f"claims mismatch {c}"
print("1. build_claims matches build's inline columns")

# --- 2. chunked build(claims=...) == unchunked build, column for column
bounds = _entity_chunks(pairs, 20_000)
print(f"   {len(bounds)} chunks: {[h-l for l,h in bounds]}")
assert len(bounds) > 1, "test is meaningless with one chunk"

# a candidate must actually be shared across chunks or the test proves nothing
seen, cross = set(), 0
for lo, hi in bounds:
    ids = set(pairs["cand_id"].iloc[lo:hi])
    cross += len(ids & seen)
    seen |= ids
assert cross > 0, "no candidate spans a chunk boundary -- test would not catch the bug"
print(f"   {cross:,} candidate ids span chunk boundaries")

parts = []
for lo, hi in bounds:
    sl = pairs.iloc[lo:hi]
    Rc = F2.record_table(F2.load_records([p2p, p3p], set(sl["cand_id"])))
    parts.append(S2.build(sl, p1[lo:hi], Rc, claims=claims.iloc[lo:hi]))
chunked = pd.concat(parts, ignore_index=True)

assert list(chunked.columns) == list(full.columns), (list(chunked.columns), list(full.columns))
bad = []
for c in full.columns:
    a, b = full[c].to_numpy(), chunked[c].to_numpy()
    if not np.allclose(a, b, equal_nan=True):
        bad.append((c, int((~np.isclose(a, b, equal_nan=True)).sum())))
assert not bad, f"stage-2 mismatch: {bad}"
print(f"2. chunked stage 2 == unchunked across all {len(full.columns)} columns")

# --- 3. the negative control: WITHOUT global claims, it must differ
naive = pd.concat(
    [S2.build(pairs.iloc[lo:hi], p1[lo:hi],
              F2.record_table(F2.load_records([p2p, p3p], set(pairs["cand_id"].iloc[lo:hi]))))
     for lo, hi in bounds], ignore_index=True)
diff = [c for c in S2.CLAIM_COLS
        if not np.allclose(full[c].to_numpy(), naive[c].to_numpy(), equal_nan=True)]
assert diff, "naive per-chunk claims matched -- the global pass would be pointless"
print(f"3. negative control: naive chunking corrupts {diff}")

# --- 4. stage-1 features are row-independent given a correct R
X_full = F.add_rank_features(F2.build_pair_features(pairs, L, R, stats=None, extra=False),
                             pairs, "core_token_sort")
xp = []
for lo, hi in bounds:
    sl = pairs.iloc[lo:hi]
    Rc = F2.record_table(F2.load_records([p2p, p3p], set(sl["cand_id"])))
    xp.append(F.add_rank_features(F2.build_pair_features(sl, L, Rc, stats=None, extra=False),
                                  sl, "core_token_sort"))
X_ch = pd.concat(xp, ignore_index=True)
bad = [c for c in X_full.columns
       if not np.allclose(X_full[c].to_numpy(), X_ch[c].to_numpy(), equal_nan=True)]
assert not bad, f"stage-1 feature mismatch under per-chunk R: {bad}"
print(f"4. per-chunk record tables give identical stage-1 features ({len(X_full.columns)} cols)")

print("\nALL PASS")
