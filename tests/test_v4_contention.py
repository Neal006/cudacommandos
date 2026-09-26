"""The claim features must be computed over the FULL frame, not the sample.

This is the whole premise of run_v4, so it gets asserted rather than assumed:

  1. claims computed over a superset and then sliced DIFFER from claims
     computed on the subset alone -- if they did not, the fix would be a no-op;
  2. the sliced values are the ones a full-population run would produce, which
     is what inference sees;
  3. integer codes for cand_id give identical results to the raw strings
     (run_v4 factorizes 66M ids to int32 to keep the frame in memory);
  4. contention rises with the number of competing entities, which is the
     mechanism behind the measured train 1.96 vs test 5.55 gap.

Data-free: synthetic frames only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stage2 as S2  # noqa: E402

CLAIMS = list(S2.CLAIM_COLS)


def make_frame(n_entities, n_records, k, seed=0):
    """Each entity lists k candidates drawn from a shared record pool."""
    rng = np.random.default_rng(seed)
    s1, cand = [], []
    for e in range(n_entities):
        picks = rng.choice(n_records, size=k, replace=False)
        s1 += [f"S1-{e:06d}"] * k
        cand += [f"S2-{c:06d}" for c in picks]
    df = pd.DataFrame({"s1_id": s1, "cand_id": cand})
    return df, rng.random(len(df))


# ---- 1 & 2: full-frame claims differ from sample-only, and match the population
full, p_full = make_frame(n_entities=400, n_records=500, k=10, seed=1)
sample_ids = {f"S1-{e:06d}" for e in range(60)}          # 15% of entities
in_sample = full["s1_id"].isin(sample_ids).to_numpy()

claims_full = S2.build_claims(full, p_full)
claims_sliced = claims_full[in_sample].reset_index(drop=True)

sub = full[in_sample].reset_index(drop=True)
claims_naive = S2.build_claims(sub, p_full[in_sample])

assert len(claims_sliced) == len(claims_naive) == in_sample.sum()

diff = {c: float(np.abs(claims_sliced[c].to_numpy() - claims_naive[c].to_numpy()).mean())
        for c in CLAIMS}
assert all(v > 0 for v in diff.values()), f"no column changed -- fix would be a no-op: {diff}"
print("1. full-frame vs sample-only claims differ on every column:")
for c in CLAIMS:
    print(f"     {c:<18} mean |diff| {diff[c]:.4f}   "
          f"full {claims_sliced[c].mean():7.3f}   naive {claims_naive[c].mean():7.3f}")

# the sliced n_claims must equal a direct count over the full population
pop_count = full["cand_id"].map(full["cand_id"].value_counts())[in_sample].to_numpy()
assert np.array_equal(claims_sliced["n_claims"].to_numpy(), pop_count.astype(float)), \
    "sliced n_claims is not the population count"
print("2. sliced n_claims equals the true population count")

# and sample-only must UNDERCOUNT -- that is the bug being fixed
assert (claims_naive["n_claims"] <= claims_sliced["n_claims"]).all()
assert claims_naive["n_claims"].mean() < claims_sliced["n_claims"].mean()
print(f"   sample-only undercounts every row "
      f"({claims_naive['n_claims'].mean():.2f} vs {claims_sliced['n_claims'].mean():.2f})")

# ---- 3: integer codes == raw strings
codes = pd.factorize(full["cand_id"], sort=False)[0].astype(np.int32)
claims_codes = S2.build_claims(pd.DataFrame({"cand_id": codes}), p_full)
for c in CLAIMS:
    assert np.allclose(claims_codes[c].to_numpy(), claims_full[c].to_numpy(), equal_nan=True), c
print("3. int32 cand_id codes give identical claims to raw strings")

# ---- 4: contention scales with the number of competing entities
print("4. contention vs population size (records and k held fixed):")
prev = 0.0
for n_ent in (100, 400, 1600):
    f, _ = make_frame(n_entities=n_ent, n_records=500, k=10, seed=2)
    m = len(f) / f["cand_id"].nunique()
    print(f"     {n_ent:>5} entities -> mean n_claims {m:6.3f}")
    assert m > prev
    prev = m
print("   -> this is why train (150k entities, 1.96) and test (1.73M, 5.55) disagree")

print("\ntest_v4_contention: all checks passed")
