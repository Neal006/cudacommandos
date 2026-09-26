"""python tests/test_decide.py — decision layer against the reference metric and brute force."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import decide  # noqa: E402
from metrics import macro_f_beta  # noqa: E402

rng = np.random.default_rng(0)

# random world: 400 entities, 0-12 candidates each, some true links outside candidates
rows, truth = [], {}
for e in range(400):
    sid = f"S1-{e}"
    n = rng.integers(0, 13)
    cands = [f"S2-{e}-{j}" for j in range(n)]
    true = {c for c in cands if rng.random() < 0.3} | ({f"S3-miss-{e}"} if rng.random() < 0.1 else set())
    truth[sid] = true
    for c in cands:
        rows.append((sid, c, float(rng.random()), int(c in true)))
df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "p", "y"])
tc = pd.Series({k: len(v) for k, v in truth.items()})

# 1. vectorized metric == reference metric, for several selections
for thr in (0.0, 0.3, 0.7, 1.1):
    sel = decide.select_threshold(df, thr)
    ref = macro_f_beta(truth, sel.groupby("s1_id")["cand_id"].apply(set).to_dict())
    assert abs(decide.macro_f05(sel, tc) - ref) < 1e-12, (thr, decide.macro_f05(sel, tc), ref)

# 2. expected-F keeps exactly the brute-force best prefix per entity
for miss in (0.0, 0.2):
    sel = decide.select_expected_f(df, miss)
    kept = sel.groupby("s1_id").size()
    for sid, g in df.groupby("s1_id"):
        p = np.sort(g["p"].to_numpy())[::-1]
        et = p.sum() + miss
        vals = [np.prod(1 - p) * np.exp(-miss)] + [1.25 * p[:k].sum() / (0.25 * et + k) for k in range(1, len(p) + 1)]
        best = int(np.argmax(vals))
        assert kept.get(sid, 0) == best, (sid, kept.get(sid, 0), best, vals)

# 3. hand cases
one = pd.DataFrame({"s1_id": ["a"], "cand_id": ["x"], "p": [0.3]})
assert decide.select_expected_f(one).empty                      # P(no match)=0.7 beats 0.35
two = pd.DataFrame({"s1_id": ["a"] * 3, "cand_id": list("xyz"), "p": [0.95, 0.9, 0.1]})
assert list(decide.select_expected_f(two)["cand_id"]) == ["x", "y"]

# 4. hard assignment: each candidate kept once, by its best claimant, ties -> smallest s1_id
claims = pd.DataFrame({"s1_id": ["a", "b", "c", "a"], "cand_id": ["x", "x", "x", "y"], "p": [0.4, 0.9, 0.9, 0.2]})
h = decide.assign(claims, "hard")
assert sorted(zip(h.s1_id, h.cand_id)) == [("a", "y"), ("b", "x")], h
s = decide.assign(claims, "soft")
assert abs(s.loc[1, "p"] - 0.9 * 0.9 / 2.2) < 1e-12 and abs(s.loc[3, "p"] - 0.2) < 1e-12

# 5. cross-fit calibration is monotone within a fold and never sees its own fold
y = (rng.random(3000) < 0.4).astype(int)
sc = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1)
folds = np.arange(3000) % 5
cal = decide.crossfit_calibrate(sc, y, folds)
assert cal.min() >= 0 and cal.max() <= 1

best, table = decide.tune(df, tc)
assert table["f05"].is_monotonic_decreasing and best["f05"] == table["f05"].max()
print("decide ok")
