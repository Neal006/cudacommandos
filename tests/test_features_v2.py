"""python tests/test_features_v2.py — v2 features == old features on edge-case records,
new features behave as designed, stage-2 context is computed as documented.
(The same equality holds on 900k real train pairs: max |old-new| = 2.9e-8.)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import features as F  # noqa: E402
import features_v2 as F2  # noqa: E402
import normalize as N  # noqa: E402
import stage2 as S2  # noqa: E402
from normalize import add_feature_columns  # noqa: E402

rec = lambda rows: pd.DataFrame(rows, columns=[C.ID, C.NAME, C.ADDR, C.COUNTRY])
s1 = rec([
    ("S1-1", "Global Business Pvt Ltd", "No. 21 - S, Icc, 5Th Street, Ambattur, Chennai, Tamil Nadu", "India"),
    ("S1-2", "Delta Telecommunication Inc", "9914 Pierpont Avenue, Cleveland, OH", "US"),
    ("S1-3", "Moncada Learning Center LLC", "5780 Fawn Ct, Fort Worth, Texas", "US"),
    ("S1-4", "Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France", "France"),
    ("S1-5", "Primary Care Group", "", "US"),
])
oth = rec([
    ("S2-1", "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "#C-21 - S, AMBATTUR, CHENNAI, தமிழ்நாடு", "India"),
    ("S3-1", "Deltatelecommunication.Com", "991 Pierpont Avenue, Cleveland, Ohio", "US"),
    ("S2-2", "Moncada Learning SARL", "5780 FAWN CT, FORT WORTH, TX", "US"),
    ("S3-2", "THERMAL ET FILS", "20 R. PARMENTIER, DUNKERQUE, Nord", "France"),
    ("S2-3", "Primary Care Group", "", "US"),
    ("S3-3", "-- Holloway Peak Inc", "105 ELM ST", "US"),
])
pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-5", "S1-2"],
                      "cand_id": ["S2-1", "S3-1", "S2-2", "S3-2", "S2-3", "S3-3", "S3-3"],
                      "block_sim": [0.1, 0.4, 0.8, 0.5, 0.9, 0.05, 0.02]})

# 1. exact equality with the reference implementation
old = F.build_pair_features(pairs, add_feature_columns(s1), add_feature_columns(oth))
L, R = F2.record_table(s1, workers=1), F2.record_table(oth, workers=1)
new = F2.build_pair_features(pairs, L, R, stats=None, extra=True)
assert list(old.columns) == F2.OLD_COLUMNS
for c in old.columns:
    assert np.allclose(old[c].to_numpy(float), new[c].to_numpy(float), atol=1e-9), (c, old[c].tolist(), new[c].tolist())

# 2. new features do what they claim
r = new.set_index(pairs["s1_id"] + ">" + pairs["cand_id"])
assert r.loc["S1-1>S2-1", "skel_eq"] == 1.0 and r.loc["S1-1>S2-1", "either_native"] == 1.0   # Tamil script
assert r.loc["S1-1>S2-1", "core_jaccard"] == 0.0                                             # old features blind to it
assert r.loc["S1-2>S3-1", "domain_ratio"] > 0.9                                              # domain stem
assert r.loc["S1-2>S3-1", "hn_near"] == 1.0 and r.loc["S1-2>S3-1", "hn_state"] == -1.0       # 9914 vs 991
assert r.loc["S1-3>S2-2", "legal_compat"] == 2.0                                             # LLC vs SARL
assert r.loc["S1-3>S2-2", "hn_state"] == 1.0
assert r.loc["S1-5>S2-3", "addr_empty_any"] == 1.0
assert r.loc["S1-2>S3-3", "domain_ratio"] == -1.0

# 3. label-free stats features
stats = {"n": 4, "name_count": pd.Series({"primary care group": 2}), "addr_count": pd.Series({"": 3}),
         "tok_df": pd.Series({"primary": 2, "care": 2, "group": 2})}
w = F2.build_pair_features(pairs, L, R, stats=stats, extra=True).set_index(r.index)
assert w.loc["S1-5>S2-3", "name_genericity"] > 0 and w.loc["S1-1>S2-1", "name_genericity"] == 0
assert w.loc["S1-5>S2-3", "colocation"] == 0.0                     # empty address never counts as co-located
assert 0 < w.loc["S1-5>S2-3", "name_weighted_overlap"] <= 1.0

# 4. stage 2: competition + peers
p1 = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.3, 0.6])
s2f = S2.build(pairs, p1, R).set_index(r.index)
# S3-3 is claimed by S1-2 (0.6) and S1-5 (0.3)
assert s2f.loc["S1-2>S3-3", "claim_rank"] == 1 and s2f.loc["S1-5>S3-3", "claim_rank"] == 2
assert abs(s2f.loc["S1-5>S3-3", "claim_gap"] - 0.3) < 1e-12 and s2f.loc["S1-5>S3-3", "n_claims"] == 2
assert s2f.loc["S1-2>S3-1", "p1_second"] == 0.6 and s2f.loc["S1-2>S3-3", "p1_rank"] == 2
# peers: a record is never compared with itself; single-candidate entities have no peers
assert s2f.loc["S1-5>S2-3", "peer1_sim"] == -1.0 and 0 <= s2f.loc["S1-5>S2-3", "peer2_sim"] <= 1
assert 0 <= s2f.loc["S1-5>S3-3", "peer1_sim"] <= 1
assert s2f.loc["S1-1>S2-1", "peer1_sim"] == -1.0 and s2f.loc["S1-1>S2-1", "peer2_sim"] == -1.0

# 5. normalizer v2 edge cases
assert N.domain_stem("Acme.co Traders") == "" and N.domain_stem("www.acme-tools.in") == "acmetools"
assert N.legal_form("PVT. ASTOR TRADING LTD.") == "PVT_LTD" and N.legal_form("Lee and Lawson") == ""
assert N.first_number(None) == "" and N.first_number("HN 753 E-1") == "753"
assert N.translit_core(None) == "" and N.skeleton("") == ""
print("features_v2 ok")
