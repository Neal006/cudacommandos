"""Pairwise features for the matcher.

One row per (Source-1 entity, candidate) pair. Features are cheap string
comparisons — rapidfuzz for edit-distance family, set ops for token overlap,
plus the blocking similarity itself.

Design note: features are deliberately symmetric and source-agnostic. The
matcher must work identically for S2 and S3 candidates and for a country
(France) it has never seen, so nothing here may key off source prefix or a
country literal.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import config as C


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set, b: set) -> float:
    """Overlap over the smaller set — catches 'partial address' noise, where
    one source holds a strict subset of the other's components."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _num_agreement(a: set, b: set) -> float:
    """-1 both have digits and none match, 0 one side missing, 1 some match.

    Numeric tokens (PIN codes, house numbers) are high precision in both
    directions, so disagreement gets its own signal rather than collapsing
    into 'no evidence'.
    """
    if not a or not b:
        return 0.0
    return 1.0 if a & b else -1.0


def build_pair_features(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    """pairs: columns [s1_id, cand_id, block_sim]. Returns a numeric frame."""
    cols = ["_name", "_core_name", "_addr", "_core_addr", "_acronym",
            "_name_nums", "_addr_nums", C.COUNTRY]
    L = s1.set_index(C.ID)[cols]
    R = others.set_index(C.ID)[cols]

    left = L.reindex(pairs["s1_id"].values).reset_index(drop=True)
    right = R.reindex(pairs["cand_id"].values).reset_index(drop=True)

    f = pd.DataFrame(index=range(len(pairs)))
    f["block_sim"] = pairs["block_sim"].values

    # --- name similarity, several views of the same comparison ---
    ln, rn = left["_name"].fillna(""), right["_name"].fillna("")
    lc, rc = left["_core_name"].fillna(""), right["_core_name"].fillna("")

    f["name_ratio"] = [fuzz.ratio(a, b) / 100 for a, b in zip(ln, rn)]
    f["name_token_sort"] = [fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(ln, rn)]
    f["name_token_set"] = [fuzz.token_set_ratio(a, b) / 100 for a, b in zip(ln, rn)]
    f["name_partial"] = [fuzz.partial_ratio(a, b) / 100 for a, b in zip(ln, rn)]
    f["core_ratio"] = [fuzz.ratio(a, b) / 100 for a, b in zip(lc, rc)]
    f["core_token_sort"] = [fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(lc, rc)]
    f["core_jw"] = [JaroWinkler.similarity(a, b) for a, b in zip(lc, rc)]

    lct = [set(x.split()) for x in lc]
    rct = [set(x.split()) for x in rc]
    f["core_jaccard"] = [_jaccard(a, b) for a, b in zip(lct, rct)]
    f["core_containment"] = [_containment(a, b) for a, b in zip(lct, rct)]
    f["core_first_tok_eq"] = [
        float(bool(a) and bool(b) and next(iter(sorted(a))) == next(iter(sorted(b))))
        for a, b in zip(lct, rct)
    ]
    f["core_len_diff"] = np.abs(lc.str.len().values - rc.str.len().values)
    f["core_tok_diff"] = np.abs(
        np.array([len(a) for a in lct]) - np.array([len(b) for b in rct])
    )

    # acronym match: "ich" vs "indian coffee house"
    la, ra = left["_acronym"].fillna(""), right["_acronym"].fillna("")
    f["acronym_eq"] = (la.values == ra.values).astype(float)
    f["acronym_vs_core"] = [
        float(bool(a) and (a == "".join(t[0] for t in sorted(b) if t)))
        for a, b in zip(la, rct)
    ]

    # --- address similarity ---
    laddr, raddr = left["_core_addr"].fillna(""), right["_core_addr"].fillna("")
    f["addr_ratio"] = [fuzz.ratio(a, b) / 100 for a, b in zip(laddr, raddr)]
    f["addr_token_sort"] = [fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(laddr, raddr)]
    f["addr_token_set"] = [fuzz.token_set_ratio(a, b) / 100 for a, b in zip(laddr, raddr)]

    lat = [set(x.split()) for x in laddr]
    rat = [set(x.split()) for x in raddr]
    f["addr_jaccard"] = [_jaccard(a, b) for a, b in zip(lat, rat)]
    f["addr_containment"] = [_containment(a, b) for a, b in zip(lat, rat)]
    f["addr_len_diff"] = np.abs(laddr.str.len().values - raddr.str.len().values)

    # --- numeric agreement: PIN codes, house numbers ---
    f["addr_num_agree"] = [
        _num_agreement(a, b) for a, b in zip(left["_addr_nums"], right["_addr_nums"])
    ]
    f["addr_num_jaccard"] = [
        _jaccard(a, b) for a, b in zip(left["_addr_nums"], right["_addr_nums"])
    ]
    f["name_num_agree"] = [
        _num_agreement(a, b) for a, b in zip(left["_name_nums"], right["_name_nums"])
    ]

    # --- country agreement (as a boolean, never as a category) ---
    f["country_eq"] = (
        left[C.COUNTRY].astype(str).values == right[C.COUNTRY].astype(str).values
    ).astype(float)

    # --- combined views ---
    f["name_addr_mean"] = (f["core_token_sort"] + f["addr_token_set"]) / 2
    f["name_addr_min"] = np.minimum(f["core_token_sort"], f["addr_token_set"])

    return f.fillna(0.0)


def add_rank_features(feat: pd.DataFrame, pairs: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Per-Source-1-entity competitive features.

    Whether a candidate is the *best* option for this entity matters more than
    its absolute similarity — a 0.8 that ranks 7th is a different thing from a
    0.8 that ranks 1st with a big gap to 2nd.
    """
    df = feat.copy()
    g = pairs.assign(_s=df[score_col].values).groupby("s1_id")["_s"]
    df["rank_in_entity"] = g.rank(ascending=False, method="first").values
    df["max_in_entity"] = g.transform("max").values
    df["gap_to_best"] = df["max_in_entity"] - df[score_col].values
    df["n_candidates"] = g.transform("size").values
    df["is_best"] = (df["rank_in_entity"] == 1).astype(float)
    return df
