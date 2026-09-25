"""Vectorized pair features — same columns and values as features.build_pair_features
(asserted by tests/test_features_v2.py), plus precision features.

Why: the per-pair Python loops in features.py cost ~90 min on the 52M test pairs.
Here every normalized form is computed once per *record* (in worker processes),
string scores run through rapidfuzz.process.cpdist (C++, multi-threaded), and
token-set overlaps are row-wise products of sparse binary matrices.

New features (precision-oriented, see docs/master-plan/LLD.md §5):
  tl_tset, tl_jw          transliterated core name (Indic scripts, domains)
  skel_jw, skel_eq        consonant skeleton of that name
  either_native           either side was written in a non-Latin script
  hn_state, hn_lev, hn_near  first house number: agree / disagree / typo (9914 vs 991)
  legal_compat            0 same or both absent, 1 one side absent, 2 conflicting classes
  name_weighted_overlap        IDF-weighted core-name overlap (generic words count little)
  name_genericity         how common this S1 core name is in its split (per million S1)
  colocation              how many S1 share this S1's core address (per million S1)
  domain_ratio            domain stem vs the other side's core name without spaces (-1 if not a domain)
  addr_empty_any          either side has no address
Source prefix and country literals are never used (features.py design rule).
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
import scipy.sparse as sp

import config as C
import normalize as N

PAIR_CHUNK = 2_000_000
_WORKERS = C.WORKERS
_REC_CHUNK = 100_000

OLD_COLUMNS = [
    "block_sim", "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "core_ratio", "core_token_sort", "core_jw", "core_jaccard", "core_containment",
    "core_first_tok_eq", "core_len_diff", "core_tok_diff", "acronym_eq", "acronym_vs_core",
    "addr_ratio", "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_containment",
    "addr_len_diff", "addr_num_agree", "addr_num_jaccard", "name_num_agree", "country_eq",
    "name_addr_mean", "name_addr_min",
]


# ------------------------------------------------------------------ per record
def _record_rows(args):
    names, addrs = args
    rows = []
    for n, a in zip(names, addrs):
        core = N.core_name(n)
        toks = set(core.split())
        tl = N.translit_core(n)
        rows.append((
            N.norm_name(n), core, N.norm_addr(a), N.core_addr(a), N.acronym(n),
            " ".join(sorted(N.numeric_tokens(n))), " ".join(sorted(N.numeric_tokens(a))),
            min(toks) if toks else "", "".join(t[0] for t in sorted(toks) if t),
            tl, N.skeleton(tl), N.legal_form(n), N.first_number(a), N.domain_stem(n),
            N.is_native_script(n), N.basic_clean(a) == "",
        ))
    return rows


_REC_COLS = ["_name", "_core_name", "_addr", "_core_addr", "_acronym", "_name_nums",
             "_addr_nums", "_core_min", "_core_initials", "_tl_core", "_skel", "_legal",
             "_hn", "_dom", "_native", "_addr_empty"]


def record_table(df: pd.DataFrame, workers=_WORKERS) -> pd.DataFrame:
    """Every normalized form a pair feature needs, one row per record, indexed by entity_id."""
    names = df[C.NAME].fillna("").astype(str).tolist()
    addrs = df[C.ADDR].fillna("").astype(str).tolist()
    jobs = [(names[i:i + _REC_CHUNK], addrs[i:i + _REC_CHUNK]) for i in range(0, len(names), _REC_CHUNK)]
    if len(jobs) > 1 and workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_record_rows, jobs)
    else:
        parts = [_record_rows(j) for j in jobs]
    rec = pd.DataFrame([r for p in parts for r in p], columns=_REC_COLS)
    rec.index = pd.Index(df[C.ID].astype(str).to_numpy(), name=C.ID)
    rec[C.COUNTRY] = df[C.COUNTRY].astype(str).to_numpy()
    return rec


def load_records(paths, ids) -> pd.DataFrame:
    """Rows of the given source files whose entity_id is in `ids` (polars filter, one pass per file)."""
    from ingest import read_polars
    ids = pl.Series(list(ids), dtype=pl.String)
    parts = [read_polars(p).filter(pl.col(C.ID).is_in(ids.implode())).to_pandas() for p in paths]
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ split stats
def split_stats(s1_path, workers=_WORKERS) -> dict:
    """Label-free statistics over the FULL Source-1 file of a split (train or test).

    Uses only S1 text, never labels, so it is legitimate on test and cannot leak.
    """
    from ingest import read_polars
    df = read_polars(s1_path)
    names, addrs = df[C.NAME].to_list(), df[C.ADDR].to_list()
    jobs = [names[i:i + _REC_CHUNK] for i in range(0, len(names), _REC_CHUNK)]
    ajobs = [addrs[i:i + _REC_CHUNK] for i in range(0, len(addrs), _REC_CHUNK)]
    with Pool(workers) as pool:
        cores = [c for p in pool.map(_cores, jobs) for c in p]
        caddr = [c for p in pool.map(_caddrs, ajobs) for c in p]
    n = len(cores)
    tok_df = pd.Series([t for c in cores for t in set(c.split())]).value_counts()
    return {
        "n": n,
        "name_count": pd.Series(cores).value_counts(),
        "addr_count": pd.Series(caddr).value_counts(),
        "tok_df": tok_df,
    }


def _cores(names):
    return [N.core_name(x) for x in names]


def _caddrs(addrs):
    return [N.core_addr(x) for x in addrs]


# ------------------------------------------------------------------ pair helpers
def _cp(a, b, scorer, scale=1.0):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float64) * scale


def _bin_matrix(texts_l, texts_r):
    """Binary token incidence (whitespace tokens, duplicates collapsed) for two record
    tables over one shared vocabulary. One pass, one dict lookup per token — sklearn's
    CountVectorizer spent ~90% of feature time here tokenizing everything twice."""
    vocab = {}

    def build(texts):
        indptr, indices = [0], []
        for t in texts:
            indices.extend({vocab.setdefault(w, len(vocab)) for w in t.split()})
            indptr.append(len(indices))
        return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)

    (pl_, il), (pr_, ir) = build(texts_l), build(texts_r)
    v = max(len(vocab), 1)
    ML = sp.csr_matrix((np.ones(len(il), np.float32), il, pl_), shape=(len(texts_l), v))
    MR = sp.csr_matrix((np.ones(len(ir), np.float32), ir, pr_), shape=(len(texts_r), v))
    ML.sort_indices(); MR.sort_indices()
    return ML, MR, vocab


def _overlap(ML, MR, li, ri):
    """|A∩B|, |A|, |B| per pair, from binary incidence rows."""
    inter = np.asarray(ML[li].multiply(MR[ri]).sum(axis=1)).ravel()
    na = np.diff(ML.indptr)[li].astype(np.float64)
    nb = np.diff(MR.indptr)[ri].astype(np.float64)
    return inter, na, nb


def _jaccard(inter, na, nb):
    out = np.zeros_like(inter)
    both = (na > 0) & (nb > 0)
    out[both] = inter[both] / (na[both] + nb[both] - inter[both])
    out[(na == 0) & (nb == 0)] = 1.0
    return out


def _containment(inter, na, nb):
    out = np.zeros_like(inter)
    both = (na > 0) & (nb > 0)
    out[both] = inter[both] / np.minimum(na[both], nb[both])
    return out


def _agree(inter, na, nb):
    out = np.zeros_like(inter)
    both = (na > 0) & (nb > 0)
    out[both] = np.where(inter[both] > 0, 1.0, -1.0)
    return out


# ------------------------------------------------------------------ main entry
def build_pair_features(pairs: pd.DataFrame, L: pd.DataFrame, R: pd.DataFrame,
                        stats: dict | None = None, extra: bool = True) -> pd.DataFrame:
    """pairs: [s1_id, cand_id, block_sim]; L/R: record_table() of S1 and S2+S3.

    extra=False returns exactly features.build_pair_features' columns.
    """
    li = L.index.get_indexer(pairs["s1_id"].astype(str))
    ri = R.index.get_indexer(pairs["cand_id"].astype(str))
    if (li < 0).any() or (ri < 0).any():
        raise KeyError(f"{int((li < 0).sum())} S1 / {int((ri < 0).sum())} candidate ids have no record")

    mats = {c: _bin_matrix(L[c].to_numpy(), R[c].to_numpy())
            for c in ("_core_name", "_core_addr", "_addr_nums", "_name_nums")}
    if extra and stats is not None:
        vocab = mats["_core_name"][2]
        dfv = stats["tok_df"].reindex(list(vocab.keys())).fillna(0).to_numpy()
        idf = np.zeros(len(vocab))
        idf[list(vocab.values())] = np.log((stats["n"] + 1) / (dfv + 1))

    out = []
    for a in range(0, len(pairs), PAIR_CHUNK):
        l, r = li[a:a + PAIR_CHUNK], ri[a:a + PAIR_CHUNK]
        g = lambda T, c, ix: T[c].to_numpy()[ix]
        f = {"block_sim": pairs["block_sim"].to_numpy()[a:a + PAIR_CHUNK].astype(np.float64)}
        ln, rn = g(L, "_name", l), g(R, "_name", r)
        lc, rc = g(L, "_core_name", l), g(R, "_core_name", r)
        f["name_ratio"] = _cp(ln, rn, fuzz.ratio, 0.01)
        f["name_token_sort"] = _cp(ln, rn, fuzz.token_sort_ratio, 0.01)
        f["name_token_set"] = _cp(ln, rn, fuzz.token_set_ratio, 0.01)
        f["name_partial"] = _cp(ln, rn, fuzz.partial_ratio, 0.01)
        f["core_ratio"] = _cp(lc, rc, fuzz.ratio, 0.01)
        f["core_token_sort"] = _cp(lc, rc, fuzz.token_sort_ratio, 0.01)
        f["core_jw"] = _cp(lc, rc, JaroWinkler.similarity)

        ML, MR, _ = mats["_core_name"]
        inter, na, nb = _overlap(ML, MR, l, r)
        f["core_jaccard"] = _jaccard(inter, na, nb)
        f["core_containment"] = _containment(inter, na, nb)
        lm, rm = g(L, "_core_min", l), g(R, "_core_min", r)
        f["core_first_tok_eq"] = ((lm != "") & (rm != "") & (lm == rm)).astype(np.float64)
        f["core_len_diff"] = np.abs(L["_core_name"].str.len().to_numpy()[l] - R["_core_name"].str.len().to_numpy()[r])
        f["core_tok_diff"] = np.abs(na - nb)

        la, ra = g(L, "_acronym", l), g(R, "_acronym", r)
        f["acronym_eq"] = (la == ra).astype(np.float64)
        f["acronym_vs_core"] = ((la != "") & (la == g(R, "_core_initials", r))).astype(np.float64)

        lad, rad = g(L, "_core_addr", l), g(R, "_core_addr", r)
        f["addr_ratio"] = _cp(lad, rad, fuzz.ratio, 0.01)
        f["addr_token_sort"] = _cp(lad, rad, fuzz.token_sort_ratio, 0.01)
        f["addr_token_set"] = _cp(lad, rad, fuzz.token_set_ratio, 0.01)
        ML, MR, _ = mats["_core_addr"]
        ai, aa, ab = _overlap(ML, MR, l, r)
        f["addr_jaccard"] = _jaccard(ai, aa, ab)
        f["addr_containment"] = _containment(ai, aa, ab)
        f["addr_len_diff"] = np.abs(L["_core_addr"].str.len().to_numpy()[l] - R["_core_addr"].str.len().to_numpy()[r])

        ML, MR, _ = mats["_addr_nums"]
        ni, nna, nnb = _overlap(ML, MR, l, r)
        f["addr_num_agree"] = _agree(ni, nna, nnb)
        f["addr_num_jaccard"] = _jaccard(ni, nna, nnb)
        ML, MR, _ = mats["_name_nums"]
        f["name_num_agree"] = _agree(*_overlap(ML, MR, l, r))
        f["country_eq"] = (g(L, C.COUNTRY, l) == g(R, C.COUNTRY, r)).astype(np.float64)
        f["name_addr_mean"] = (f["core_token_sort"] + f["addr_token_set"]) / 2
        f["name_addr_min"] = np.minimum(f["core_token_sort"], f["addr_token_set"])

        if extra:
            lt, rt = g(L, "_tl_core", l), g(R, "_tl_core", r)
            f["tl_tset"] = _cp(lt, rt, fuzz.token_set_ratio, 0.01)
            f["tl_jw"] = _cp(lt, rt, JaroWinkler.similarity)
            ls, rs = g(L, "_skel", l), g(R, "_skel", r)
            f["skel_jw"] = _cp(ls, rs, JaroWinkler.similarity)
            f["skel_eq"] = ((ls != "") & (ls == rs)).astype(np.float64)
            f["either_native"] = (g(L, "_native", l) | g(R, "_native", r)).astype(np.float64)

            lh, rh = g(L, "_hn", l), g(R, "_hn", r)
            both = (lh != "") & (rh != "")
            lev = _cp(lh, rh, Levenshtein.distance)
            f["hn_state"] = np.where(both, np.where(lh == rh, 1.0, -1.0), 0.0)
            f["hn_lev"] = np.where(both, lev, -1.0)
            f["hn_near"] = (both & (lh != rh) & (lev <= 1)).astype(np.float64)

            lg, rg = g(L, "_legal", l), g(R, "_legal", r)
            f["legal_compat"] = np.where((lg == rg), 0.0, np.where((lg == "") | (rg == ""), 1.0, 2.0))

            rd = g(R, "_dom", r)
            nospace = np.array([s.replace(" ", "") for s in lc], dtype=object)
            dr = _cp(nospace, rd, fuzz.ratio, 0.01)
            f["domain_ratio"] = np.where(rd != "", dr, -1.0)
            f["addr_empty_any"] = (g(L, "_addr_empty", l) | g(R, "_addr_empty", r)).astype(np.float64)

            if stats is not None:
                A, B = mats["_core_name"][0][l], mats["_core_name"][1][r]
                shared = A.multiply(B) @ idf
                union = A @ idf + B @ idf - shared
                f["name_weighted_overlap"] = np.where(union > 0, shared / np.where(union > 0, union, 1), 0.0)
                per_m = 1e6 / stats["n"]
                f["name_genericity"] = np.log1p(stats["name_count"].reindex(lc).fillna(0).to_numpy() * per_m)
                f["colocation"] = np.where(lad != "", np.log1p(
                    stats["addr_count"].reindex(lad).fillna(0).to_numpy() * per_m), 0.0)
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True).fillna(0.0)
