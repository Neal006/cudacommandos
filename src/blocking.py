"""Candidate generation at 10M-record scale.

Blocking sets the recall ceiling for the whole pipeline: a true pair that never
becomes a candidate cannot be recovered downstream. Measure the ceiling first,
then tune.

Scale drives the design. Test is ~1.7M Source-1 entities against ~10M S2+S3
records, so any approach that materializes a dense similarity matrix — sklearn
NearestNeighbors included — is out by many orders of magnitude.

What works instead: word-level TF-IDF with aggressive document-frequency
pruning, then chunked sparse matrix multiplication. Two records only produce a
nonzero if they share a surviving (rare) token, so the product stays sparse and
memory tracks the number of real co-occurrences rather than n x m.

Character n-grams are deliberately NOT used here — they explode the nonzero
count at this scale. They earn their keep later, in features.py, where they run
over a few hundred candidates per entity instead of ten million.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C


def build_vectorizer(corpus, max_df, min_df=2):
    """TF-IDF over word tokens, pruning tokens that are too common to block on.

    max_df is the lever that keeps the sparse product tractable: a token
    appearing in a large fraction of records ("restaurant", "road", "pvt")
    generates enormous posting lists and almost no discriminative signal.
    Rows come out L2-normalized, so a dot product is cosine similarity.
    """
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-z0-9]+",
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit(corpus)


def _topk_from_sparse_rows(mat, index_ids, top_k):
    """Top-k by score for each row of a CSR similarity block."""
    out = []
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    for r in range(mat.shape[0]):
        lo, hi = indptr[r], indptr[r + 1]
        if lo == hi:
            out.append([])
            continue
        cols = indices[lo:hi]
        vals = data[lo:hi]
        if hi - lo > top_k:
            sel = np.argpartition(vals, -top_k)[-top_k:]
            cols, vals = cols[sel], vals[sel]
        order = np.argsort(-vals)
        out.append([(index_ids[cols[i]], float(vals[i])) for i in order])
    return out


def _block_pair(q_df, i_df, vec, top_k, chunk_size, progress, label):
    """Candidates for one (queries, index) partition."""
    res = {}
    if len(q_df) == 0:
        return res
    if len(i_df) == 0:
        return {sid: [] for sid in q_df[C.ID]}

    XI = vec.transform(i_df["_blob"]).T.tocsc()   # V x m, transposed once
    index_ids = i_df[C.ID].to_numpy()
    q_ids = q_df[C.ID].to_numpy()
    blobs = q_df["_blob"].to_numpy()

    for start in range(0, len(q_df), chunk_size):
        stop = min(start + chunk_size, len(q_df))
        XQ = vec.transform(blobs[start:stop])     # c x V
        sims = (XQ @ XI).tocsr()                  # c x m, sparse
        for sid, hits in zip(q_ids[start:stop], _topk_from_sparse_rows(sims, index_ids, top_k)):
            res[sid] = hits
        if progress and (start // chunk_size) % 10 == 0:
            print(f"    {label}: {stop:,}/{len(q_df):,}", flush=True)
    return res


def generate_candidates(s1, s2, s3, top_k=None, within_country=None,
                        max_df=None, chunk_size=None, progress=True):
    """Candidates for every Source-1 entity.

    Inputs must already carry the `_`-prefixed normalized columns from
    normalize.add_norm_columns.

    Returns dict: s1_entity_id -> list of (candidate_id, blocking_similarity).
    Every Source-1 id gets a key even with no candidates — the submission
    format requires one row per entity regardless.
    """
    top_k = C.TOP_K if top_k is None else top_k
    within_country = C.BLOCK_WITHIN_COUNTRY if within_country is None else within_country
    max_df = C.BLOCK_MAX_DF if max_df is None else max_df
    chunk_size = C.BLOCK_CHUNK if chunk_size is None else chunk_size

    others = pd.concat([s2, s3], ignore_index=True)
    vec = build_vectorizer(
        pd.concat([s1["_blob"], others["_blob"]], ignore_index=True), max_df=max_df
    )
    if progress:
        print(f"  vocab: {len(vec.vocabulary_):,} tokens (max_df={max_df})", flush=True)

    out = {sid: [] for sid in s1[C.ID]}

    if within_country:
        # country is an OPEN set of strings. Never enumerate it: the test set
        # contains France, which appears nowhere in training. Grouping by the
        # raw label handles any unseen value for free.
        groups = sorted(set(s1[C.COUNTRY].astype(str)) | set(others[C.COUNTRY].astype(str)))
        parts = [
            (s1[s1[C.COUNTRY].astype(str) == g], others[others[C.COUNTRY].astype(str) == g], g)
            for g in groups
        ]
    else:
        parts = [(s1, others, "all")]

    for q_df, i_df, label in parts:
        if progress:
            print(f"  blocking [{label}]: {len(q_df):,} queries vs {len(i_df):,} records", flush=True)
        out.update(_block_pair(q_df, i_df, vec, top_k, chunk_size, progress, label))

    return out


def candidates_to_frame(cands):
    """Flatten into one row per (s1, candidate) pair."""
    s1_ids, cand_ids, sims = [], [], []
    for s1_id, pairs in cands.items():
        for cand_id, sim in pairs:
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)
            sims.append(sim)
    return pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "block_sim": sims})


def orphan_check(cands, s1_ids):
    """Source-1 entities blocking found nothing for.

    A high orphan count usually means either max_df pruned away every token
    these records had, or the country partition is splitting true pairs apart.
    """
    missing = [i for i in s1_ids if not cands.get(i)]
    return {"n_orphans": len(missing), "examples": missing[:10]}
