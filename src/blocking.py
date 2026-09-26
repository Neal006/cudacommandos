"""Candidate generation at 10M-record scale.

Blocking sets the recall ceiling for the whole pipeline: a true pair that never
becomes a candidate cannot be recovered downstream. Measure the ceiling first,
then tune.

Scale drives every choice here. Test blocking is ~1.7M Source-1 entities
against ~10M S2+S3 records, so anything that materializes a dense similarity
matrix is off by ten orders of magnitude.

What works: word-level TF-IDF with aggressive document-frequency pruning, then
chunked sparse matmul. Two records only produce a nonzero if they share a
surviving (rare) token, so memory tracks real co-occurrences rather than n x m.

Character n-grams are deliberately NOT used here — they explode the nonzero
count at this scale. They earn their keep in features.py, over a few dozen
candidates per entity instead of ten million.

Three things keep this inside a ~10GB RAM budget:
  - the vocabulary is FIT on a sample (DF estimates are statistical)
  - the index matrix is transformed in batches, not one allocation
  - every phase prints progress, because a silent 20-minute step is
    indistinguishable from a hang
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C

try:  # Apache-2.0; multithreaded top-k of A@B without materializing A@B.
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # identical results, ~2.7x slower (measured, 2M-record index)
    sp_matmul_topn = None


def _log(msg):
    print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_vectorizer(corpus, max_df=None, min_df=None, fit_sample=None, seed=C.SEED):
    """TF-IDF over word tokens, fit on a sample of the corpus.

    max_df is the lever that keeps the sparse product tractable: a token in a
    large fraction of records ("restaurant", "road", "pvt") produces enormous
    posting lists and almost no discriminative signal. Rows come out
    L2-normalized, so a dot product is cosine similarity.

    Fitting on a sample matters at this scale — the full 10.5M-document fit is
    a silent, single-threaded pass that also builds a vocabulary dict large
    enough to matter against the RAM budget.
    """
    max_df = C.BLOCK_MAX_DF if max_df is None else max_df
    min_df = C.BLOCK_MIN_DF if min_df is None else min_df
    fit_sample = C.BLOCK_FIT_SAMPLE if fit_sample is None else fit_sample

    corpus = pd.Series(corpus)
    if fit_sample and len(corpus) > fit_sample:
        fit_on = corpus.sample(n=fit_sample, random_state=seed)
        _log(f"fitting vocabulary on {len(fit_on):,} of {len(corpus):,} documents")
    else:
        fit_on = corpus
        _log(f"fitting vocabulary on all {len(fit_on):,} documents")

    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-z0-9]+",
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit(fit_on)
    _log(f"vocabulary: {len(vec.vocabulary_):,} tokens "
         f"(max_df={max_df}, min_df={min_df}) in {time.time()-t0:.0f}s")
    return vec


def transform_batched(vec, texts, batch=None, label="index"):
    """Transform in batches and stack. Bounds peak memory during the transform.

    Transforming 6M documents in one call allocates the whole result at once;
    batching keeps the high-water mark to one batch plus the accumulated
    (already compact) sparse blocks.
    """
    batch = C.BLOCK_INDEX_BATCH if batch is None else batch
    # Any positional-sliceable sequence (numpy, list, pandas ExtensionArray).
    # Not converted to one big object array: for Arrow-backed strings that
    # would materialize ~10M Python strings at once.
    if len(texts) <= batch:
        return vec.transform(texts)

    parts = []
    t0 = time.time()
    for start in range(0, len(texts), batch):
        parts.append(vec.transform(texts[start:start + batch]))
        done = min(start + batch, len(texts))
        _log(f"{label} transform {done:,}/{len(texts):,} "
             f"({time.time()-t0:.0f}s elapsed)")
    return sp.vstack(parts, format="csr")


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


def _block_pair(q_df, i_df, vec, top_k, chunk_size, label):
    """Candidates for one (queries, index) partition."""
    res = {}
    if len(q_df) == 0:
        return res
    if len(i_df) == 0:
        return {sid: [] for sid in q_df[C.ID]}

    XI = transform_batched(vec, i_df["_blob"].array, label=f"{label} index").T
    # sparse_dot_topn wants B as CSR and converts on EVERY call otherwise, so
    # pay the transpose copy once here. scipy's product wants CSC (a free view).
    XI = XI.tocsr() if sp_matmul_topn is not None else XI.tocsc()
    index_ids = i_df[C.ID].to_numpy()
    q_ids = q_df[C.ID].to_numpy()
    blobs = q_df["_blob"].to_numpy()

    _log(f"{label}: scoring {len(q_df):,} queries against {len(i_df):,} records")
    t0 = time.time()
    for start in range(0, len(q_df), chunk_size):
        stop = min(start + chunk_size, len(q_df))
        q = vec.transform(blobs[start:stop])
        if sp_matmul_topn is not None:
            sims = sp_matmul_topn(q, XI, top_n=top_k, n_threads=C.BLOCK_THREADS)
        else:
            sims = (q @ XI).tocsr()
        for sid, hits in zip(q_ids[start:stop], _topk_from_sparse_rows(sims, index_ids, top_k)):
            res[sid] = hits
        if (start // chunk_size) % 5 == 0:
            rate = stop / max(time.time() - t0, 1e-9)
            eta = (len(q_df) - stop) / max(rate, 1e-9)
            _log(f"{label}: {stop:,}/{len(q_df):,}  ({rate:,.0f} q/s, ETA {eta/60:.1f} min)")
    _log(f"{label}: done in {(time.time()-t0)/60:.1f} min")
    del XI
    return res


def generate_candidates(s1, s2, s3, top_k=None, within_country=None,
                        max_df=None, chunk_size=None, progress=True):
    """Candidates for every Source-1 entity.

    Inputs must already carry `_blob` from normalize.add_blocking_columns.

    Returns dict: s1_entity_id -> list of (candidate_id, blocking_similarity).
    Every Source-1 id gets a key even with no candidates — the submission
    format requires one row per entity regardless.
    """
    top_k = C.TOP_K if top_k is None else top_k
    within_country = C.BLOCK_WITHIN_COUNTRY if within_country is None else within_country
    chunk_size = C.BLOCK_CHUNK if chunk_size is None else chunk_size

    others = pd.concat([s2, s3], ignore_index=True)
    vec = build_vectorizer(others["_blob"], max_df=max_df)

    out = {sid: [] for sid in s1[C.ID]}

    if within_country:
        # country is an OPEN set of strings. Never enumerate it: test contains
        # France, which appears nowhere in training. Grouping on the raw label
        # handles any unseen value for free.
        groups = sorted(set(s1[C.COUNTRY].astype(str)) | set(others[C.COUNTRY].astype(str)))
        parts = [
            (s1[s1[C.COUNTRY].astype(str) == g], others[others[C.COUNTRY].astype(str) == g], g)
            for g in groups
        ]
    else:
        parts = [(s1, others, "all")]

    for q_df, i_df, label in parts:
        out.update(_block_pair(q_df, i_df, vec, top_k, chunk_size, label))

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

    A high orphan count usually means either max_df/min_df pruned away every
    token those records had, or the country partition is splitting true pairs.
    """
    missing = [i for i in s1_ids if not cands.get(i)]
    return {"n_orphans": len(missing), "examples": missing[:10]}
