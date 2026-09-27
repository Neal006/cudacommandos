"""Union blocking: TF-IDF (existing) + skeleton/translit (new) for India.

Why this exists: normalize.skeleton()'s own docstring says native-script India
positives share ~0% raw name tokens with their true match, but >=1 skeleton
token 94.4% of the time. blocking.py's `_blob` is built from raw core_name
tokens (normalize.add_blocking_columns), so standard TF-IDF blocking is almost
certainly dropping the bulk of India's true pairs before they ever reach the
classifier or reranker -- no downstream model can recover a pair that was
never made a candidate.

This module does NOT change blocking.py. It runs the existing TF-IDF pass
unchanged, then a second pass using skeleton(translit_core(name)) as the
blocking text, restricted to India (the only country where translit_core does
real work -- for Latin-script countries this second pass is a near no-op).
Candidates from both passes are unioned per s1 entity.

IMPORTANT: by the time s1/s2/s3 reach here, data.load_split has already run
add_blocking_columns, which builds `_blob` and then DROPS the raw name/address
columns (BLOCK_DROP_TEXT) to stay inside the RAM budget over the full 10M-record
set. So the raw text needed for skeleton/translit is gone from these frames --
this module re-reads it from disk for just the (small) India subset, the same
technique run_pipeline.featurize() already uses for post-blocking candidates.

Everything else here reuses blocking.py's already-tested primitives
(build_vectorizer, _block_pair) with a different `_blob` column -- no new
blocking math, just a new signal for the same machinery.

`is_skel_source`: candidates found ONLY by the pass-2 skeleton pass (not by
TF-IDF) are tracked in a module-level set returned alongside `cands`, so
downstream featurization can add a binary flag letting GBDT distinguish
pass-1 vs pass-2 candidates explicitly, rather than inferring it implicitly.
"""
import pandas as pd

import config as C
import blocking as B
import data as D
from normalize import translit_core, skeleton, core_addr


def _reread_india_text(which, ids, is_s1):
    """Re-read raw NAME/ADDR/COUNTRY for the given ids from the source files.

    is_s1=True reads only the source1 file; is_s1=False reads source2+source3
    (matching how `others` is built as concat(s2, s3) everywhere else).
    """
    s1_path, s2_path, s3_path = D.source_paths(which)
    if is_s1:
        return D.load_records_by_id(s1_path, ids)
    parts = [D.load_records_by_id(p, ids) for p in (s2_path, s3_path)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[C.ID, C.NAME, C.ADDR, C.COUNTRY])


def _skeleton_blob(df):
    """Build the skeleton/translit blocking text from freshly re-read raw text."""
    df = df.copy()
    skel = df[C.NAME].map(lambda s: skeleton(translit_core(s)))
    addr = df[C.ADDR].map(core_addr)
    df["_blob"] = (skel + " " + addr).str.strip()
    return df


def generate_candidates_union(s1, s2, s3, which, top_k=None, skel_top_k=None,
                               max_df=None, skel_max_df=None, chunk_size=None,
                               india_label="India"):
    """Drop-in replacement for blocking.generate_candidates.

    `which` must be "train" or "test" -- needed to re-read raw text for the
    India subset from the right source files (data.source_paths).

    Returns (cands, skel_source_ids):
      - cands: same shape as blocking.generate_candidates -- dict s1_id ->
        list of (cand_id, block_sim), every s1_id present.
      - skel_source_ids: set of cand_ids that were found ONLY by pass 2
        (the skeleton pass), i.e. NOT already present in pass-1 TF-IDF
        results for that s1_id. Used to build the is_skel_source feature.

    Pass 1 is the existing TF-IDF blocking, completely unchanged. Pass 2 runs
    the same _block_pair machinery on a skeleton blob, restricted to India,
    and is unioned in. A candidate found by both passes keeps its TF-IDF
    score; one found only by pass 2 gets its skeleton similarity as
    block_sim, so it stays usable as a model feature, AND is flagged in
    skel_source_ids.
    """
    others = pd.concat([s2, s3], ignore_index=True)

    # Pass 1: existing, unchanged.
    B._log("union blocking: pass 1 (TF-IDF, unchanged)")
    cands = B.generate_candidates(s1, s2, s3, top_k=top_k, max_df=max_df,
                                   chunk_size=chunk_size)

    skel_source_ids = set()

    # Pass 2: skeleton/translit, India only. COUNTRY survives add_blocking_columns
    # (only NAME/ADDR are dropped), so this filter works on the frames as given.
    s1_india_ids = set(s1.loc[s1[C.COUNTRY].astype(str) == india_label, C.ID])
    oth_india_ids = set(others.loc[others[C.COUNTRY].astype(str) == india_label, C.ID])
    if not s1_india_ids:
        B._log(f"union blocking: no '{india_label}' rows in this split, skipping pass 2")
        return cands, skel_source_ids

    B._log(f"union blocking: pass 2 -- re-reading raw text for "
           f"{len(s1_india_ids):,} S1 and {len(oth_india_ids):,} S2/S3 "
           f"{india_label} records")
    s1_raw = _reread_india_text(which, s1_india_ids, is_s1=True)
    oth_raw = _reread_india_text(which, oth_india_ids, is_s1=False)
    if len(s1_raw) == 0 or len(oth_raw) == 0:
        B._log("union blocking: re-read returned no rows, skipping pass 2")
        return cands, skel_source_ids

    s1_skel = _skeleton_blob(s1_raw)
    oth_skel = _skeleton_blob(oth_raw)

    B._log(f"union blocking: pass 2 (skeleton, {india_label} only, "
           f"{len(s1_skel):,} queries vs {len(oth_skel):,} records)")
    vec = B.build_vectorizer(oth_skel["_blob"], max_df=skel_max_df or max_df)
    skel_top_k = skel_top_k or (top_k or C.TOP_K)
    skel_cands = B._block_pair(s1_skel, oth_skel, vec, skel_top_k,
                                chunk_size or C.BLOCK_CHUNK, label="skel_India")

    # Union: merge pass-2 hits into pass-1 results, dedup by cand_id.
    added = 0
    for sid, hits in skel_cands.items():
        existing_ids = {cid for cid, _ in cands.get(sid, [])}
        new_hits = [(cid, sim) for cid, sim in hits if cid not in existing_ids]
        if new_hits:
            cands[sid] = cands.get(sid, []) + new_hits
            skel_source_ids.update(cid for cid, _ in new_hits)
            added += len(new_hits)
    B._log(f"union blocking: pass 2 added {added:,} new candidate pairs "
           f"across {len(s1_india_ids):,} {india_label} entities")

    return cands, skel_source_ids