# TECH_DEBT_AUDIT.md

Branch `nealultraprohopeso` · audited 2026-09-27 · scope `src/`, `src/gpu/`, `tests/`, `tools/eda/`, `requirements.txt`

**How this was produced:**
- ECC `python-reviewer` scan, with `ruff check` and an AST function-length walk.
- ECC `mle-reviewer` (leakage, compute fit), ECC `silent-failure-hunter`, and ECC `code-reviewer` (correctness of the branch diff).
- A 5-advisor LLM council with peer review, on whether to launch.

**Status column:**
- **FIXED** = changed on this branch and covered by `tests/test_review_fixes.py`.
- **OPEN** = documented, not changed today.

---

## Executive summary

The pipeline is fit to run. The code is sound on the two things that decide the score:
- **No label leakage** in the new passes, the gang features or the holdout. The MLE review verified this.
- **Alignment and caching are guarded against silent misuse.**

The debt is concentrated in three places:
1. **Two long `main()` functions** (`run_v4.py` ≈250 lines, `run_v2.py` 140) that hold the whole flow and can only be tested end to end.
2. **No automated test of the I/O boundary**: blocking, candidate caching, `write_outputs`, and `predict_test_chunked` with the reranker.
3. **Cache keys are built by hand in four places**, and none of them include a normalizer version.

None of these block today's run. All are real risk for the next person who changes a parameter.

The biggest *operational* risk is not code but wall-clock. The chunk loops run one after another, so 192 cores help inside each chunk, not across chunks. The runbook in `context_hopeso.md` sizes chunks for the big box, and a local 90k end-to-end check measures the real effect before anyone spends SageMaker hours.

---

## Findings ranked by risk to production

### HIGH

| # | Area | File:line | Finding | Status |
|---|---|---|---|---|
| H0 | Silent failure | `src/score_test.py:71` (before: `cached_candidates`) | `score_test` always scored the **base** test frame. A model trained with `--cands-tag hopeso` would be scored without the recovered candidates. The output is valid and the recall gain is silently gone. | **FIXED**: shared `hopeso.load_frame`; v4 stores `cands_tag` in `model.pkl` and `score_test` refuses a mismatch (`src/score_test.py:55-58`) |
| H1 | Brittle | `src/hopeso.py:167` | Tagged candidate frame name did not encode the pass settings (`VIBE`). Retuning and forgetting to rebuild would silently train on a stale frame. | **FIXED**: settings are in the file name (`_t3c50d10`) |
| H2 | Silent failure | `src/hopeso.py` `fill_sim` | Unknown ids became empty blobs, which gave `block_sim = 0`, a real model feature that looks legitimate. | **FIXED**: refuses with a count |
| H3 | Coverage | `src/run_v2.py:85` `predict_test_chunked` | 118 lines, nesting 3. The reranker band path and the cache resume path have no test. `test_chunk_equiv.py` covers chunking only. | OPEN |
| H4 | Coverage | `src/blocking.py:161`, `src/run_pipeline.py:37`, `src/data.py:151` | The recall-critical top-k, the shared candidate cache and the submission writer have zero tests. | OPEN (`data.py` row check added, see L6) |
| H5 | Complexity | `src/run_v4.py:173` `main`, `src/run_v2.py:264` `main` | Whole pipeline inline; only testable end to end. | OPEN |
| H6 | Operational | `src/run_v2.py:161-193`, `src/run_v4.py:131-170` | Chunk loops are sequential. On 192 cores the per-chunk overhead dominates at the laptop chunk size. | **Mitigated**: runbook sets `--chunk 4000000` and `AMLC_WORKERS`; parallel loop is OPEN |

### MEDIUM

| # | Area | File:line | Finding | Status |
|---|---|---|---|---|
| M1 | Coupling | `src/run_v4.py:177-179` (before) | v4 mutated `run_v2.MAX_ROUNDS`, a global in another module. | **FIXED**: `fit_cv(..., rounds=)` at `src/run_v2.py:210` |
| M2 | Duplication | `src/run_pipeline.py:50`, `src/hopeso.py:159,167`, `src/gpu/embed_knn.py:106` | The candidate cache file name is hand-built in 4 places. The key has no normalizer version (also an AGENTS.md gotcha). | OPEN |
| M3 | Duplication | `run_v2.py:389-399`, `run_v4.py:401-410`, `score_test.py:103-113`, `redecide.py:116-127`, `ensemble.py:143-154` | The same "select → matches/cand sets → write → per-country rates" block appears 5 times. | OPEN |
| M4 | Duplication | `src/gpu/reranker.py:134`, `src/gpu/laya_rr.py:227` | Near-identical `train()` loops. laya was rejected in the A/B but is still live. | OPEN |
| M5 | Silent failure | `src/data.py` `check_score_meta` | A missing sidecar only warns. This is deliberate, so 003's `testp_f550ffb02552.npy` stays usable. | OPEN (by design) |
| M6 | Silent failure | `src/run_pipeline.py:221` | Legacy path zero-fills missing feature columns. v2/v4 select `X[feat]` by name and fail loudly instead. | OPEN (legacy only) |
| M7 | Leakage | `src/run_v4.py` reranker guard | The guard checks S1 **entity** overlap, not S2/S3 **record** overlap (a known AGENTS.md gotcha). | OPEN |
| M8 | Perf | `src/hopeso.py` `fill_sim` | Re-tokenized every duplicated row, single-threaded. | **FIXED**: each unique blob is transformed once, rows gathered by position |
| M9 | Perf | `src/stage2.py` `gang` | A Python per-row loop to test for empty keys. | **FIXED**: vectorized |
| M10 | Brittle | `src/run_v4.py:229,278` | The test contention `5.549` literal was repeated. | **FIXED**: `TEST_CONTENTION` constant (`src/run_v4.py:88`) |

### LOW

| # | Area | File:line | Finding | Status |
|---|---|---|---|---|
| L1 | Brittle | `sys.path.insert` in 9 `src` files (e.g. `src/run_v4.py:65`) and all tests | No package install; boilerplate in every script. | OPEN |
| L2 | Brittle | `src/gpu/embed_knn.py:139` | Test-mix weights `{"US": 0.383, "India": 0.468}` are hardcoded. | OPEN |
| L3 | Brittle | `src/hopeso.py:42` `VIBE` | Caps chosen from one 30k measurement. That is an assumption, not a law (see the note at the end). | OPEN, documented |
| L4 | Deps | `requirements.txt` | No `pytest`. Tests are assert scripts. `groupby().apply(set)` is used widely and is the pandas-3 break class. | OPEN |
| L5 | Dead code | `src/run_v4.py:78` `entity_f05` | Unused import. | **FIXED** |
| L6 | Dead code | `src/data.py:154` | The `c = write_submission(...)` result was unused. | **FIXED**: now checks both files have equal rows |
| L7 | Dead code | `src/features_v2.py:22` `os`, `src/analyze_blocking.py:29`, `tools/eda/design_eda.py:2` | Unused imports (ruff F401). | OPEN |
| L8 | Clutter | `src/run_pipeline.py`, `src/features.py` | Superseded by v2/v4, but still imported for `cached_candidates` and `OLD_COLUMNS`. | OPEN |
| L9 | Style | `src/features_v2.py:165,221,222`, `src/gpu/reranker.py:164,166` | E702, E741 and E731 lint. | OPEN |
| L10 | Clutter | TODO/FIXME/HACK/XXX | **None found** anywhere in scope. | n/a |

---

## By category

1. **Code quality & complexity.** H5, H6, M3, M4, L9. Longest functions:
   - `run_v4.main` ≈250 lines
   - `run_v2.main` 140
   - `predict_test_chunked` 118
   - `run_pipeline.main` 108
   - `features_v2.build_pair_features` 103
   - `ensemble.main` 92
2. **Test coverage & gaps.** H3 and H4. Modules with **no** test: `config`, `blocking`, `ingest`, `run_pipeline`, `ensemble`, `redecide`, `score_test`, `diagnose_country`, `runlog`. Covered: `decide`, `stage2` (claims + gang), `hopeso` passes, `run_v4` helpers, score sidecars, and `features_v2`.
3. **Architecture & brittle logic.** H1, M1, M2, M7, M10, L1, L2, L3.
4. **Dependencies & deprecations.** L4. Pins are right for today (`pandas<3`, `polars>=1`). Heavy GPU dependencies are lazy-imported, so a CPU box never loads them unless `--rerank` is passed.
5. **Dead code & clutter.** L5-L10.

---

## Top 5 immediate priorities (before the SageMaker run)

1. **Copy the full train and test candidate caches to the box before the clock starts.** `hopeso.py build` needs `cands_train_..._nall.parquet`, which Priyanshu's `011` run already built. Without it, step 1 is another ~2 h of blocking.
2. **Read the local 90k number first** (`runs/local_v4_hopeso`). Go only if stage 2 beats the same-sample baseline (0.9511).
3. **Run hopeso on a second instance, in parallel with 004**, not after it. The quota allows 2× m7i.48xlarge. This was the council's peer-review catch.
4. **Size for the box:** `AMLC_WORKERS=176 AMLC_BLOCK_THREADS=176`, `--chunk 4000000`.
5. **Gate the upload:** `validate_submission.py` PASS, then compare links per entity and singleton rate per country against 003 before spending a slot.

## Step-by-step remediation plan (after the deadline)

1. Extract `candidates_path(split, n, tag=None)` into `config.py`. Replace the 4 hand-built names (M2) and add a normalizer version to the key.
2. Extract `finish_test(tsel, t_pairs, test_ids, country)`, the shared tail of 5 scripts (M3).
3. Split `run_v4.main` into `stage_frame / stage1 / claims / stage2 / decide / test` functions that pass a small state object (H5).
4. Add a 2k-entity end-to-end smoke test: blocking, then `predict_test_chunked` with a stub reranker, then `write_outputs` and the validator (H3, H4).
5. Delete `laya_rr.py` (rejected) or merge it through a shared `train()` (M4).
6. Parallelize the chunk loops with a process pool and `num_threads` capped per worker (H6).
7. Make the repo installable (`pyproject.toml`), drop `sys.path` hacks, add pytest (L1, L4).
8. Clean the ruff F401/E7xx findings (L7, L9).

---

## Weakest-hypothesis note on the tuned constants

`VIBE = (top 3, cap 50, dost 10)` came from a single 30k measurement. The evidence *forces* only this: exact-key siblings and rare-name lookups recover misses cheaply. It does **not** force top 3 over top 5, or 50 over 20. Those values were picked for cost, not proven optimal.

The weaker, better-supported statement is "any pool cap between 20 and 50 recovers 16-25% of misses at 1-10 candidates per entity". So the constant lives in one place, and it is encoded in the cache name (H1) so a change can never be silent.

The same goes for `TEST_CONTENTION = 5.549`. It is a measured fact about one test frame, and is used only in log lines, never in computation.
