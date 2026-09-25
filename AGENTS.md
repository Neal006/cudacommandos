# AGENTS.md — Project Memory (auto-maintained)
Last updated: 2026-09-25 | Sessions logged: 1

## Identity
Team cudacommandos' solution to the Amazon ML Challenge 2026 "Business Entity Resolution": match S2/S3 business records to deduplicated S1 entities (US/India train, + France test), scored by macro F0.5.

## Stack & Commands
Python 3.11–3.13 (pandas<3, polars, scikit-learn, LightGBM, rapidfuzz) + Rust (mlguard). Dataset lives outside the repo: `AMLC_DATA_DIR` (expects `dataset/{train,test}`).
- install: `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`
- blocking ceiling only: `python src/run_pipeline.py --blocking-only`
- full run (guarded): `tools/mlguard/train_guarded.sh <run_id>`  (plain: `python src/run_pipeline.py`)
- validate: `python validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <dataset/test>`
- mlguard: `cargo test --release --manifest-path tools/mlguard/Cargo.toml`; `python tools/mlguard/test_runlog.py`
- EDA numbers: `AMLC_DATASET_DIR=<dataset> python tools/eda/design_eda.py`

## Current State & Focus
- Works: word TF-IDF blocking (per country, K=30, recall ceiling 0.95), ~30 rapidfuzz features, LightGBM GroupKFold, global threshold, output writer (branch pipeline/entity-resolution).
- Master-Plan branch adds the plan (docs/master-plan/), the mlguard checker + CI, runlog.py, EDA + skeleton prototype.
- Next (SYSTEM_ARCHITECTURE §4): baseline OOF + LB #1, decision layer (assignment + expected-F), transliteration + passes B/C/D, vectorized features.
- Not yet wired: the runlog hooks in run_pipeline.py (LLD §9).

## Architecture
TSV → L0 ingest (parquet) → L1 contract → L2 normalize (translit, skeleton, legal form, FR/IN/US tables) → L3 multi-pass blocking ∪ (A name+addr, B address, C translit, D reverse) → L4 candidates parquet → L5 stage-1 GBDT → stage-2 GBDT (competition/peer) → L6 calibrate → assign (partition) → expected-F0.5 per entity → L7 write + validate.
mlguard runs beside training (watch), after it (run/split/submission) and in GitHub Actions.
Caches: `<DATA_DIR>/interim/*.parquet` keyed by parameters; run artefacts in `runs/<id>/`.

## File Map
- `src/config.py` — paths (AMLC_DATA_DIR), knobs: TOP_K=30, BLOCK_MAX_DF=0.01, TRAIN_SAMPLE=150k, N_FOLDS=5
- `src/data.py` — read_source, read_ground_truth, load_split, write_outputs
- `src/normalize.py` — norm_name/core_name/norm_addr, LEGAL_SUFFIXES, ADDR_ABBREV (no `r`→rue globally), add_blocking_columns
- `src/blocking.py` — build_vectorizer, generate_candidates (chunked sparse top-k), orphan_check
- `src/features.py` — build_pair_features (python loops; vectorize with cpdist), add_rank_features
- `src/metrics.py` — f_beta_entity, macro_f_beta, scores_breakdown, blocking_recall
- `src/run_pipeline.py` — main(): train sample → blocking → features → LGBM OOF → threshold → test
- `src/runlog.py` — RunLog: metrics.jsonl, summary.json, folds.tsv, lgb_callback (MLGUARD_STOP)
- `tools/mlguard/` — Rust checker: src/checks.rs (run+watch rules), src/files.rs (submission/split), mlguard.toml (thresholds), fixtures/, train_guarded.sh
- `.github/workflows/mlguard.yml` — CI: fair-play import gate, cargo test, fixture self-test, gate runs/*/summary.json
- `tools/eda/design_eda.py` — reproduces ANALYSIS §1; `tools/eda/skeleton_prototype.py` — cross-script name key
- `docs/master-plan/*` — README (problem, concepts, requirements), ANALYSIS (facts, questions), HLD, LLD, SYSTEM_ARCHITECTURE, EXPERIMENT_PLAN, EDGE_CASES, DATA_SECURITY_AND_LEAKAGE, MODEL_SELECTION, MLGUARD
- `docs/DATA_BRIEF.md`, `docs/EXPERIMENTS.md`, `docs/SUBMISSION.md` — Priyanshu's brief, experiment log, upload guide
- `validate_submission.py` — official stdlib validator

## Conventions
- TSV always with explicit tab; ids keep the S2-/S3- prefix; write with `newline="\n"`.
- `country` is an open set: equality or rule-table key only, never one-hot/filter.
- Every training run goes through train_guarded.sh; no upload without mlguard PASS.
- Log every run/upload in docs/EXPERIMENTS.md; tag uploads `lb-YYYYMMDD-n`.
- Tests: Rust `#[cfg(test)]` in each module; Python smoke scripts with asserts.

## Dependencies & Gotchas
- GT is a partition: no S2/S3 id links to 2 S1 (use it at decision time).
- 0 cross-country positives in train; 26% of S2/S3 are distractors; up to 5–6 dup copies per source.
- ~23% of India S2 names are native Indic script (9 scripts); anyascii + skeleton recovers 94% token sharing.
- India has no PIN codes; US ZIP only ~10%; never block on postal code.
- pandas 3.x breaks code: requirements pin <3.0. The global interpreter has polars 1.37 (is_in needs `.implode()`).
- validate_submission.py passes an all-empty submission; mlguard submission --summary does not.
- Laptop: ~9.7 GB free RAM of 25 GB; test blocking ~4–5 h locally.

## Decisions Log
- 2026-09-25 — TOP_K=30 — F0.5 ceiling gain 30→50 is +0.002 for 35M pairs (Priyanshu, run 001)
- 2026-09-25 — Add decision layer (assignment + expected-F) before new models — cheapest, direct metric gain
- 2026-09-25 — Deterministic transliteration + skeleton over model-based — offline, auditable, measured 94% token sharing
- 2026-09-25 — Rust mlguard as an independent watcher + CI gate — survives Python OOM, fast on 10M ids

## Changelog
2026-09-25 | Master plan + mlguard + EDA | docs/master-plan/*, tools/mlguard/*, tools/eda/*, src/runlog.py, .github/workflows/mlguard.yml, AGENTS.md | plan built on measured data; checker gates runs in bg/CI/test

## Archived Summary
