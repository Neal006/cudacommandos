# AGENTS.md — Project Memory (auto-maintained)
Last updated: 2026-09-25 | Sessions logged: 2

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
- v2 pipeline (OOF only): `python src/run_v2.py --sample 30000 --train-only --ablate`; full: `PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh <run_id> --sample 150000`
- data-free tests: `python tests/test_decide.py && python tests/test_features_v2.py` (+ `tests/test_embed_knn.py` needs torch)
- GPU reranker: `python src/gpu/reranker.py train --exclude runs/<id>/folds.tsv --entities 40000 --out models/rr_e5s`

## Current State & Focus
- Works: word TF-IDF blocking (per country, K=30, recall ceiling 0.95), ~30 rapidfuzz features, LightGBM GroupKFold, global threshold, output writer (branch pipeline/entity-resolution).
- Master-Plan branch adds the plan (docs/master-plan/), the mlguard checker + CI, runlog.py, EDA + skeleton prototype.
- Next (SYSTEM_ARCHITECTURE §4): baseline OOF + LB #1, decision layer (assignment + expected-F), transliteration + passes B/C/D, vectorized features.
- nealstuff branch: v2 pipeline (src/run_v2.py) = lean ingest + threaded top-k blocking (same candidates) + vectorized features (same values, +11 new) + stage 2 + decision layer + optional GPU reranker, all logged to mlguard. run_pipeline.py untouched.
- Not yet wired: the runlog hooks in run_pipeline.py (LLD §9) — run_v2.py has them.
- 30k OOF (EXPERIMENTS 003–005): old 0.9266 → v2 stage 1 0.9490 → stage 2 0.9508 → decision 0.9512; +reranker stage 2 0.9600 UNVERIFIED (record-overlap leak audit pending, PR for Priyanshu).
- 150k OOF (EXPERIMENTS 007, full sample, CPU only): stage 1 0.9500 -> stage 2 0.9526 -> decision 0.9532; India 0.9399 / US 0.9620; blocking recall 0.9498. Ahead of 30k at every stage.
- Test phase run for the first time: blocking is ~52 min, not the 255 min EXPLAINER budgets (4.4x).
  Throughput is a property of the INDEX, not the machine -- test rates France 1066 q/s (1.43M recs),
  US 800 (3.82M), India 397 (4.72M); US searches fewer records than India and is still 2x faster
  because Indian tokens are denser. Never quote q/s without the index it was measured against.
  EXPLAINER §11 and DATA_BRIEF corrected. Candidate cache `cands_test_k30_df0.01_mdf3_ctry1_nall.parquet` (52.0M pairs, 758 MB) and `stats_test_v1.pkl` are built -- share via `./aws/s3.sh share-cache`.
- SUBMISSION 001 uploaded (run 007 model, score_test.py, 117 min scoring): public LB **0.943, rank 933**
  vs OOF 0.9532. Mix reweighting explains only 0.0006; if US/India transfer 1:1, France (15% of test,
  0 labels) scores ~0.898. Deadline 27 Sep 2026 23:59 IST, 5 uploads/day. Team target 0.985 (vs
  blocking ceiling ~0.989 -- needs recall AND matcher work). Plan: docs/master-plan/PLAN_985.md.
- Shipped decision is assign=soft: one S2/S3 id CAN appear under several S1 in the output (GT is a
  partition). Calibrator + miss=0.1 tuned on US/India only, applied blind to France.
- Teammate branch sentence-transformer-embeddings-feature: English MiniLM cosines on the OLD
  pipeline, 2k sample, no baseline, has an indentation bug -- do not merge as-is.
- Test cache is 50 S1 entities short of the 1,732,544 required (1,732,494 got candidates). Expected,
  not a bug -- no token survives df pruning for them; write_outputs iterates the full test id list so
  they emit empty match sets. Do not treat it as a blocker at upload time.
- Decision layer measured on 288 combos at 150k: expected_f beats a global threshold by +0.0006,
  soft assign beats none by +0.00007. Keep expected_f, stop tuning assign. The 3.7 points to the
  0.9903 blocking ceiling are in the matcher, concentrated in India.
- Windows: pools are capped by `config.WORKERS` (4 on spawn, AMLC_WORKERS to override) or the run OOMs at pool startup; `train_guarded.sh` must use the venv (system python is pandas 3.x); `*.sh` pinned LF.
- RTX 3050 was absent from the PCI bus (Code 45, torch.cuda False) this boot, so the reranker is unverified and unrunnable in practice -- CPU fallback turns its 11 min into ~4-7 h. Needs a reboot.

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
- `src/runlog.py` — RunLog: metrics.jsonl, summary.json, folds.tsv, lgb_callback (stops only the fold named in MLGUARD_STOP)
- `src/ingest.py` — blocking_frame/load_split_lean: polars + parallel blobs, Arrow strings (S2 peak 3.1 GB vs 6-8)
- `src/features_v2.py` — record_table (per record, parallel), build_pair_features (cpdist + sparse overlaps; OLD_COLUMNS identical), split_stats (label-free S1 stats)
- `src/stage2.py` — build(pairs, p1, R): entity shape, competition (claim_rank/gap), peers (peer1/2_sim)
- `src/decide.py` — crossfit_calibrate, assign(none|hard|soft), select_threshold/expected_f, macro_f05 (vectorized), tune
- `src/run_v2.py` — orchestrator; --ablate, --no-stage2, --rerank DIR --band lo hi, --train-only
- `src/gpu/reranker.py` — e5-small cross-encoder (frozen word embeddings, bf16), train/score/bench, entities.txt leak guard
- `src/gpu/embed_knn.py` — measurement-only dense kNN: streamed shards, mmap .npy, kill < +0.002 F0.5 ceiling
- `tests/` — test_decide, test_features_v2, test_embed_knn (assert scripts)
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
- sparse_dot_topn converts a CSC `B` to CSR on EVERY call — pass the index as CSR once (blocking.py does).
- Candidate cache key has no normalizer version: a blob-changing normalize.py edit needs a manual cache purge.
- mlguard loss_ratio only fires when valid loss also stalls (>2% above best); plain ratio tripped healthy boosting.
- A guard stop must roll LightGBM back to the best valid iter (runlog tracks it); summary.json must be strict JSON (NaN → null).
- Reranker leak: disjoint S1 is not enough — S2/S3 records can repeat across rr training and the GBDT sample.
- Never start another python job next to run_v2 on a 16 GB box (run 005 OOM).
- Bash heredocs through the agent tool collapse `\` escapes — edit Python/Rust escapes with the Edit tool.

## Decisions Log
- 2026-09-25 — TOP_K=30 — F0.5 ceiling gain 30→50 is +0.002 for 35M pairs (Priyanshu, run 001)
- 2026-09-25 — Add decision layer (assignment + expected-F) before new models — cheapest, direct metric gain
- 2026-09-25 — Deterministic transliteration + skeleton over model-based — offline, auditable, measured 94% token sharing
- 2026-09-25 — Rust mlguard as an independent watcher + CI gate — survives Python OOM, fast on 10M ids
- 2026-09-25 — GPU goes to a band reranker, not recall or trees — review: blocking backlog worth +0.005, LightGBM is 8 of 414 min
- 2026-09-25 — run_v2.py beside run_pipeline.py, sharing its cache — no untested edits to the 6 h pipeline

## Changelog
2026-09-26 | LB 0.943 post-mortem + plan to 0.985 | AGENTS.md, docs/master-plan/PLAN_985.md | France + partition + recall are the levers; no code changed
2026-09-26 | runs 003–005 + guard fixes | src/runlog.py, src/run_v2.py, tools/mlguard/test_runlog.py, docs/EXPERIMENTS.md, GPU_PLAN.md | stop rolls back to best iter; NaN-safe summary; reranker gain held until leak audit
2026-09-25 | v2 pipeline + GPU reranker + review fixes | src/{ingest,features_v2,stage2,decide,run_v2}.py, src/gpu/*, blocking.py, normalize.py, runlog.py, tools/mlguard/src/*, tests/*, docs/master-plan/GPU_PLAN.md | precision over recall; identical-output speedups; GPU job 2/4 dropped per review
2026-09-25 | Master plan + mlguard + EDA | docs/master-plan/*, tools/mlguard/*, tools/eda/*, src/runlog.py, .github/workflows/mlguard.yml, AGENTS.md | plan built on measured data; checker gates runs in bg/CI/test

## Archived Summary
