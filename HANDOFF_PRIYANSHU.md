# Handoff for Priyanshu (2026-09-27)

Branch: https://github.com/Neal006/amazonml/tree/nealultraprohopeso
`git fetch && git checkout nealultraprohopeso && git pull` before anything. Old checkouts crash at the holdout (fixed in 042c8f9).

## What is on this branch, in one screen

| Piece | Where | Measured |
|---|---|---|
| hopeso recall passes (siblings + rare-name lookup) | `src/hopeso.py` | recall 0.9496 -> 0.9623 (30k) |
| gang features in stage 2 | `src/stage2.py` | part of the next line |
| v4 = full-frame contention + test-like holdout | `src/run_v4.py` | local 90k: stage 1 0.9490 -> **0.9516**, stage 2 0.9511 -> **0.9549** |
| exact expected-F decoder, score-file fingerprints, cands-tag guard | `src/decide.py`, `src/data.py`, `src/score_test.py` | tests |
| bge-reranker-v2-m3 (Apache-2.0, 0.57B) as band reranker, CPU bf16 | `src/gpu/reranker.py --base` | bench on the box decides |
| every limit is a setting | `AMLC_VIBE`, `AMLC_TOP_K`, `AMLC_MAX_DF`, `AMLC_TRAIN_SAMPLE` | `context_hopeso.md` |
| Rust mlguard removed, CI is Python only | `.github/workflows/ci.yml` | 6 data-free tests pass |

## What to run (details and timings in PIPELINE_98.md)

Three boxes in parallel:

1. **m7i #1: submission A (safe).** The `context_hopeso.md` runbook, default settings. Copy run 011's full train and test candidate caches into `interim/` first, or blocking reruns.
2. **c7i #1: the reranker.** `AMLC_CPU_BF16=1 AMLC_TORCH_THREADS=190 python src/gpu/reranker.py bench --model BAAI/bge-reranker-v2-m3`. At 500 pairs/s or more, train with `--entities 40000`. From 150 to 500, use `--entities 20000`. Under 150, skip B.
3. **m7i #2: submission B.** hopeso build, then v4 with `--rerank models/rr_bge --band 0.05 0.95`, then score_test with the same band and `--cands-tag hopeso`.

Upload rule: validator PASS, then the higher `holdout_score` in `runs/<id>/summary.json` wins. If B is not written by 19:30, upload A.

## Known gaps (do not be surprised)

- The reranker's leak guard checks S1 entities, not S2/S3 records (audit item M7). The holdout is still clean.
- `AMLC_VIBE` (bigger sibling pools) is not measured yet. Run `python src/hopeso.py measure --n 30000` before changing it. Leave the default for today's runs.
- The local 90k holdout number is still running on Neal's laptop. Stage 2 already beat the baseline.
- 0.98 is not promised. A perfect matcher on this shortlist tops out near 0.982 on the leaderboard. Realistic: A about 0.957 to 0.962, B higher if the reranker holds up on the holdout.

## Docs to read, in order

1. `PIPELINE_98.md`: the plan, box by box
2. `context_hopeso.md`: hopeso runbook and knob table
3. `TECH_DEBT_AUDIT.md`: what is fixed and what is open
4. `context_help.md`: plain-language brief of the whole project
