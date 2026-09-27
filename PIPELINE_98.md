# PIPELINE_98: hopeso + a billion-parameter cross-encoder, on the SageMaker fleet

Branch `nealultraprohopeso`. Written 2026-09-27 for Priyanshu's quota (screenshot, CPU only, no GPU).

## Read this first: what 98 needs

A perfect matcher at shortlist recall R scores `1.25R / (0.25 + R)`. Our offline numbers run about 0.010 above the leaderboard.

| Shortlist recall | Perfect-matcher ceiling (offline) | Same, on the leaderboard |
|---|---:|---:|
| 0.9496 (base) | 0.9897 | about 0.980 |
| 0.9623 (hopeso) | 0.9922 | about 0.982 |

So 0.98 on the leaderboard needs a matcher that is almost perfect on everything the shortlist found. Today the matcher sits about 4 points under the ceiling (local 90k: stage 2 0.9549 vs ceiling 0.9922). This pipeline attacks exactly that gap with the strongest allowed model the fleet can run. **Nobody can promise 98 from it.** What it does promise: every step is measured before the next one spends a box, and the safe submission is never at risk.

## The model

**BAAI/bge-reranker-v2-m3**: Apache-2.0, 0.57B parameters, multilingual (XLM-R large, covers Hindi, Tamil, French). A real cross-encoder with a reranking head already trained, so fine-tuning starts from a model that already knows "same thing / different thing". Licence and size checked on the Hugging Face API on 2026-09-27.

Why not 7B or 8B: the fleet has no GPU. An 8B model on CPU reads maybe a few pairs per second per box (estimate), so even the 1.6M uncertain test pairs would take days. Qwen3-8B and Qwen3-Reranker-8B are also 8.19B, which is over a strict "8B" reading. Llama and Gemma licences are not MIT/Apache; jina-reranker-v2 is non-commercial.

## Where the model is used

Only where the trees are unsure. On the local 90k OOF (stage 2):

| Band of score | Share of pairs | Share of all true matches inside it |
|---|---:|---:|
| 0.2 to 0.8 | 0.94% | 4.95% |
| 0.1 to 0.9 | 1.61% | 7.79% |
| 0.05 to 0.95 | 2.40% | 10.91% |

We use **0.05 to 0.95**: about 1.6M test pairs (66M test pairs x 2.4%). The cross-encoder score goes into stage 2 as the `rr` feature, the same slot e5-small used (e5-small gave about +0.009 at 30k, still pending the record-overlap audit).

## The fleet plan (all boxes start in parallel)

| Box | Job | Output |
|---|---|---|
| **m7i.48xlarge #1** | hopeso v4, no reranker (`context_hopeso.md` runbook) | **Submission A, the safe one** |
| **c7i.48xlarge #1** (Sapphire Rapids, AMX bf16) | bench, then fine-tune bge-reranker-v2-m3 | `models/rr_bge` |
| **m7i.48xlarge #2** | builds the hopeso frames, waits for `models/rr_bge`, then v4 with the reranker | **Submission B** |
| c6i / r5 / c5 | spare. c6i and r5 have no AMX; do not put the reranker there | |

### c7i #1: bench, then train (about 5 min + estimate 45 to 90 min)

```
export AMLC_WORKERS=176 AMLC_BLOCK_THREADS=176 AMLC_TORCH_THREADS=190 AMLC_CPU_BF16=1
python src/gpu/reranker.py bench --model BAAI/bge-reranker-v2-m3
```

The bench prints `pairs/s` (untrained head, speed only). Decide from it:

| Bench | Train with | Band for v4 |
|---|---|---|
| 500 pairs/s or more | `--entities 40000` | `0.05 0.95` |
| 150 to 500 | `--entities 20000` | `0.1 0.9` |
| under 150 | stop, use submission A, not worth the risk | |

```
python src/gpu/reranker.py train --base BAAI/bge-reranker-v2-m3 \
    --exclude runs/<the 150k run>/folds.tsv --entities 40000 --bs 32 --lr 2e-5 \
    --out models/rr_bge --run-dir runs/rr_bge
```

`--exclude` must be the folds.tsv of any 150k run on seed 42 (same sample as v4). The reranker then never sees a GBDT training entity; v4 also keeps its holdout outside the reranker's entities and refuses to start on a leak. Gate: the final `valid AUC` line should beat e5-small's (`models/rr_e5s/meta.json`). If not, stop and use A.

### m7i #2: v4 with the reranker (estimate 2.5 to 3 h)

```
export AMLC_WORKERS=176 AMLC_BLOCK_THREADS=176 AMLC_TORCH_THREADS=190 AMLC_CPU_BF16=1
python src/hopeso.py build --split train      # while c7i trains
python src/hopeso.py build --split test
# copy models/rr_bge from c7i when it is done
python src/run_v4.py --sample 150000 --rounds 4000 --chunk 4000000 --cands-tag hopeso \
    --rerank models/rr_bge --band 0.05 0.95 --train-only
python src/score_test.py --run runs/<id> --rerank models/rr_bge --band 0.05 0.95 \
    --chunk 4000000 --cands-tag hopeso
```

The band in `score_test` must equal the band in `run_v4`. `score_test` refuses a reranker-trained model when `--rerank` is missing.

## Go / no-go before any upload

1. `python validate_submission.py ...` PASS.
2. Compare B with A: `holdout_score` in `runs/<id>/summary.json`, and per-country links/entity. Upload B only if its holdout beats A's.
3. Singleton rate per country within 1 point of the train rate (5.6%).

## Timeline to 20:00 IST

| Time | Event |
|---|---|
| 15:15 | all three boxes start |
| 15:20 | bench result, pick entities and band |
| about 16:30 | `models/rr_bge` ready (estimate) |
| about 17:00 | submission A written |
| about 19:15 | submission B written (estimate) |
| before 19:45 | validate, compare, upload the winner |

If B is not written by 19:30, upload A. No exceptions.

## Honest expectation

- A: about 0.957 to 0.962 on the leaderboard (local +0.0038 from hopeso over the 0.953 base).
- B: A plus the reranker gain. e5-small gave +0.009 offline at 30k on the 0.2 to 0.8 band; a model 5x bigger on a band that holds twice the true matches can plausibly do more, but that is a guess until the holdout says so.
- 98 needs the matcher to fix most of the remaining errors, and 2,986 of the 30k-sample errors (825 confident wrong links, 2,161 confident misses) sit outside any band. No reranker on a band reaches them.
