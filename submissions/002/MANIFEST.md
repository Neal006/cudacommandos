# Manifest — submission 002

## Identity

```
submission        002
built             2026-09-26 23:20 IST
code commit       f88b95e
branch            priyanshu/push985
model             runs/ab_e5s/model.pkl  (40.9 MB, 30k sample)
band reranker     models/rr_e5s  (multilingual-e5-small, band 0.2-0.8)
leaderboard score PENDING -- fill in after upload
```

## Command

```bash
AMLC_OUTPUT_DIR=D:/amlc_data/output AMLC_WORKERS=2 \
  python src/score_test.py --run runs/ab_e5s --chunk 1200000 --rerank models/rr_e5s
```

## Configuration

| Knob | Value |
|---|---|
| `TOP_K` | 30 |
| `BLOCK_MAX_DF` / `BLOCK_MIN_DF` | 0.01 / 3 |
| `TRAIN_SAMPLE` | 30,000 |
| `N_FOLDS` | 5 (GroupKFold by S1 entity) |
| chunk | 1,200,000 pairs (44 chunks) |
| reranker band | 0.2 - 0.8, ~1.5% of pairs |
| decision | assign=hard, select=expected_f, miss=0.0 |

## Reranker

Trained on 514,335 pairs excluding run 007's folds; same pairs and valid split
as the laya challenger, so the A/B differs only in the model.

```
valid_logloss 0.03339   valid_auc 0.99907   train 23 min
```

## Offline scores (runs/ab_e5s)

| Layer | macro F0.5 |
|---|---|
| stage 1 | 0.9490 |
| stage 2 + reranker | 0.9607 |
| + decision layer | 0.9607 |

Per country: US 0.9697, India 0.9469. The 30k no-reranker baseline is 0.9512,
so the reranker adds +0.0095 -- under an open leak suspicion (see README).

## Output

```
matching_results.tsv   1,732,544 rows    95,221,072 bytes  md5 a874bf6cbb6e6369f26be9a508a6fe0c
candidate_pairs.tsv    1,732,544 rows   693,990,794 bytes
```

| Statistic | Value |
|---|---|
| total links | 5,512,946 |
| links per entity | 3.182 |
| singleton rate | 0.0623 (107,972) |
| entities with no candidates | 50 |
| matches outside candidates | 0 |

Per country, measured for the first time (run 007 crashed before printing them):

| | links/entity | singleton% | blocking top_sim p10 | model top_p p10 | orphan% |
|---|---|---|---|---|---|
| India | 3.085 | 6.81% | 0.8580 | 0.9808 | 0.002% |
| US | 3.267 | 5.87% | 0.8027 | 0.9939 | 0.000% |
| France | 3.266 | 5.35% | 0.9128 | 0.9985 | 0.013% |

## Validation

`validate_submission.py`, exit 0:

```
matching_results.tsv: 1732544 rows (107972 empty, 1624572 non-empty)
candidate_pairs.tsv:  1732544 rows (50 empty, 1732494 non-empty)
PASS - no blocking issues found. Safe to submit.
```

## Timings

| Stage | Time |
|---|---|
| load + candidate cache | 3.6 min |
| stage-1 pass | **skipped -- loaded from cache** |
| global claim features | 1.5 min |
| stage-2 pass + reranker, 44 chunks | 56 min |
| write both TSVs | 4 min |
| **total** | **68 min** |

Two earlier attempts at this run were killed by the memory reaper. The first
was contention (an analysis run alongside it); the second was the reranker
reloading its model on every chunk -- 42 loads, each leaving GPU allocations --
which died at stage-2 chunk 42 of 44. Both are fixed: the model is cached for
inference, and stage-1 scores are cached so a restart resumes at pass 2.

## Not in git

| File | Size | Where |
|---|---|---|
| `candidate_pairs.tsv` | 694 MB | `D:\amlc_data\output\` |
| `model.pkl` | 40.9 MB | `runs/ab_e5s/` (gitignored) |
| `models/rr_e5s` | ~500 MB | local (gitignored) |
| cached scores | 208 MB each | `D:\amlc_data\interim\testp*_50cf8414638d.npy` |
