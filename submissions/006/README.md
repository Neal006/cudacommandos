# Submission 006 — hopeso + e5 band reranker

**Status: READY TO UPLOAD. Best model of the competition. Upload this one.**

---

## Krisha / Priyanshu — what to do

1. Download `matching_results.tsv.gz` (39 MB) from this folder.
2. Unzip → `matching_results.tsv`, ~96 MB, **1,732,545 lines**.
3. Upload the **`.tsv`** to Unstop. Not the `.gz`.
4. **Post the score back here immediately.** It is the only measurement that
   tells us how much to trust the holdout for the remaining uploads.

---

## What this is

| | |
|---|---|
| run | `runs/014_hopeso_e5` (cloud, `ml.c7i.48xlarge`, 192 vCPU) |
| candidates | **hopeso** — base TF-IDF top-30 ∪ sibling expansion |
| blocking recall | **0.9630** (base was 0.9498) |
| model | 150k sample, two-stage LightGBM + e5-small reranker on the 0.2–0.8 band |
| stage 1 → stage 2 | 0.9533 → **0.9685** |
| **holdout (50,000 unseen entities)** | **0.9704** |
| holdout per country | India **0.9619** · US **0.9760** |
| decision | `assign=none`, `select=threshold`, `thr=0.70` |
| singletons | 0.0614 predicted (true train rate 0.0563) |

### Against everything before it

| | recall | OOF | holdout | LB |
|---|---|---|---|---|
| 003 (uploaded) | 0.9498 | 0.9626 | — | 0.953 |
| 004 | 0.9498 | 0.9644 | — | not uploaded |
| 005 (hopeso, no reranker) | 0.9630 | 0.9602 | 0.9614 | not uploaded |
| **006 (hopeso + e5)** | **0.9630** | **0.9685** | **0.9704** | — |

Two independent gains stack here:

- **hopeso** (sibling expansion) lifted blocking recall 0.9498 → 0.9630 for
  8.15 extra candidates per entity, worth **+0.0032 on stage 1**.
- **the e5 reranker** on the uncertain band is worth **+0.0090 on the holdout**
  (005 → 006 on identical candidates and sample).

India, our weak country all year, moved **0.9491 → 0.9619**.

## What the holdout is, and what it is not

The holdout is 50,000 entities held out of training entirely, outside the
reranker's training set too, and scored by the **mean of the five fold models**
— which is how test is scored. That makes it a far better leaderboard
predictor than OOF, which scores each entity with the single fold model that
did not see it.

It is still optimistic, for two reasons we can name:

1. **No France.** Train is US + India; France appears only in test, about
   1.43M of 9.97M records (14.3%). The holdout cannot measure it. (France has
   looked like our strongest country on every label-free measure, so this may
   cost little.)
2. **Easier decoys.** Test's source pool carries 1.89x the decoys per entity
   (39.9% against 26.0%). Blocking equalises the *count* — 38.34 candidates
   per entity against train's 38.15 — but the decoys that survive into the top
   38 are nearer misses in test.

**Honest leaderboard expectation: 0.961 – 0.965**, with 0.967+ only if both
structural fixes fully hold and France behaves. Our OOF→LB gap has been stable
at −0.0102 / −0.0097 / −0.0096, and this holdout reads +0.0019 above its own
OOF, so do not expect 0.970.

## Validation

```
validate_submission.py --check-ids
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (106377 empty, 1626167 non-empty)
  candidate_pairs.tsv:  1732544 rows (38 empty, 1732506 non-empty)
PASS - no blocking issues found.
```

The 38 empty candidate rows are expected: no token of those S1 entities
survives df pruning, so they get no candidates and are written empty.

## Decision layer

`src/decide_source.py` cross-fitted every recipe across two entity halves on
this run's sibling (005) holdout. Per-source thresholds are real — S2 wants
0.68 and S3 0.71 — but worth **+0.0004**, under the +0.0010 adoption gate, so
the global threshold stands. `hard` assignment beat `none` by the same
negligible margin. Our calibration is already good enough that per-class
thresholds have little left to repair.
