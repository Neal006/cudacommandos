# Manifest — submission 001

Provenance for the file in this folder, so a leaderboard number can be traced
to the exact code and model that produced it.

## Identity

```
submission        001
built             2026-09-26 08:22 IST
code commit       6897207c77596a0ceb92be4e7a0b49a65b411fcd
branch            nealstuff
model             runs/007_v2_full/model.pkl   (35.7 MB)
training run      v2_20260926_0140
leaderboard score PENDING -- fill in after upload
```

## Command

```bash
AMLC_OUTPUT_DIR=/d/amlc_data/output AMLC_WORKERS=3 \
  python src/score_test.py --run runs/007_v2_full --chunk 2000000
```

## Pipeline configuration

| Knob | Value |
|---|---|
| `TOP_K` | 30 |
| `BLOCK_MAX_DF` | 0.01 |
| `BLOCK_MIN_DF` | 3 |
| `BLOCK_WITHIN_COUNTRY` | True |
| `TRAIN_SAMPLE` | 150,000 |
| `N_FOLDS` | 5 (GroupKFold by Source-1 entity) |
| `TEST_CHUNK_PAIRS` | 2,000,000 |
| stage-1 features | 46 |
| stage-2 features | 58 |

Decision layer, selected by sweeping all 288 combinations on out-of-fold
predictions:

```
assign  soft
select  expected_f
miss    0.1
thr     n/a (expected-F is per entity, not a global cut)
```

## Offline scores (run 007, out-of-fold)

| Layer | macro F0.5 |
|---|---|
| stage 1 | 0.9500 |
| stage 2 | 0.9526 |
| + decision layer | **0.9532** |

Per country: US 0.9620, India 0.9399. France has no training labels and
therefore no offline number.

Blocking recall 0.9498 → an F0.5 ceiling of 0.9903, so 3.7 points of the
remaining gap are in the matcher and 1.0 point is in blocking.

## Output

```
matching_results.tsv   1,732,544 rows   93,955,304 bytes   md5 cc101bdba923078a55bc2863e5357d1f
candidate_pairs.tsv    1,732,544 rows  693,990,794 bytes
```

| Statistic | Value | OOF reference |
|---|---|---|
| total links | 5,414,237 | — |
| links per entity | 3.125 | 3.138 |
| singleton rate | 0.0625 (108,221) | 0.0628 |
| max links on one entity | 11 | — |
| entities with no candidates | 50 | — |

## Validation

`validate_submission.py`, the organisers' own checker, exit code 0:

```
required S1 entities: 1732544
matching_results.tsv: 1732544 rows (108221 empty, 1624323 non-empty)
candidate_pairs.tsv:  1732544 rows (50 empty, 1732494 non-empty)
PASS - no blocking issues found. Safe to submit.
```

The ID-existence check (`--check-ids`) was **not** run. It is off by default,
costs significant memory, and a nonexistent id only lowers the score rather
than causing rejection — and every id here comes from the test sources by
construction, since candidates are generated from them.

`data.write_outputs` separately enforces the row count, duplicate and
subset rules as it writes.

## Timings

| Stage | Time |
|---|---|
| load test sources + candidate cache | 3 min |
| stage-1 pass, 26 chunks | 74 min |
| global claim features (52M rows) | 2 min |
| stage-2 pass, 26 chunks | 43 min |
| write both TSVs | 2 min |
| **total** | **~2 h** |

Test blocking was reused from run 007's cache, saving a further 52 minutes.

## Known issues

- The scoring script raised `TypeError` **after** both files were written, in
  a logging line: `rate_stats` returns the per-country dicts before the two
  scalars, and the unpacking took the first two. The outputs are unaffected --
  `write_outputs` had already returned. Fixed in this commit, along with the
  same mis-unpack in `run_v2.py`, where it silently wrote dicts into
  `summary.json` instead of floats.
- The GPU reranker is **not** in this model. It was measured at +0.009 on a
  30k sample but its record-overlap leak audit is still open, so it is
  excluded until that is settled.

## Not in git

| File | Size | Where |
|---|---|---|
| `candidate_pairs.tsv` | 694 MB | `D:\amlc_data\output\` — over GitHub's 100 MB limit |
| `model.pkl` | 35.7 MB | `runs/007_v2_full/` — gitignored (`*.pkl`) |
| test candidate cache | 758 MB | `D:\amlc_data\interim\` |

Share via `./aws/s3.sh` (bucket `amazon-cuda-commandos-2026`, ap-south-1).
