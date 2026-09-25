# Amazon ML Challenge 2026 — Business Entity Resolution

Match business records across three noisy sources. Source 1 is the deduplicated
reference; for each Source 1 entity, find all matching records in Source 2 and
Source 3. An entity may match zero, one, or many.

**Window:** 25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST · **5 submissions/day**
**Metric:** macro-averaged F_0.5 (precision weighted 2×)

| Doc | What's in it |
|---|---|
| [`docs/DATA_BRIEF.md`](docs/DATA_BRIEF.md) | Measured stats, the train/test country shift, what the scale forces |
| [`docs/SUBMISSION.md`](docs/SUBMISSION.md) | Output format, rejection rules, metric, final zip, checklist |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Append-only run log — record every leaderboard submission here |
| [`docs/TEAM_BUCKET.md`](docs/TEAM_BUCKET.md) | Shared S3 bucket: auth, helper script, sharing the candidate cache |
| `Documentation_template.md` | Methodology write-up — graded, fill in as you go |

---

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows
pip install -r requirements.txt
```

Put the dataset outside the repo (it is ~2.4 GB and gitignored):

```
<data dir>/dataset/train/train_source{1,2,3}.tsv
<data dir>/dataset/train/train_ground_truth.tsv
<data dir>/dataset/test/test_source{1,2,3}.tsv
```

`src/config.py` looks for `D:\amlc_data`, then `E:\amlc_data`, then falls back
to `~/amlc_data`. Override with `AMLC_DATA_DIR`:

```bash
AMLC_DATA_DIR=/path/to/data python src/run_pipeline.py --blocking-only
```

## Run

```bash
# 1. recall ceiling at K = 5/10/20/30/50 — no training, run this first
python src/run_pipeline.py --blocking-only

# 2. full pipeline -> output/matching_results.tsv + output/candidate_pairs.tsv
python src/run_pipeline.py

# 3. validate before uploading (a rejected file still costs a submission)
python validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir <data dir>/dataset/test
```

Blocking recall is the hard cap on the final score — no matcher can exceed what
candidate generation hands it. Optimise that number before touching the model.
`DATA_BRIEF.md` §7 has the table for reading the ceiling and what to change.

## Pipeline

```
src/config.py        paths, TOP_K, BLOCK_MAX_DF, TRAIN_SAMPLE, threshold
src/normalize.py     suffix stripping, abbreviations, accent folding, acronyms,
                     numeric tokens; split into cheap (blocking) / heavy (features)
src/blocking.py      word TF-IDF + document-frequency pruning + chunked sparse
                     matmul — the only shape that survives 1.7M × 10M
src/features.py      ~30 pair features: rapidfuzz ratios, Jaccard, containment,
                     numeric agreement, per-entity rank and gap-to-best
src/metrics.py       macro F_0.5, per-slice breakdown, blocking recall ceiling
src/data.py          TSV I/O with sep="\t" enforced, format-validating writer
src/run_pipeline.py  GroupKFold by entity, threshold sweep, writes both TSVs
```

**Shape of the approach:** blocking narrows ~10M candidates per entity to a few
dozen, a LightGBM pairwise classifier scores each surviving pair, and a single
threshold tuned on out-of-fold predictions turns scores into match sets.
Candidates are generated within country partitions, which handles France (15%
of test, absent from train) without any country ever being enumerated.

## Things that will bite

- **Read TSVs with `sep="\t"`.** Without it pandas silently returns one column
  containing the whole line. Names and addresses contain commas.
- **France is 15% of test and 0% of train.** Never hard-code `{US, India}`;
  French vocabulary has to come from the tables in `normalize.py`.
- **Singletons are only 5.6%.** Predicting empty when unsure feels safe and is
  not — 94.4% of entities have matches, averaging 3.46.
- **Don't put character n-grams in blocking.** They explode at 10M records.
  They belong in features, over the few dozen candidates that survive.
- **No external lookups.** No geocoding, no registries, no APIs. Reviewed, and
  violations mean disqualification.

## Constraints

Final model MIT/Apache-2.0, ≤8B parameters. Current stack: LightGBM (MIT),
rapidfuzz (MIT), scikit-learn (BSD-3), pandas (BSD-3) — all clear.
