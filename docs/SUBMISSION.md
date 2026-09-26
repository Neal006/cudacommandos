# Submission guide

Everything that decides whether a submission is accepted, scored, and counted.
Read this before your first upload — a rejected file still burns a slot.

---

## 1. The budget

**5 submissions per day, 3 days.** The submit button disables after the fifth.
That is ~15 uploads total for the whole challenge, so every one should be
deliberate. Validate locally first; it costs nothing.

**Window:** 25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST.

## 2. What you upload

Two files come out of the pipeline. Only the first is scored on the
leaderboard; both go in the final zip.

### `matching_results.tsv` — scored

| Column | Content |
|---|---|
| `source1_entity_id` | The `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3 |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Tab between the two columns. Commas inside the ID list, no spaces, no quoting.
`S1-00003` is a singleton — the cell is empty, the row still exists.

### `candidate_pairs.tsv` — audited, not scored

Same shape, column named `candidate_entity_ids`. This must be **the exact set
your matching model ran inference over** — the last stage before scoring, not
an earlier blocking pass you filtered further. Organisers use it to measure
blocking quality (recall ceiling, reduction ratio) and to verify the pipeline
is what the methodology claims.

Every ID in `matching_results.tsv` must also appear in `candidate_pairs.tsv`.
A match that was never a candidate means a pipeline bug, and the validator
flags it.

## 3. Rules that cause outright rejection

1. **Every** Source 1 test entity gets exactly one row — all 1,732,544 of them,
   France included. Missing entities → rejected.
2. Duplicate `source1_entity_id` rows → rejected.
3. Duplicate IDs inside a single list → rejected.
4. Only `S2-` and `S3-` IDs in the lists. Self-matches to `S1-`, or IDs not
   present in the test set → rejected.
5. Tab-separated, with exactly the column names above.

A correctly formatted upload shows status `SCORED` with your F_0.5.

## 4. Validate before every upload

The organisers ship a stdlib-only checker. It reads your outputs and the test
sources; it does not compute your score.

```bash
python validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir <path to dataset/test>
```

`PASS` (exit 0) means the files are safe to submit. Otherwise it prints a
numbered list of problems (exit 1).

`src/data.py` also enforces the row-count, duplicate and subset rules as it
writes, so most failures are caught before the file exists — but run the
official checker anyway. It is the one that matches the grader.

## 5. The metric

**F_0.5, macro-averaged per Source 1 entity.**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed for each entity, then averaged across all of them. Precision is
weighted 2× recall — merging two distinct businesses is treated as worse than
missing a link.

Worked example:

```
predicted  [S2-00047, S2-00193, S3-00812]
truth      [S2-00047, S3-00812]
precision  2/3 = 0.667
recall     2/2 = 1.000
F_0.5      (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
```

Singleton handling:

| Truth | Prediction | Score |
|---|---|---|
| empty | empty | **1.0** |
| empty | anything | **0.0** |
| non-empty | empty | **0.0** |

Singletons are only 5.6% of entities (see `DATA_BRIEF.md` §3), so the
"predict nothing when unsure" instinct is a trap — 94.4% of entities have
matches averaging 3.46 each. Tune the threshold on out-of-fold predictions
rather than reasoning about the metric in the abstract; `run_pipeline.py`
sweeps it automatically.

## 6. Leaderboards

- **Public** — a subset of the test set, live during the challenge.
- **Private** — the remaining portion, revealed after it closes.
- **Final rankings come from the private leaderboard.**

You submit predictions for the full test set either way; the split is applied
at scoring time. When cross-validation and the public leaderboard disagree,
believe CV — the public board is a sample and the private one decides.

Keep the version history of every submission. Shortlisting is based on
submitted solutions and teams may be asked for source code later.

## 7. Final submission package

Separate from leaderboard uploads. **Every team must submit this**, and the top
teams' packages are reviewed in detail before final rankings are confirmed.

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          # exact end-to-end reproduction steps
│       └── requirements.txt   # pinned versions
└── Documentation_template.md  # filled in
```

`code/` must be self-contained: someone with only that folder and the dataset
should be able to regenerate both TSVs.

The methodology document must cover: methodology, candidate generation /
blocking strategy, model architecture and feature engineering, and anything
else relevant. No page limit — depth beats brevity. **Fill it in as you go**;
it is graded, and leaving it to the last hour is the classic way teams lose
placement they already earned.

## 8. Hard constraints

- **No external data lookup.** No entity-resolution APIs, no business
  registries, no geocoding services, no internet augmentation of any kind.
  Pipelines are reviewed; violations mean immediate disqualification.
- **Final model must be MIT or Apache-2.0 licensed and ≤8B parameters.**
  The current stack is clear on both counts: LightGBM (MIT), rapidfuzz (MIT),
  scikit-learn (BSD-3), pandas (BSD-3).
- One account per participant. No multi-ID attempts.
- Desktop/laptop only, no simultaneous logins — the system may terminate the
  attempt if it detects them.

## 9. Pre-upload checklist

- [ ] Row count equals 1,732,544 (every test Source 1 entity)
- [ ] `validate_submission.py` prints `PASS`
- [ ] Mean matches per entity is in a sane range (truth averages 3.46)
- [ ] Predicted singleton rate is near 5–6%, not 40%
- [ ] `entities_with_matches_outside_candidates` is 0
- [ ] OOF F_0.5 recorded in the experiment log next to what changed
- [ ] Submissions remaining today is > 0
