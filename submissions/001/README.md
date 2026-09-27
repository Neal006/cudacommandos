# Submission 001 — first leaderboard upload

**Status: UPLOADED. Scored 94.3, rank 934.**

Our first submission. Nothing has been on the leaderboard yet, so this is the
run that tells us whether our offline score means anything.

---

## Krisha — what to do

1. Download **`matching_results.tsv.gz`** from this folder (38 MB).
2. **Unzip it.** You need the `.tsv`, not the `.gz`.
   - Windows: right-click → 7-Zip → *Extract Here* (or WinRAR → *Extract Here*)
   - Git Bash / Mac: `gzip -d matching_results.tsv.gz`
   - You should get `matching_results.tsv`, **94 MB**, 1,732,545 lines
     (1 header + 1,732,544 rows).
3. Upload **`matching_results.tsv`** to the Unstop submission portal.
4. **Post the score in the team chat.** Then it gets written into
   `docs/EXPERIMENTS.md` next to what produced it.

That is the whole job. One file.

### Do not upload

- `candidate_pairs.tsv` — **not scored**, not part of the leaderboard. It is
  694 MB and only goes in the final package zip. It is not in git; it is on
  Priyanshu's `D:` drive.
- The `.gz` itself. Unzip first.

### If the portal rejects it

**Stop and post the exact error.** Do not re-upload a guess — that spends a
second slot to learn nothing. This file already passed the organisers' own
validator locally (`PASS`, exit 0), so a rejection means something none of us
has seen, and we should look at it together.

---

## What this is

| | |
|---|---|
| Model | `runs/007_v2_full/model.pkl` — two-stage LightGBM |
| Trained on | 150,000 Source-1 entities, GroupKFold ×5 |
| **Offline OOF macro F0.5** | **0.9532** |
| Per country (OOF) | US 0.9620 · India 0.9399 |
| Blocking | word TF-IDF per country, K=30, recall 0.9498 |
| Decision layer | isotonic calibration → per-entity expected-F0.5, soft assignment |
| Code commit | `6897207` |

**Result: 94.3 against 95.32 offline — about one point of optimism.**

That gap is the thing to explain, and France is the leading suspect: 15% of
the test set, zero training labels, so no cross-validation number covers it.
A 1-point drop is roughly what a weak 15% slice would cost. Worth measuring
before optimizing anything else.

**Expect the leaderboard to differ from 0.9532, and that is fine.** France is
15% of the test set and has zero training labels, so no cross-validation
number can measure it. The public leaderboard is also only a sample — the
private one decides final rankings. If the two disagree, believe CV.

## Sanity checks — all passed

The organisers' validator:

```
matching_results.tsv: 1732544 rows (108221 empty, 1624323 non-empty)
candidate_pairs.tsv:  1732544 rows (50 empty, 1732494 non-empty)
PASS - no blocking issues found. Safe to submit.
```

And the output matches what cross-validation predicted, which is the real
check that nothing broke between training and inference:

| | OOF (train) | this output |
|---|---|---|
| singleton rate | 6.28% | **6.25%** |
| links per entity | 3.138 | **3.125** |

Truth averages 3.46 links per entity, so predicting 3.13 is the expected
shape for an F0.5 metric — precision counts double, so the model is
deliberately conservative.

### The 50 empty rows in `candidate_pairs.tsv`

50 of the 1,732,544 Source-1 entities got **zero** candidates: their name and
address produce no token that survives the blocking vocabulary pruning
(`min_df=3`, `max_df=0.01`). They are written as empty rows rather than
dropped, because a missing Source-1 id is an outright rejection. This is
expected, it is 0.003% of the test set, and it is not a bug to chase.

## Reproducing this file

```bash
export AMLC_OUTPUT_DIR=/d/amlc_data/output
python src/score_test.py --run runs/007_v2_full --chunk 2000000
```

Scoring took 117 minutes for 51,974,499 candidate pairs (52 chunks across two
passes). Test blocking was already cached, saving another 52 minutes.

## Checksums

```
matching_results.tsv      md5 cc101bdba923078a55bc2863e5357d1f   93,955,304 bytes
matching_results.tsv.gz   md5 f2079c6666fd48c4bdaac8f05351f3fa   39,458,396 bytes
candidate_pairs.tsv                                             693,990,794 bytes
```

Verify after unzipping (optional):

```bash
md5sum matching_results.tsv    # cc101bdba923078a55bc2863e5357d1f
```

See [`MANIFEST.md`](MANIFEST.md) for the full configuration.
