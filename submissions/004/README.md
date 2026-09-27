# Submission 004 — v4 contention fix, no hopeso

**Status: HOLD. Validated and ready, but do not spend a slot on it yet.**

We have **3 uploads left**. This one is our best *non-hopeso* model, and the
hopeso boxes finishing this evening are expected to beat it. Its real job is to
be an ensemble member and a fallback. Upload it only if the hopeso runs fail.

---

## Krisha — if and only if the team says to upload

1. Download `matching_results.tsv.gz` (39 MB) from this folder.
2. Unzip it → `matching_results.tsv`, ~96 MB, 1,732,545 lines.
3. Upload the **`.tsv`** to Unstop. Not the `.gz`.
4. Post the score back here.

---

## What this is

| | |
|---|---|
| run | `runs/011_v4_contention` |
| model | 150k sample, two-stage LightGBM, e5-small band reranker (0.2–0.8) |
| candidates | base TF-IDF top-30, blocking recall **0.9498** |
| OOF | **0.9644** (stage 1 0.9494 → stage 2 0.9644) |
| per country | India 0.9514, US 0.9730 |
| decision | soft assign, global threshold 0.71 |

The change over 003 is the **contention fix**: stage-2 claim features are now
computed over the whole 2.2M-entity frame instead of the 150k training sample,
so `n_claims` means at training what it means at inference. Measured
like-for-like, training went from 1.957 claims per record to 6.717 against
test's 5.549 — from 3.5x too low to 1.2x too high.

OOF rose 0.9626 → **0.9644**, which I had predicted it could not do (the fix
targets a train/test mismatch that OOF cannot see). It did, so the sample-only
claim counts were noisy as well as mis-scaled.

## Why it is on hold

Blocking recall here is **0.9498**. The hopeso frames built this afternoon
measure **0.9630** on the same 2.2M-entity train frame, and hopeso boxes are
already reporting stage 1 at 0.9526–0.9533 against this run's 0.9494.

So this model is a generation behind by construction. Keeping it is still
worth it: it was built on *different candidates*, which makes it genuinely
decorrelated from the hopeso models and therefore a useful ensemble member.

## Validation

```
validate_submission.py --check-ids
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (109433 empty, 1623111 non-empty)
  candidate_pairs.tsv:  1732544 rows (50 empty, 1732494 non-empty)
PASS - no blocking issues found.
```

Predicted singleton rate 0.0632 against a true train rate of 0.0563; 3.22
matches per entity. France 0.0554 / India 0.0687 / US 0.0594.

The 50 empty candidate rows are expected, not a bug: no token of those 50 S1
entities survives df pruning, so they get no candidates and are written empty.

Cached score arrays for ensembling: `interim/testp1_488b25ee4baf.npy` (stage 1)
and `interim/testp_488b25ee4baf.npy` (final).
