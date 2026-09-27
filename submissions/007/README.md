# Submission 007 — final output

**Status: VALIDATED, READY TO UPLOAD.**

Team's final submission for 2026-09-27.

---

## Upload

1. Download `matching_results.tsv.gz` (40 MB).
2. Unzip → `matching_results.tsv`, ~97 MB, **1,732,545 lines**.
3. Upload the **`.tsv`** to Unstop.

---

## Validation

```
validate_submission.py
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (102039 empty, 1630505 non-empty)
PASS - no blocking issues found. Safe to submit.
```

Row count exact, header correct, S1 ordering correct, no duplicate entities.

## Singleton rate

| | predicted singleton rate | vs true train rate 0.0563 |
|---|---|---|
| 006 (hopeso + e5, LB 0.959) | 0.0614 | +0.0051 |
| **007 (this file)** | **0.0589** | **+0.0026** |

Closer to the true rate than anything else we produced. Under macro F0.5 the
singleton decision is high-stakes — an entity with no true matches scores 1 if
predicted empty and 0 for any prediction at all — so a predicted rate nearer
the truth is a positive signal about the decision layer.

No offline score accompanies this file, so it cannot be ranked against the runs
below using `LB = OOF - 0.0098`, the relation measured across our four uploads.
Its leaderboard score is known only once submitted.

## Our own pipeline, for the record

| sub | model | recall | OOF | LB |
|---|---|---|---|---|
| 003 | 150k + e5 reranker | 0.9498 | 0.9626 | 0.953 |
| 004 | + contention fix | 0.9498 | 0.9644 | not uploaded |
| 005 | hopeso, no reranker | 0.9630 | 0.9602 | not uploaded |
| **006** | **hopeso + e5** | **0.9630** | **0.9685** | **0.959** |

Box `wide` (e5 reranker on the 0.05–0.95 band) was stopped at 23:00: its
measured chunk rate put completion at 00:43, past the deadline, and the $200
ceiling at 23:33.
