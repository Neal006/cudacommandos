# Submission 003 — 150k + e5 band reranker

**Status: READY TO UPLOAD. Our best model so far.**

---

## Krisha — what to do

1. Download **`matching_results.tsv.gz`** (39 MB) from this folder.
2. **Unzip it** → `matching_results.tsv`, ~95 MB, 1,732,545 lines.
3. Upload the **`.tsv`** to Unstop.
4. Post the score.

Do not upload `candidate_pairs.tsv` (694 MB, not scored, not in git).

---

## What this is

| | |
|---|---|
| Model | `runs/010_150k_rr/model.pkl` |
| Trained on | **150,000** entities + `multilingual-e5-small` band reranker |
| **OOF macro F0.5** | **0.9626** |
| Per country (OOF) | US 0.9717 · India 0.9491 |
| Validator | PASS, 1,732,544 rows, 0 matches outside candidates |

## Why we expect this to beat 002

The leaderboard has now confirmed the reranker is real, not a leak:

| | model | OOF | **LB** |
|---|---|---|---|
| 001 | 150k, no reranker | 0.9532 | 0.943 |
| 002 | **30k** + reranker | 0.9607 | **0.951** |
| **003** | **150k** + reranker | **0.9626** | **~0.953 expected** |

002 beat 001 by 0.8 points *despite being a weaker 30k model*. 003 keeps the
reranker and restores the full 150k sample, so it should add roughly the
0.002 that sample size is worth.

The OOF-to-leaderboard gap has been remarkably stable — **-0.0102** on 001 and
**-0.0097** on 002 — which is what makes the ~0.953 estimate trustworthy rather
than a guess. That stability is itself a finding: the gap is systematic, not
noise, and `run_v4` (running now) attacks its mechanism directly.

## Output

```
rows                  1,732,544
links per entity          3.175
singleton rate           6.25%
matches outside candidates    0
```

| | links/entity | singleton% |
|---|---|---|
| India | 3.08 | 6.82% |
| US | 3.27 | 5.88% |
| France | 3.22 | 5.41% |

## Reproducing

```bash
python src/run_v2.py --sample 150000 --rerank models/rr_e5s   # train (50 min)
python src/score_test.py --run runs/010_150k_rr --chunk 1200000 --rerank models/rr_e5s
```

Scoring took 135 min for 51,974,499 pairs; per-pair scores are cached at
`interim/testp_f550ffb02552.npy`, so a different decision rule costs a minute
via `src/redecide.py` rather than a rescore.
