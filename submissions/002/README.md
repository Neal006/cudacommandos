# Submission 002 — e5 band reranker

**Status: UPLOADED. Scored 0.951.** Superseded by 003, then 004.
Kept for the record and for back-testing on a scratch account.

Second submission. Its job is as much to answer a question as to score well —
see [What this one is really for](#what-this-one-is-really-for).

---

## Krisha — what to do

1. Download **`matching_results.tsv.gz`** from this folder (39 MB).
2. **Unzip it** → `matching_results.tsv`, 95 MB, 1,732,545 lines.
   - Windows: right-click → 7-Zip → *Extract Here*
   - Git Bash / Mac: `gzip -d matching_results.tsv.gz`
3. Upload the **`.tsv`** (not the `.gz`) to the Unstop portal.
4. **Post the score in the team chat.**

Do not upload `candidate_pairs.tsv` — it is not scored, it is 694 MB, and it is
not in git.

If the portal rejects it, post the exact error rather than retrying; this file
already passed the organisers' validator locally, so a rejection is something
new and a blind retry costs a second slot.

---

## What this is

| | |
|---|---|
| Model | `runs/ab_e5s/model.pkl` + `multilingual-e5-small` band reranker |
| Trained on | 30,000 Source-1 entities |
| Offline OOF macro F0.5 | **0.9607** |
| Per country (OOF) | US 0.9697 · India 0.9469 |
| Code commit | `f88b95e` |

## What this one is really for

Submission 001 (150k, no reranker) scored **0.943**. This model scores 0.9607
offline against 001's 0.9532 — but **the two are not comparable**, because this
is a 30k-sample model and 001 was 150k. Its honest baseline is the 30k
no-reranker run at 0.9512, over which the reranker adds **+0.0095**.

The reason to spend a slot on it is that **it settles an open question**. Two
completely different rerankers — `multilingual-e5-small` and mmBERT-base via
`laya` — each lifted the 30k score by the same ~0.0096, and an earlier e5 run
gave +0.009. Three numbers within 0.0006 of each other, across architectures
with different tokenizers and parameter counts, is not what genuine model
quality looks like; it is what one shared confound looks like. The suspect is
record overlap between the reranker's training pairs and the classifier's
training sample.

Held-out test data cannot be leaked into, so:

- **LB ≈ 0.95+** → the gain is real. A 150k model plus the reranker becomes the
  obvious next run, and Neal's `finetune` branch is worth pursuing.
- **LB ≈ 0.943 or below** → the gain was a leak. It is fake for both rerankers,
  and the entire reranker line should be dropped.

Either answer is worth more than the score itself, because right now several
people's next day of work depends on not knowing.

## Sanity checks — all passed

```
matching_results.tsv: 1732544 rows (107972 empty, 1624572 non-empty)
candidate_pairs.tsv:  1732544 rows (50 empty, 1732494 non-empty)
entities_with_matches_outside_candidates: 0
PASS - no blocking issues found. Safe to submit.
```

| | OOF | this output |
|---|---|---|
| singleton rate | 6.2% | 6.23% |
| links per entity | 3.19 | 3.18 |

## The per-country numbers, measured for the first time

Run 007's scoring crashed before printing these, so this is the first look:

| | links/entity | singleton% | blocking top_sim p10 | model top_p p10 |
|---|---|---|---|---|
| India | 3.085 | 6.81% | 0.8580 | 0.9808 |
| US | 3.267 | 5.87% | 0.8027 | 0.9939 |
| **France** | **3.266** | **5.35%** | **0.9128** | **0.9985** |

**France is not the weak country on any label-free measure.** It has the best
blocking similarity, the highest model confidence and an output distribution
matching the US. The standing assumption — that France is broken and needs
pseudo-labels or its own calibration — is not supported by this.

That matters for what the 0.943 means. Even if France scored exactly as well as
the US, the expected total would be 0.9517 against an actual 0.943 — so **0.009
is unexplained by France at all**. The likelier story is that all three
countries drop together, because test has **1.73M Source-1 entities against our
150k training sample**: 11.5× the contention for the same records, and every
extra entity is another chance for the wrong one to claim a record. Precision
degrades with entity count and out-of-fold validation at 150k cannot see it.

If that is right, the lever is enforcing the partition harder rather than
anything France-specific — and `assign='hard'` measured as worth only +0.00007
out-of-fold *precisely because* OOF has 11.5× less contention than test.

## Reproducing

```bash
export AMLC_OUTPUT_DIR=/d/amlc_data/output
python src/score_test.py --run runs/ab_e5s --chunk 1200000 --rerank models/rr_e5s
```

61 minutes, reusing the cached stage-1 scores. Per-pair scores are cached to
`interim/testp_50cf8414638d.npy`, so a different decision rule now costs about a
minute via `src/redecide.py` instead of a rescore.

## Checksums

```
matching_results.tsv      md5 a874bf6cbb6e6369f26be9a508a6fe0c   95,221,072 bytes
candidate_pairs.tsv                                             693,990,794 bytes
```

See [`MANIFEST.md`](MANIFEST.md) for the full configuration.
