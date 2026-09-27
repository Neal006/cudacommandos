# `chasing99` — what Priyanshu is doing on this branch

**Updated 2026-09-27 09:00 IST.** Deadline tonight 23:59.

Working branch for the final push. Isolated so nobody's pushes collide with a
running 4-hour job. Everything submittable gets cherry-picked to `main`.

---

## Where we stand

| Sub | Model | OOF | **Leaderboard** |
|---|---|---|---|
| 001 | 150k, no reranker | 0.9532 | **0.943** |
| 002 | 30k + e5 reranker | 0.9607 | **0.951** |
| 003 | **150k + e5 reranker** | **0.9626** | uploaded, awaiting score |

All three are in `submissions/<nnn>/` on `main`. Krisha uploads: download the
`.tsv.gz`, unzip, upload the `.tsv`.

---

## Four findings that changed what we're doing

### 1. The band reranker is real (+0.008 LB)

We suspected a leak: e5 and laya each added the same ~0.0096 out-of-fold, and
two different architectures agreeing that closely looks like a shared
confound, not model quality. **The leaderboard settled it.** Submission 002 —
a *weaker* 30k model — beat the 150k no-reranker model 0.951 to 0.943. Held-out
test cannot be leaked into, so the gain transfers.

Consequence: the reranker stays, and Neal's `finetune` branch is worth
pursuing.

### 2. laya loses to e5 — reject it

Head to head, same band, same 514,335 training pairs, same 30k GBDT sample,
only the model swapped:

| | baseline | laya | e5 |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| train time | — | 132 min | **23 min** |

laya is **worse on India** — the one metric the hypothesis was built to win —
at 5.8x the training cost and a 647 MB checkpoint. The premise had a flaw: the
incumbent is `intfloat/multilingual-e5-small`, already multilingual, so the
test was multilingual-vs-multilingual rather than multilingual-vs-English.

### 3. France is NOT our weak country

This one reverses a standing assumption. Per-country test statistics,
measured for the first time (run 007 crashed before printing them):

| | blocking top_sim p10 | model confidence p10 | links/ent | singleton% |
|---|---|---|---|---|
| India | 0.8580 | 0.9808 | 3.085 | 6.81% |
| US | 0.8027 | 0.9939 | 3.267 | 5.87% |
| **France** | **0.9128** | **0.9985** | 3.266 | 5.35% |

France has the **best** blocking similarity, the **highest** model confidence,
and an output distribution matching the US. The planned France work —
pseudo-labels, France-specific calibration, raising its `miss` — was all
premised on the opposite and would likely have made things worse.

Also: even if France scored exactly as well as the US, expected total is 0.9517
against an actual 0.943, so **0.009 is unexplained by France at all**.

### 4. The OOF-to-leaderboard gap is a real bug, and we found it

The gap is stable and systematic: **-0.0102** (001) and **-0.0097** (002).
Cause: stage 2's competition features are raw counts over whatever Source-1
set is present.

```
mean n_claims (entities competing for one record)
  train  30k   1.371
  train 150k   1.957
  TEST         5.549      <- 2.8x the training distribution
```

`n_strong_claims` is the **second most important feature in stage 2** (2.52M
importance, behind only p1's 14.35M). It means a different thing at inference
than it did in training.

Cross-validation cannot see this — the OOF split carries the same wrong
contention as the training data — which is exactly why it only showed up on
the leaderboard.

---

## Running now: `run_v4.py` (the contention fix)

Computes the claim features over the **full 2.2M-entity train frame** while
still training the GBDT on a 150k sample, so train contention becomes ~6.6
against test's 5.55 instead of 1.96.

```
A  block every train S1                 ~66M pairs
B  stage 1 on the sampled rows          out of fold, unchanged
C  stage-1 score over ALL 66M pairs     chunked
D  claim features over the whole frame  <- the fix
E  stage 2 on the sample, honest claims
F  decision layer -> test
```

Plus the e5 reranker and `--rounds 4000` (fold 1 hit the 2000 cap at 150k, so
it was still improving when cut off).

**OOF number expected ~13:30.** It runs `--train-only` first: if it doesn't
beat 0.9626 we keep submission 003 and have spent 4 hours, not 6.5.

Leakage: sampled rows keep their out-of-fold p1. Non-sampled rows get the fold
mean, which is sound because they are never training targets — they exist only
to create the competition test will actually have.

---

## Why 99% isn't reachable, and what is

```
blocking recall 0.9498  ->  perfect-matcher ceiling  0.9903 OOF
measured OOF-to-LB gap                              -0.010
->  maximum possible LB at our current blocking      ~0.980
```

Even a flawless matcher tops out near 0.980 with the candidates we generate.
LB 0.99 would need OOF 1.000.

We currently capture **97.2%** of that ceiling. Two levers, and the sizes are
not what we assumed:

| Lever | Worth | Cost |
|---|---|---|
| matcher efficiency 97.2% -> 99% | **+0.017 OOF** | cheap, no re-blocking |
| recall 0.95 -> 0.98 | +0.0056 OOF | ~4 h of blocking |

**The matcher has ~3x the headroom of retrieval at a quarter of the cost.**
That is why v4 (a matcher fix) is running instead of a multi-retriever union.

Realistic landing zone: **0.955-0.960**.

---

## New tools on this branch

| File | What it does |
|---|---|
| `src/run_v4.py` | contention-corrected training (above) |
| `src/score_test.py` | score test from a trained run, no retraining; caches per-pair scores |
| `src/redecide.py` | re-run the decision layer over cached scores in ~1 min, optionally per country |
| `src/diagnose_country.py` | label-free per-country diagnosis; separates blocking failure from matcher failure |
| `tools/package_submission.sh` | builds the graded zip; refuses if rows, validator or docs are wrong |
| `tests/test_chunk_equiv.py` | chunked test scoring == unchunked, with a negative control |
| `tests/test_v4_contention.py` | asserts the contention fix is not a no-op |

**Score caching is the quality-of-life win.** Test scoring is ~2 h; the
decision layer is ~1 min. They used to be welded together, so nobody ever
tried a different threshold. Now `interim/testp_<key>.npy` holds the per-pair
scores and any decision variant is a minute.

---

## Things we checked that turned out to be non-issues

- **"One record matched to several S1 costs precision."** Measured: 137 of
  5,414,100 records, 0.0025%. `assign='soft'` already enforces the partition;
  fixing the rest is worth ~0.00002.
- **"We under-predict (3.17 vs 3.46 true links)."** True, but correct — F0.5
  weights precision double, so the expected-F decoder declines marginal links
  on purpose. Pushing toward 3.46 lowers the score.
- **Tuning `TOP_K` with Optuna.** K sets a monotone ceiling with no interior
  optimum, and the ceiling is not the binding constraint.
- **Optuna over "the 46 features".** 46 is how many were written, not a
  hyperparameter. LightGBM's actual hyperparameters *are* untuned — that is a
  real Optuna target nobody has used.

## Code audit

Verified against the spec rather than assumed: the F0.5 implementation matches
`1.25PR/(0.25P+R)` algebraically including edge cases; stage 2 trains on
out-of-fold stage-1 scores with the same folds; calibration is cross-fitted;
the reranker has a hard S1-entity leak guard; output formatting dedupes and
sorts; blocking top-k is correct. Two minor issues, both immaterial: the
decision layer is tuned on the same OOF it calibrates on (worth 0.0006
total), and `split_stats` is per-split so IDF features shift slightly between
train and test.

## Status of other branches

| Branch | Measured result |
|---|---|
| `lgb-xgb-cat-blend` | **none** — its own notes say it has never run on real data |
| `finetune` | **none** — code only |
| `sentence-transformer-…` | 0.9083 at 2k, vs our 0.9296 baseline at the same size — **worse**, and its own summary notes the baseline was never run |

Worth knowing before more time goes into them.
