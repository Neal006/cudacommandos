# `chasing99` — state of play

**Updated 2026-09-27 10:10 IST. Deadline tonight 23:59 (~14 h).**

Priyanshu's working branch for the final push. Isolated so nobody's push
collides with a running 4-hour job. Anything submittable gets cherry-picked to
`main`.

**Neal — if you change something, read §6 first.** Some things look like
obvious wins and are measured dead ends, and one of them is the test I set up
wrong.

---

## 1. Leaderboard

| Sub | Model | OOF | **LB** | Rank |
|---|---|---|---|---|
| 001 | 150k, no reranker | 0.9532 | **0.943** | 934 |
| 002 | 30k + e5 reranker | 0.9607 | **0.951** | — |
| 003 | **150k + e5 reranker** | 0.9626 | **0.953** | 1760 |

Files in `submissions/<nnn>/` on `main`. Note 003 scored *higher* than 001 but
ranked *worse* — the field filled in behind us. The pack around 0.95 is dense,
so small gains move many places.

### The OOF→LB gap is a constant

```
001   -0.0102
002   -0.0097
003   -0.0096
```

Three very different models, agreeing within 0.0006. This is the single most
useful number we have: it makes LB predictable from OOF, and it is what
`run_v4` exists to attack.

---

## 2. Running right now

**`run_v4.py --sample 150000 --rerank models/rr_e5s --rounds 4000 --train-only`**
(run id `011_v4_contention`, started 08:50)

| stage | status |
|---|---|
| block all 2.2M train S1 — India 883k | ✅ 38.2 min |
| — US 1.32M | 🔵 in progress |
| — France | pending |
| stage 1, 5 folds @ 4000 rounds | pending |
| stage-1 score over all 66M pairs | pending |
| claims + stage 2 + decision | **OOF ~13:40** |

### What it fixes

Stage 2's competition features are **raw counts over whatever Source-1 set is
present**:

```
mean n_claims (entities competing for one record)
  train  30k   1.371
  train 150k   1.957
  TEST         5.549     <- 2.8x the training distribution
```

`n_strong_claims` is the **second most important feature in stage 2**
(importance 2.52M, behind only `p1`'s 14.35M). It means a different thing at
inference than it did in training.

Cross-validation cannot see this: the OOF split carries the *same wrong
contention* as the training data. That is exactly why it only ever showed up
on the leaderboard.

v4 computes the claim features over the full 2.2M-entity frame while still
training the GBDT on 150k, so train contention becomes ~6.6 against test's
5.55. Leakage is preserved: sampled rows keep their out-of-fold `p1`;
non-sampled rows get the fold mean and are never training targets — they exist
only to create the competition test will have.

### ⚠️ How to judge it — I set this gate up wrong at first

I originally said "keep it if OOF beats 0.9626." **That is the wrong test.**

v4 deliberately trades OOF fit for test transfer. OOF is *measured* in the
low-contention regime, so matching test contention can leave v4's OOF flat or
slightly **lower** while its leaderboard score is higher. Judging on OOF alone
discards the only thing it was built to do.

| v4 OOF | read |
|---|---|
| < 0.955 | something broke — discard |
| 0.958 – 0.966 | working as intended — score it |
| > 0.966 | the rounds increase is paying off too |

Forecast: **LB 0.956–0.962**, most likely ~0.958.

---

## 3. Ready but not running

| Tool | What | Why not running |
|---|---|---|
| `src/ensemble.py` | blends cached per-pair scores, logit/mean, ~2 min | waiting on v4's scores to be a third member |
| `src/redecide.py` | re-runs the decision layer on cached scores, ~1 min, per-country capable | nothing left to try — see §6 |
| `src/diagnose_country.py` | label-free per-country diagnosis | already run, see §4 |

**Score caching is why those are cheap.** Test scoring is ~2 h; the decision
layer is ~1 min. They used to be welded together, so nobody ever tried a
second threshold. Now `interim/testp_<key>.npy` holds the per-pair scores.

---

## 4. Four findings that redirected the work

### The reranker is real (+0.008 LB)

We suspected a leak: e5 and laya each added the same ~0.0096 OOF, and two
architectures agreeing that closely looks like a shared confound. **The
leaderboard settled it** — 002, a *weaker* 30k model with the reranker, beat
the 150k model without it. Held-out test cannot be leaked into.

### laya loses to e5 — rejected

| | baseline | laya | e5 |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| train | — | 132 min | **23 min** |

laya is worse on India — the metric its hypothesis was built to win — at 5.8x
the cost. The premise assumed e5 can't read native script, but the incumbent
is `multilingual-e5-small`, already multilingual, so the A/B was multilingual
vs multilingual.

### France is NOT our weak country

First per-country test statistics ever measured (007 crashed before printing
them):

| | blocking top_sim p10 | model confidence p10 | links/ent | singleton% |
|---|---|---|---|---|
| India | 0.8580 | 0.9808 | 3.085 | 6.81% |
| US | 0.8027 | 0.9939 | 3.267 | 5.87% |
| **France** | **0.9128** | **0.9985** | 3.266 | 5.35% |

France has the **best** blocking similarity and the **highest** model
confidence. The planned France work — pseudo-labels, France-specific
calibration, raising its `miss` — was premised on the opposite and would
likely have hurt.

And the arithmetic: even if France scored exactly as well as the US, expected
total is 0.9517 against an actual 0.943. **0.009 is unexplained by France at
all.** That is what pointed at contention instead.

### Why 99% is not reachable

```
blocking recall 0.9498  ->  perfect-matcher ceiling  0.9903 OOF
measured OOF-to-LB gap                              -0.010
->  maximum possible LB at our current blocking      ~0.980
```

A flawless matcher tops out near 0.980 with the candidates we generate. The
leader at 0.988 is **retrieving** things we never generate, not out-matching
us. We capture 97.2% of our own ceiling.

| Lever | Worth | Cost |
|---|---|---|
| matcher efficiency 97.2% -> 99% | **+0.017 OOF** | cheap, no re-blocking |
| recall 0.95 -> 0.98 | +0.0056 OOF | ~4 h blocking |

The matcher has ~3x the headroom at a quarter of the cost. That is why v4 is a
matcher fix and not a multi-retriever union.

---

## 5. New code on this branch

| File | What |
|---|---|
| `src/run_v4.py` | contention-corrected training (§2) |
| `src/score_test.py` | score test from a trained run, no retraining; caches per-pair scores |
| `src/redecide.py` | decision layer over cached scores, ~1 min, per-country |
| `src/ensemble.py` | blend cached score arrays (logit/mean) with agreement diagnostics |
| `src/diagnose_country.py` | label-free per-country diagnosis |
| `tools/package_submission.sh` | builds the graded zip; refuses on bad rows, failed validator, or unfilled docs |
| `tests/test_chunk_equiv.py` | chunked scoring == unchunked, with a negative control |
| `tests/test_v4_contention.py` | asserts the contention fix is not a no-op |
| `Documentation_template.md` | filled in (it is graded, and it was empty) |

Also fixed: `run_v2.py` reranker support in the chunked path, score caching,
a reranker model-reload leak (42 loads in one run, killed it at chunk 42/44),
and `rate_stats` unpacking — it returns the per-country dicts *first*, and
both callers took them as the scalars. In `run_v2` that silently wrote dicts
into `summary.json`.

---

## 6. Do not re-derive these — they are measured dead ends

| Idea | Verdict |
|---|---|
| "One record matched to several S1 costs precision" | **137 of 5,414,100** — 0.0025%. `assign='soft'` already enforces it. Worth ~0.00002 |
| "We under-predict, 3.17 vs 3.46 true links" | True and **correct**. F0.5 weights precision 2x, so the expected-F decoder declines marginal links deliberately. Pushing to 3.46 *lowers* the score |
| Tune `TOP_K` with Optuna | K sets a **monotone ceiling** — no interior optimum — and the ceiling is not binding |
| Optuna over "the 46 features" | 46 is how many were written, not a hyperparameter |
| Raise K 30 → 50 | +0.002 of *ceiling* for 35M extra pairs, while we sit 3.7 points below the ceiling we have |
| More retrievers | +0.0056 for ~4 h of blocking; the matcher is worth 3x that |

**What Optuna *is* right for and nobody has used: LightGBM's actual
hyperparameters.** `num_leaves`, `min_data_in_leaf`, `feature_fraction`,
`lambda_l1/l2` are all at defaults.

---

## 7. Other branches

| Branch | Measured result |
|---|---|
| `finetune` | e5 fine-tune on SageMaker — **handed to Krisha**, running. Was laya, switched after the A/B. Her NaN crash was fp16 on a T4 (no bf16); fixed, see `docs/T4_FIX.md` |
| `lgb-xgb-cat-blend` | **none** — its own notes say it has never run on real data |
| `sentence-transformer-…` | 0.9083 at 2k vs our 0.9296 baseline at the same size — **worse**, and its summary notes the baseline was never run |

The blend branch's idea is sound but its cost is not: three libraries x five
folds triples training. `src/ensemble.py` gets most of the benefit by
averaging cached predictions in ~2 minutes instead.

---

## 8. Plan for the remaining hours

1. **~13:40** — v4 OOF. Judge by §2's table, *not* against 0.9626.
2. If it holds, score it (~2.3 h) → submission 004 around 16:00.
3. **Dry-run the ensemble** across 002 + 003 + v4 cached scores. `--dry-run`
   reports member agreement and how far the blend moves, so we can tell
   whether it is worth a slot *before* spending one.
4. Hard stop **20:00**. Anything unfinished does not ship; the last hours go to
   the package and documentation, not a run that might not land.

Three validated submissions are already banked, so there is no path to
finishing empty-handed.
