# Everything we tried, what it was worth, and what we submitted

Neal — this is the scoreboard. Every number here comes from a `summary.json`
on disk or the Unstop leaderboard; nothing is estimated unless it says so.

---

## The three submissions

| # | Model | Offline OOF | **Leaderboard** | Rank |
|---|---|---|---|---|
| 001 | 150k entities, no reranker | 0.9532 | **0.943** | 934 |
| 002 | 30k entities + e5 reranker | 0.9607 | **0.951** | — |
| 003 | 150k entities + e5 reranker | 0.9626 | **0.953** | 1760 |

Files live in `submissions/<nnn>/` on `main` — gzipped TSV plus a manifest
recording the exact commit, model and config that produced it.

Worth noticing: **003 scored higher than 001 but ranked worse.** The field
filled in behind us between uploads. The pack around 0.95 is dense, so a small
gain moves a lot of places — and standing still costs a lot of places.

### The offline→leaderboard gap is a constant

```
001    -0.0102
002    -0.0097
003    -0.0096
```

Three quite different models agreeing within 0.0006. This turned out to be
the most useful number we have: it makes the leaderboard predictable from
offline scores. We predicted 003 at ~0.953 before uploading and it came back
0.953.

It's also a bug, not noise — see §5 of `02_where_the_model_goes_wrong.md`.

---

## The ablation: what each change was actually worth

### Pipeline layers (150k sample)

| Layer | OOF | Gain |
|---|---|---|
| pre-v2 feature set | 0.9266 | — |
| stage 1 (46 features, LightGBM) | 0.9500 | **+0.0234** |
| stage 2 (context features) | 0.9526 | +0.0026 |
| decision layer (expected-F) | 0.9532 | +0.0006 |

The feature set did the heavy lifting. Stage 2 adds a tenth of that, and the
decision layer a tenth again — worth keeping because they cost nothing at
inference, but not where the score comes from.

### Training sample size

| Sample | OOF | India | US |
|---|---|---|---|
| 2,000 | 0.9296 | 0.9018 | 0.9471 |
| 30,000 | 0.9512 | 0.9369 | 0.9606 |
| 150,000 | 0.9532 | 0.9399 | 0.9620 |

Sharply diminishing. 2k→30k bought +0.0216; 30k→150k bought +0.0020 for 5x
the data. Extrapolating, 150k→400k would be worth maybe +0.0015 for six hours
of training — which is why we didn't.

**A subtlety worth knowing:** part of why more data helps isn't the data at
all. Bigger samples move the contention statistics closer to test (1.37 → 1.96
against test's 5.55). So the sample-size curve and the contention bug are
partly the same phenomenon.

### The band reranker — the biggest single win

| | baseline | + reranker |
|---|---|---|
| 30k OOF | 0.9512 | **0.9607** |
| 150k OOF | 0.9532 | **0.9626** |
| India (150k) | 0.9399 | **0.9491** |
| US (150k) | 0.9620 | **0.9717** |

About **+0.0095** wherever we measured it, and it helped India most — the
country where the hard cases live.

**We nearly threw this away.** Two completely different architectures (e5 and
laya) each added the same ~0.0096, and an earlier run had given +0.009. Three
numbers within 0.0006 across different tokenizers and parameter counts looks
exactly like a shared confound rather than model quality, and the suspected
cause was record overlap between reranker training and the GBDT sample.

The leaderboard settled it: **submission 002, a *weaker* 30k model with the
reranker, beat the 150k model without it, 0.951 to 0.943.** Held-out test
can't be leaked into. The gain is real.

### laya vs e5 — head to head, rejected

Same band, same 514,335 training pairs, same valid split, same 30k sample,
only the model swapped:

| | baseline | laya | e5 |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | **0.9607** |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| valid AUC | — | 0.99893 | **0.99907** |
| trainable params | — | 55.1M | 22M |
| train time | — | 132 min | **23 min** |
| band inference | — | 55 s | **15 s** |

**laya lost on India — the single metric its hypothesis was built to win** —
and was 0.0001 better overall, which is noise. At 5.8x the training cost and a
647 MB checkpoint.

The premise had a flaw worth recording: laya was proposed because e5 supposedly
handles native-script names poorly. But the incumbent is
`intfloat/multilingual-e5-small` — already multilingual. So the A/B was
multilingual against multilingual, which explains a null result more simply
than anything about mmBERT.

### Things that didn't work, or turned out not to be problems

| Idea | Result |
|---|---|
| Sentence-transformer embedding features (Krisha's branch) | 0.9083 at 2k vs our **0.9296** baseline at the same size — worse. Her summary notes the baseline was never run; this is it |
| LightGBM + logistic regression blend | 0.8959 vs 0.9083 — the ensemble hurt, dropped |
| Raising K from 30 to 50 | +0.002 of *ceiling* for 35M extra pairs, while we sit 3.7 points below the ceiling we already have |
| "One record claimed by several entities costs precision" | Measured: **137 of 5,414,100** records, 0.0025%. `assign='soft'` already handles it. Worth ~0.00002 |
| "We under-predict, 3.17 vs 3.46 true links" | True and **correct** — F0.5 weights precision 2x, so declining marginal links is optimal. Pushing to 3.46 lowers the score |
| France being our weak country | Reversed — France is at or near the top of every label-free measure. See `01_france_diagnostic.md` |
| Tuning `TOP_K` with Optuna | K sets a monotone ceiling with no interior optimum, and the ceiling isn't the binding constraint |
| Optuna over "the 46 features" | 46 is how many features someone wrote, not a hyperparameter |

That last pair is worth separating from the rest: **Optuna is the right tool,
just pointed at the wrong thing.** LightGBM's actual hyperparameters —
`num_leaves`, `min_data_in_leaf`, `feature_fraction`, `lambda_l1/l2` — are all
still at defaults and have never been tuned. That's a genuine unexplored
lever; we just ran out of hours.

---

## What's running now

**`run_v4`** — the contention fix. Stage 2's claim features computed over the
full 2.2M-entity frame instead of the 150k sample, so they mean the same thing
in training as at inference.

Measured live on the job: contention went from **1.96 to 6.717**, against
test's 5.55. It now brackets test instead of sitting 2.8x below it.

Stage 1 came in at 0.9494 against 003's 0.9490 — flat, which is the *right*
signal, since v4 only changes stage 2. If stage 1 had moved, something would
be wrong.

**One thing to flag, because I set the acceptance test up wrong at first:**
v4 deliberately trades offline fit for test transfer. Its OOF is measured in
the low-contention regime, so matching test contention can leave OOF flat or
even slightly lower while the leaderboard score rises. Judging it against
003's 0.9626 would throw away the only thing it was built to do.

| v4 OOF | how to read it |
|---|---|
| < 0.955 | something broke — discard |
| 0.958 – 0.966 | working as intended — score it |
| > 0.966 | the raised round cap is helping too |

---

## Where this can realistically land

```
blocking recall 0.9498  ->  perfect matcher scores  0.9903 offline
minus the measured gap                              -0.010
->  best possible leaderboard with our candidates    ~0.980
```

We're at 0.953, which is **97.2% of our own ceiling**. So:

- **99% is not reachable.** It would need an offline score of 1.000.
- **98% is not reachable either** with the candidates we generate. The leader
  at 0.988 is *retrieving* things we never propose — a different pipeline,
  not a tuning gap.
- **0.956–0.962 is realistic today**, if v4 closes part of the 0.010 gap and
  the score-level ensemble adds its usual bit.

And the two levers aren't the sizes we assumed:

| Lever | Worth | Cost |
|---|---|---|
| matcher efficiency 97.2% → 99% | **+0.017 offline** | cheap, no re-blocking |
| blocking recall 0.95 → 0.98 | +0.0056 offline | ~4 h of blocking |

**The matcher has about 3x the headroom of retrieval at a quarter of the
cost.** I had this backwards for a while and corrected it — it's why v4 is a
matcher fix rather than a multi-retriever union.

---

## Reproducing anything here

```bash
# train and score end to end
python src/run_v2.py --sample 150000 --rerank models/rr_e5s

# score test from an existing run, no retraining
python src/score_test.py --run runs/010_150k_rr --chunk 1200000 --rerank models/rr_e5s

# try a different decision rule on cached scores (~1 min, not 2 hours)
python src/redecide.py --scores interim/testp_<key>.npy --cfg soft:expected_f:0.1

# blend cached score arrays from several runs
python src/ensemble.py --scores a.npy b.npy --dry-run
```

Per-pair test scores are cached to `interim/testp_<key>.npy`, keyed on a hash
of the model file, reranker and row count. That's what makes the last two
commands minutes instead of a full rescore — before caching, trying a second
threshold meant two hours, so nobody ever tried one.
