# What we're running, what to expect, and can we reach 99%

**Written 2026-09-27 12:30 IST. Deadline 23:59 — 11.5 hours.**

Compute stopped being the constraint this morning. This is what changes as a
result, and what doesn't.

---

## 1. What is running right now

| Where | What | Hardware | ETA | Expect |
|---|---|---|---|---|
| **laptop** | `011_v4_contention` — contention fix, train-only | 10 cores / 24 GB | **~12:55** | OOF ~0.9628 (flat vs 003) |
| **AWS** | `pytorch-training-…-06-43-52` — contention **+ holdout calibration** + reranker, train **and** score | **96 vCPU / 768 GB** | **~14:00–14:30** | OOF ~0.963, **LB 0.958–0.963** |
| **SageMaker T4** | Krisha's e5 reranker fine-tune | T4 16 GB | hers | beat AUC 0.99907 or discard |

### What to expect from each, and why

**Local v4 (contention only).** Its OOF will be ~flat against 003's 0.9626 —
folds are coming in at 0.9628 / 0.9629 / 0.9627. **That is the expected
result, not a disappointment.** OOF is measured with 1.96 competitors per
record; test has 5.55. A fix that makes training match test *cannot* show up
in a metric computed on the training distribution. Its value is only visible
on the leaderboard. If we score it: **LB ~0.956–0.960**.

**Cloud v4 (contention + calibration).** Both halves of the measured −0.010
gap. Contention fixes the *features*; the holdout fixes the *calibrator*,
which was fitted on single-model OOF scores and applied to five-model-averaged
test scores. **LB ~0.958–0.963.** This is our best shot today.

**Krisha's fine-tune.** Upside only if it beats the existing reranker's
`valid_auc 0.99907` / `valid_logloss 0.03339`, and then only if the downstream
30k A/B on India confirms it. Do not adopt on the intrinsic number alone.

---

## 2. Can we reach 99%? No — and here is the arithmetic

The score is bounded by retrieval. A perfect matcher given blocking recall R
scores `1.25R/(0.25+R)`. Our measured leaderboard offset is −0.010.

| blocking recall | perfect-matcher ceiling | at our current 97.2% efficiency | at 99% efficiency |
|---|---|---|---|
| **0.9498 (today)** | 0.9903 | 0.9626 → **LB 0.953** | 0.9804 → LB 0.970 |
| 0.975 | 0.9949 | 0.9670 → LB 0.957 | 0.9850 → LB 0.975 |
| 0.99 | 0.9980 | 0.9700 → LB 0.960 | 0.9880 → LB 0.978 |
| 1.00 (perfect retrieval) | 1.0000 | 0.9720 → LB 0.962 | 0.9900 → LB 0.980 |

Read the bottom-right cell. **Even with perfect retrieval and a matcher at 99%
of its ceiling, the leaderboard lands at 0.980.** To reach 0.99 we would need
the −0.010 gap fully eliminated *and* a matcher at ~99.5% of ceiling *and*
near-perfect recall, simultaneously.

We are at **97.2%** matcher efficiency. Going to 99.5% means cutting matcher
error by **82%** — on a problem where 80,000 Source-1 businesses share both a
name and a locality with another business, and 26% of records match nothing.
A meaningful share of that error is irreducible from text alone.

And the decisive point: **the current leader is at 0.988.** Reaching 0.99 means
beating the best team in the competition, today, with 11 hours left. That is
not a compute problem and more credits will not buy it.

### What the compute *does* buy

| Lever | Was | Now (96 cores) | Worth |
|---|---|---|---|
| multi-retriever union | ~4 h blocking | ~25 min | **+0.005 – 0.008** |
| K=30 → 50 | +35M pairs, infeasible on 24 GB | trivial at 768 GB | +0.002 |
| 400k–2.2M training sample | 6 h+ | ~1 h | +0.002 |
| Optuna on LightGBM params | never fitted | parallel trials | +0.002 – 0.005 |

**Honest best case for today: 0.965 – 0.970.** That is +0.012 to +0.017 over
our current 0.953, and it would move us several hundred ranks in a field this
dense. It is a good outcome. It is not 0.99.

---

## 3. The revised plan

Compute is free, so the ordering changes: things previously rejected *on cost*
are back, and things rejected *on value* stay rejected.

### Phase A — land what's running (now → 15:00)

1. Local v4 OOF at ~12:55. Do **not** judge it against 0.9626; see §1.
2. Cloud v4 finishes ~14:30 → **submission 004**.
3. Ensemble the cached score arrays from 003 + local v4 + cloud v4
   (`src/ensemble.py --dry-run` first: it reports member agreement, so we can
   tell whether a slot is worth spending before spending it).

### Phase B — recall, which compute just made cheap (14:00 → 17:30)

This is the one genuinely new option. Our blocking misses 5.02% of true pairs,
and we profiled exactly where:

- **empty-address records: 7.8× over-represented** among misses
- **native-script (Indic) names: 3.6× over-represented**

The empty-address case is our own design flaw. Blocking searches
`name + " " + address` as one blob, so a record with no address is diluted in
a space where everything else has both, sinks below the top-30 cut, and is
never seen again. The feature set even has an `addr_empty_any` flag with
**exactly 0.0 importance** — because by the time a pair reaches the model, the
records that mattered were never retrieved.

Plan: a **union of retrievers**, each contributing its own top-k:

| pass | targets | cost on 96 cores |
|---|---|---|
| A: name + address (current) | general | ~6 min |
| B: **name only** | empty-address records | ~6 min |
| C: char 3-gram on transliterated name | typos, Indic scripts | ~12 min |
| D: address only | records whose name is a domain | ~5 min |

Gate each one: keep it only if it raises the measured recall ceiling by
≥ 0.002. Expected recall 0.950 → 0.970-0.975, worth **+0.005-0.008 LB**.

Cost: ~1 h to write, ~30 min to block, ~1.5 h to retrain and score on the big
box. Fits, but only just — and it invalidates every candidate cache, so there
is no partial fallback.

### Phase C — hard stop 20:00

Package, validate, documentation. Nothing that isn't finished by 20:00 ships.

### Explicitly still rejected

More credits do not change these:

- **Tuning `TOP_K`.** It sets a monotone ceiling with no interior optimum.
- **Optuna over the feature count.** 46 is how many features someone wrote.
- **France-specific work.** France is the *strongest* country on every
  label-free measure, and even a perfect France leaves 0.009 unexplained.
- **laya.** Lost to e5 on India at 5.8× the cost.

---

## 4. What we learned about using the cloud

Worth recording, because the intuition was wrong in both directions.

**Wrong assumption #1: "GPU will help."** The pipeline is ~95% CPU — sparse
matmul, rapidfuzz, LightGBM. The GPU does ~7% of the work (the reranker band,
1.5% of pairs). The useful cloud instance is a *large CPU* box.

**Wrong assumption #2: "we only have 4-vCPU instances."** I checked the
Training quota pool and stopped. Processing jobs draw on a **separate pool**,
and between them we have up to **192 vCPU and 768 GB**. `ml.r5.24xlarge` is
9× the laptop's cores and 32× its RAM.

At 768 GB, all the chunking we built to survive 24 GB is simply unnecessary —
the 52M-pair frame is processed in one pass.

**What made it viable:** the dataset was *already in S3* from Krisha's upload.
Server-side copy to an accessible bucket took **9 seconds for 2.4 GB**. Had we
needed to push it from home broadband, the arithmetic would have been very
different — setup was 35 minutes against a 7-hour local job, and most of that
was SDK friction, not data movement.

Five snags, each a one-liner: cross-account bucket policy → copy to our own;
`GetObjectTagging` denied → `--copy-props none`; SDK v3 removed
`sagemaker.pytorch` → downgrade to v2; `aws login` credentials → `botocore[crt]`;
worktree ran system python → use the venv.
