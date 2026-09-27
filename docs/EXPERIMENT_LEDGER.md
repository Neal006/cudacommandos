# Experiment ledger — every run, every branch, one page

**Updated 2026-09-26 20:00 IST.** Deadline 2026-09-27 23:59 IST.

The one-page answer to "what have we tried, what worked, what is actually our
best, and what is safe to submit". `EXPERIMENTS.md` is the long-form log; this
is the scoreboard. Every number here comes from a `summary.json` on disk or the
Unstop leaderboard — nothing is estimated.

---

## The headline

| | Score | Where |
|---|---|---|
| **Best SUBMITTED** | **LB 0.943** (rank 934) | run 007, submission 001 |
| **Best VERIFIED offline** | **OOF 0.9532** | run 007, 150k sample |
| **Best offline, UNVERIFIED** | OOF 0.9607 | `ab_e5s`, 30k + band reranker |
| Leader | ~0.988 | — |
| Perfect-matcher ceiling at our blocking | 0.9903 | K=30, recall 0.9498 |

**Read the third row carefully.** `ab_e5s` scores higher but is *not* known to
be better: it is a 30k-sample model, and its +0.010 lift is under an open leak
suspicion (§4). Until a leaderboard number confirms it, **run 007 is our real
best and the only thing that should be submitted.**

---

## 1. Every run

| Run | Branch | Sample | OOF F0.5 | India | US | Status |
|---|---|---|---|---|---|---|
| 001 blocking ceiling | `pipeline/entity-resolution` | — | — | — | — | ✅ settled K=30 |
| 002 blocking leak analysis | `pipeline/entity-resolution` | — | — | — | — | ✅ done |
| `smoke_2k_v2` | `nealstuff` | 2,000 | 0.9296 | 0.9018 | 0.9471 | smoke only |
| `003_v2_30k` | `nealstuff` | 30,000 | 0.9510 | 0.9366 | 0.9604 | superseded by 004 |
| `004_v2_30k_fixstop` | `nealstuff` | 30,000 | 0.9512 | 0.9369 | 0.9606 | ✅ **the 30k baseline** |
| **`007_v2_full`** | `main` | 150,000 | **0.9532** | 0.9399 | 0.9620 | ✅ **SUBMITTED → LB 0.943** |
| `008` (scoring only) | `main` | — | — | — | — | ✅ produced submission 001 |
| `ab_laya` | `laya` | 30,000 | 0.9608 | 0.9460 | 0.9704 | ❌ **rejected** (§3) |
| `ab_e5s` | `laya` | 30,000 | 0.9607 | **0.9469** | 0.9697 | ⏳ **scoring test now** |
| ST embeddings | `sentence-transformer-…` | 2,000 | 0.9083 | — | — | ⚠️ inconclusive (§5) |

Compare like with like: `ab_*` are **30k** runs, so their baseline is 004's
0.9512, not 007's 0.9532. Both rerankers add ~+0.0095 **over the 30k baseline**.

## 2. Submissions

| # | Model | OOF | Leaderboard | Status |
|---|---|---|---|---|
| **001** | run 007, 150k, no reranker | 0.9532 | **0.943** (rank 934) | ✅ uploaded |
| 002 | `ab_e5s`, 30k + e5 reranker | 0.9607 | — | ⏳ scoring, ETA ~22:30 |

Budget: 5 uploads/day. Files live in `submissions/<nnn>/`; Krisha uploads.

### What the 0.943 tells us

OOF said 0.9532, the leaderboard said 0.943. That one point is the most
informative number we have. Using actual test shares (France 14.97%, India
46.75%, US 38.27%) and assuming India/US transfer at their OOF values:

```
0.4675(0.9399) + 0.3827(0.9620) + 0.1497 · France = 0.943
                                          France  ≈ 0.904
```

**France is ~6 points below US and is 15% of the grade.** It is the largest
known loss, it is invisible to every offline number we have (no French labels),
and nothing tried so far addresses it.

## 3. Reranker A/B — laya vs e5 (settled)

Neal's hypothesis: India lags because of native-script names, e5 handles them
badly, mmBERT-base should do better. Tested as a pure swap — same band, same
514,335 training pairs, same valid split, same 30k GBDT sample.

| | baseline (004) | laya | e5 |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| US | 0.9606 | 0.9704 | 0.9697 |
| valid AUC | — | 0.99893 | **0.99907** |
| train time | — | 132 min | **23 min** |
| band inference | — | 55 s | **15 s** |

**Verdict: reject laya, keep e5.** laya is worse on India — the one metric the
hypothesis was built to win — and 0.0001 better overall, which is noise, at
5.8× the training cost and a 647 MB checkpoint.

The premise had a flaw worth recording: the incumbent is
`intfloat/multilingual-e5-small`, *already* multilingual. "e5 cannot read
Devanagari" was the motivating assumption and was never true, so the test was
multilingual-vs-multilingual.

## 4. The open question blocking all reranker work

Both rerankers lift the baseline by **the same ~0.0096**, and the earlier e5
run is recorded at "+0.009 UNVERIFIED (record-overlap leak audit pending)".

Three numbers within 0.0006 of each other, across two architectures with
different tokenizers, parameter counts and pretraining, is not what genuine
model-quality differences look like. It is what **one shared confound** looks
like.

Prime suspect, already flagged by Neal and still unaudited: `entities.txt`
guards S1 *entities*, but S2/S3 *records* can repeat between the reranker's
training pairs and the GBDT sample. Both models would exploit it equally.

Note this is **not** the same as the label-leakage check the MLE review did and
passed — that covered stage-2 claims, `split_stats` and folds. Record overlap
in reranker training is a different mechanism and was not in scope.

**Submission 002 settles it.** Held-out test cannot be leaked into: if the
lift is real the LB goes up ~0.010; if it was a leak the LB stays near 0.943.
Until then, no further reranker tuning is worth doing.

## 5. Sentence-transformer branch — inconclusive, and we can now say why

Krisha's branch reports **0.9083 OOF at a 2k sample** with `all-MiniLM-L6-v2`
name/address cosine features, and notes: *"No-embeddings baseline not run yet —
don't know how much of the 0.9083 is from the embeddings vs just the sample."*

**We have that baseline.** `smoke_2k_v2` is the same pipeline at the same 2k
sample without embeddings: **0.9296**.

So at 2k, adding the embedding features scores **0.021 lower** than not adding
them. That is not proof they hurt — the branch may sit on older pipeline code,
and 2k is small enough to be noisy — but it does mean the 0.9083 cannot be read
as progress, and the comparison should be redone at 30k against 004's 0.9512
before investing further. Worth telling Krisha before more time goes in.

## 6. Claims checked, and one that did not survive

| Claim | Verdict |
|---|---|
| France is the main LB loss | ✅ **~0.904**, consistent with 0.943 |
| One record matched to several S1 costs precision | ❌ **measured: 137 of 5,414,100 (0.0025%)** — `assign='soft'` already enforces it; fixing it is worth ~0.00002 |
| We under-predict (3.13 vs 3.46 true links) | ✅ true, but **correct** — F0.5 weights precision 2×, so under-predicting is optimal |
| Blocking recall limits us | ❌ we are at 0.9532 against a **0.9903** ceiling; 3.7 points are in the matcher |
| Tune `TOP_K` with Optuna | ❌ K sets a monotone ceiling, no interior optimum; and the ceiling is not binding |
| Tune "the 46 features" with Optuna | ❌ 46 is how many were written, not a hyperparameter. **LightGBM's own hyperparameters are untuned** — that *is* an Optuna job |

The partition row is the useful one: `PLAN_985` lists it as "unmeasured" and
sized it as a real risk. It is now measured and it is a non-issue.

## 7. Branches

| Branch | PR | State |
|---|---|---|
| `main` | — | trunk; run 007 + submission 001 live here |
| `pipeline/entity-resolution` | #1 | ✅ merged |
| `nealstuff` | #3 | ✅ merged — v2 pipeline, chunked test scoring, memory fixes |
| `laya` | #5 | draft — A/B done, laya rejected, e5 kept |
| `finetune` | #6 | draft — reranker fine-tuning (band pairs, augmentation, LoRA, listwise, distillation). **Gated on §4** |
| `worktree-plan-985` | #4 | open — 0.943 post-mortem and plan toward 0.985 |
| `sentence-transformer-embeddings-feature` | #2 | open — see §5 |

## 8. What is worth doing next

Ordered by expected gain per hour, given ~28 h left:

1. **France diagnostic** — label-free, cheap, never done. It is ~0.904 and 15%
   of the grade. Per-country orphan rate and top-candidate similarity from the
   cached test candidates will say whether France looks like India or like US.
2. **Wait for submission 002** — it answers §4 and decides whether the whole
   reranker line (including `finetune`) is real or wasted effort.
3. **Optuna on LightGBM hyperparameters** — genuinely untuned, unlike the two
   things proposed for it in §6.
4. **Train at 400k+ entities** — 30k→150k bought +0.002 and the curve had not
   flattened.
5. **`Documentation_template.md`** — still unfilled and it is **graded**.

Explicitly not worth doing: tuning `TOP_K`, Optuna over the feature count, or
enforcing the partition.

## 9. Reproducing any of this

```bash
# train + score end to end
python src/run_v2.py --sample 150000

# score test from an existing run, no retraining (caches per-pair scores)
python src/score_test.py --run runs/007_v2_full --chunk 1200000
python src/score_test.py --run runs/ab_e5s --chunk 1200000 --rerank models/rr_e5s

# build the graded package
tools/package_submission.sh
```

Per-pair test scores are cached to `<DATA_DIR>/interim/testp_<key>.npy`, keyed
on model identity, so a changed decision rule costs a load instead of a
117-minute rescore.
