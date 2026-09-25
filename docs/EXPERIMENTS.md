# Experiment log

One entry per run that produced a number. Append, never rewrite — the
methodology document (`Documentation_template.md`) is graded and the top 100
teams have their pipelines audited, so the trail of what was tried and why is
worth as much as the final score.

**Log every leaderboard submission here too.** Five per day, ~15 total. Without
a record of what changed between them, a score that moves tells you nothing.

Template:

```
## NNN — short title
date · who · commit
Question: what this run was supposed to answer
Setup:    the parameters that differ from the previous run
Result:   the numbers
Read:     what it means and what changes next
```

---

## 001 — Blocking recall ceiling
2026-09-25 · Priyanshu · `7b4079f`

**Question.** Candidate generation caps everything downstream: a true pair that
never becomes a candidate cannot be recovered by any matcher. What fraction of
true pairs does the current blocking reach, and at what K?

**Setup.**
```
command        src/run_pipeline.py --blocking-only
S1 sample      150,000 of 2,206,821   (C.TRAIN_SAMPLE)
S2 / S3 index  5,034,616 / 5,285,603  (complete, not sampled)
blocking       word TF-IDF, per-country partitions, chunked sparse matmul
max_df         0.01      min_df 3
vocab fit      1,000,000 of 10,320,219 documents sampled
vocabulary     126,023 tokens
TOP_K          50 (to measure the curve; lowered afterwards — see Read)
wall clock     ~28 min
```

**Result.**

| K | pair recall | F_0.5 ceiling |
|---:|---:|---:|
| 5 | 0.8340 | 0.962 |
| 10 | 0.9187 | 0.983 |
| 20 | 0.9411 | 0.988 |
| 30 | 0.9499 | 0.990 |
| 50 | 0.9586 | 0.992 |

Orphans (entities with no candidates): **0**.

Throughput against the full index:

| Partition | Queries | Index | Rate | Time |
|---|---:|---:|---:|---:|
| India | 59,869 | 4,133,346 | 75 q/s | 11.1 min |
| US | 90,131 | 6,186,873 | 150 q/s | 9.6 min |

**Read.**

Recall is still climbing at K=50, and the obvious conclusion — raise K — is
wrong. Recall is not what is scored. F_0.5 weights precision 2×, so a perfect
matcher limited only by blocking recall R scores `1.25R / (0.25 + R)`. That
collapses the gap:

```
K=30   recall 0.9499  ->  F_0.5 ceiling 0.990
K=50   recall 0.9586  ->  F_0.5 ceiling 0.992
```

**+0.002 of ceiling for 35 million extra pairs to featurize on test.** Not a
trade worth making. `TOP_K` set to 30.

The conclusion that should drive the next few hours:

> Blocking is not the bottleneck. Matcher precision is.
> Zero orphans and a 0.990 ceiling means candidate generation is done for now.
> Spend the time on features, the decision threshold, and per-country
> validation.

India runs 2× slower than US on a *smaller* index. Indian names and addresses
share more tokens, so posting lists are denser and each query touches more of
the matrix. This sets the test-blocking budget: 1.73M queries (~810k India,
~663k US, ~260k France) is **roughly 4–5 hours**, which is why candidates are
now cached to parquet rather than recomputed per experiment.

**Next.** Train the matcher and get an honest OOF F_0.5. Until that number
exists, nothing about feature quality is known.

---

## 002 — Where blocking recall leaks (country × script)
2026-09-25 · Priyanshu · `2cd608a`

**Question.** Run 001's ceiling came from a random 150k sample, which inherits
train's 60/40 US/India mix. Test is 38/47/15 US/India/France. Neal's
weakest-hypothesis audit (Master-Plan, E03) argues that if the misses
concentrate in India — or in non-Latin-script records, which the normalizer
folds accents on but does not transliterate — the headline is optimistic.

**Setup.** `src/analyze_blocking.py --per-country 15000`. Equal sample per
country so neither can hide inside the other; full 10.3M index; same blocking
parameters as run 001. ~8.5 min.

**Result.**

| K | India | US | ALL |
|---:|---:|---:|---:|
| 10 | 0.8720 | 0.9511 | 0.9116 |
| 20 | 0.9007 | 0.9686 | 0.9347 |
| 30 | 0.9132 | 0.9735 | 0.9434 |

5,866 missed of 103,567 true pairs. Missed targets vs a covered baseline:

| Property | missed | covered | enrichment |
|---|---:|---:|---:|
| **empty address** | **24.9%** | **3.2%** | **7.8×** |
| kannada | 3.1% | 0.6% | 5.2× |
| devanagari | 16.5% | 4.3% | 3.8× |
| all Indic combined | 26.8% | 7.5% | 3.6× |
| domain-style name | 10.9% | 4.5% | 2.4× |
| latin | 60.7% | 87.5% | 0.69× |

**Read.**

**E03 confirmed: India lags US by 6pp at K=30** (0.9132 vs 0.9735), below the
0.93 line that was supposed to make transliteration P0.

**But the test-set impact is small.** Blocking is unsupervised — the vocabulary
is fit on the corpus, not learned from labels — so France's absence from
training does not hurt it, and French names are Latin with accents the
normalizer already folds. France should track US. Weighting by the real test
mix: `0.383(0.9735) + 0.468(0.9132) + 0.150(~0.97) ≈ 0.946`, against 0.9499
measured in run 001. **F_0.5 ceiling 0.990 → 0.989. TOP_K=30 stands.**

**The unpredicted finding: empty address, not script, is the strongest single
signal** (7.8× vs 3.6× for all Indic combined). That is a design bug, not a
data property — `_blob` is `core_name + " " + core_address`, so a record
without an address contributes only its name, carries fewer rare tokens, and
loses top-K slots to records matching on address noise. Roughly 3% of S2/S3
records have a blank address; they are a quarter of all misses.

Keep it in proportion: latin is still **60.7% of misses in absolute terms**.
The majority of missed pairs are ordinary records that did not rank top-30 —
hard pairs, not a category failure.

**Ceiling arithmetic for the candidate fixes**, if each were perfect and
non-overlapping: empty-address +1.41pp recall, Indic +1.52pp, domain +0.62pp.
Fixing all three lands around 0.97 recall → F_0.5 ceiling 0.9939, against
0.9887 test-weighted today: **+0.005 over doing nothing.** That is the honest
size of the prize.

Per-slice ceilings, for when the matcher is measured by country:

```
India  recall 0.9132 -> F_0.5 ceiling 0.9813
US     recall 0.9735 -> F_0.5 ceiling 0.9946
```

**Next.** The ceiling says blocking is still not the bottleneck, and no
blocking fix can be worth more than ~0.005 while the matcher is unmeasured.
Train it and get a real OOF F_0.5 first; revisit these three fixes only if the
per-country breakdown of the *matcher* shows India dragging.

---

## 003–005 — v2 pipeline on a 30k-entity sample (OOF only)
2026-09-25/26 · Neal (Claude) · branch `nealstuff`
Question: how much does each v2 layer add over the old feature set, all else equal?
Setup:    `PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh <id> --sample 30000 --train-only`
          same cached candidates (K=30, recall 0.9496), GroupKFold(5) by S1, laptop RTX 3050 4 GB.

| Layer (OOF macro F_0.5) | 003 | 004 (stop fix) | 005 (+reranker) |
|---|---:|---:|---:|
| old 27 features, global threshold (`--ablate`) | 0.9266 | – | – |
| stage 1: v2 features (+translit, skeleton, house no., legal form, domain, label-free S1 stats) | 0.9491 | 0.9490 | 0.9490 |
| stage 2: + competition / peer context | 0.9507 | 0.9508 | **0.9600** ⚠ |
| decision layer (calibrate + assign + expected-F) | 0.9510 | 0.9512 | crashed (OOM, see below) |

Per country (004): India 0.9369, US 0.9606. Predicted singletons 6.5% vs 5.8% true; 3.13 links/entity.
Global threshold lands at 0.68. Every decision mode is within 0.0005 — stage 2's claim features already
encode the partition, so assignment adds ~nothing. Stage-2 gain: p1 72%, n_strong_claims 14%, p1_rank 6%.

Reranker (`src/gpu/reranker.py`, e5-small, frozen word embeddings, bf16):
- smoke (24k/6k entity split of the same sample): held-out AUC 0.9989, AP 0.9909 vs block_sim AUC 0.9566;
  train 390 pairs/s, inference 1,870 pairs/s.
- run 005: trained on 20k entities **outside** the sample (257k pairs, 11 min, valid AUC 0.9987); the
  0.2–0.8 band is only **1.2% of pairs** (10.5k scored in 23 s) and lifted stage 2 by +0.009.

Read:
- v2 features are the big win (+0.0225). Stage 2 +0.0017, decision layer +0.0004.
- ⚠ **Reranker +0.009 is not yet trusted.** S1 entities are disjoint, but an S2/S3 *record* can be in both
  the reranker's training pairs (e.g. as a hard negative of another entity) and the GBDT sample's candidates
  — memorised records do not exist at test time. Next: exclude every `cand_id` of the GBDT sample from
  `make_training_pairs`, retrain, rerun 005. Keep the reranker only if the gain survives.
- Run 005 died with MemoryError in the decision layer because an ad-hoc audit script was started next to it
  (run 004 passed the same step). Re-run alone.
- mlguard fixes found by these runs: `loss_ratio` false positive (now needs a stalled valid loss), a guard
  stop now rolls LightGBM back to the best valid iteration (it used the stop iteration), NaN in
  summary.json → null (strict JSON).
- Train/valid F@0.5 gap: stage 1 ≈ 0.03 (train F is in-sample), stage 2 ≈ 0.007–0.014; mlguard run PASS.

Open questions worth an entry each:

- **Per-country F_0.5.** France is 15% of test with zero training rows and the
  US/India ratio inverts between splits (`DATA_BRIEF.md` §4). The aggregate
  score will hide a weak French slice. Use `metrics.scores_breakdown()` split
  by country.
- **Threshold behaviour.** Singletons are only 5.6%, so the precision-heavy
  metric should *not* push the threshold as high as intuition suggests. Check
  where the sweep actually lands and what the predicted singleton rate is —
  if it is far from ~6%, something is off.
- **Feature ablation.** Which of the ~30 pair features carry the gain? Cheap
  to check from LightGBM importances, and it is exactly what the methodology
  document asks for.
