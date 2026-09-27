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

---

## 007 — first full-sample run (150k) and first test phase ever run
2026-09-26 · Priyanshu (Claude) · branch `nealstuff` + Windows fixes
Question: does v2 hold at the full training sample, and what does the test phase actually cost?
Setup:    `PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh 007_v2_full --sample 150000`
          (no `--train-only`), laptop 10 cores / 23.7 GB, **CPU only — the RTX 3050 was absent from
          the PCI bus this boot (Code 45), so no reranker.** GroupKFold(5) by S1, K=30.

| Layer (OOF macro F_0.5) | 30k (004) | **150k (007)** |
|---|---:|---:|
| stage 1: v2 features | 0.9490 | **0.9500** |
| stage 2: + competition / peer context | 0.9508 | **0.9526** |
| decision layer | 0.9512 | **0.9532** |

Per country: India 0.9399, US 0.9620 (30k: 0.9369 / 0.9606). Blocking recall 0.9498 on 4,499,993
pairs from 150,000 entities, 492,739 positives. Predicted singletons 6.28% vs 5.63% true;
3.138 links/entity against a true average of 3.46.

Blocking recall is flat across sample size — 0.9506 at 2k, 0.9496 at 30k, 0.9498 at 150k — so the
~0.95 ceiling is a property of the blocking design, not a small-sample artifact.

### Test phase timings — the estimate this project was planned around is wrong

First time the test phase has been run on any machine. Blocking, 1,732,544 S1 queries:

| Partition | Queries | Index | Rate | Wall |
|---|---:|---:|---:|---:|
| France | 259,452 | 1,434,993 | ~1,010 q/s | 4.1 min |
| India | 809,986 | 4,717,565 | 397 q/s | 34.0 min |
| US | 663,106 | 3,817,031 | ~800 q/s | 13.8 min |
| | | | **total** | **51.9 min** (+~5 min index builds) |

`EXPLAINER.md` §"Full run on the laptop" budgets **255 min** for this stage, 62% of a 414-min run.
Measured: **~57 min**, a 4.4x speedup — more than the 2.7x `sparse_dot_topn` was credited with,
because the gain grows as the per-country index shrinks. Throughput tracks index SIZE, not query
count: France's 1.4M-record index runs 2.5x the rate of India's 4.7M.

That kills the case for renting hardware (§"Where to run it"). The laptop is the right box.

Also: test US index is 3.82M records vs train's 6.19M, while test India is 4.72M vs train's 4.13M —
the documented country shift, visible in the index sizes.

Cached and worth sharing (`./aws/s3.sh share-cache`): `cands_test_k30_df0.01_mdf3_ctry1_nall.parquet`,
**51,974,499 pairs / 758 MB**, plus `stats_test_v1.pkl` (153 MB, 63 s).

### Two conclusions from 003-005 that do not survive at full sample

- **"Every decision mode is within 0.0005 ... assignment adds ~nothing"** — not at 150k. The best mode
  is `assign='soft'` + `select='expected_f'` with `miss=0.1`, not the global threshold that won at 30k.
  The gain is small (+0.0006 over stage 2) but the *choice* is sample-size dependent, so the decision
  layer is doing real work where it looked inert. Do not delete it on the 30k evidence.

  Reading the full 288-row `decision_table.csv` back, the two axes separate cleanly and neither is
  large. `select` is where the gain lives: best `expected_f` 0.953182 vs best `threshold` 0.952573,
  **+0.0006**. `assign` is noise: within `expected_f`, `soft` beats `none` by 0.00007 — seven
  ten-thousandths, on 150k entities. So "keep the decision layer" is right, but the honest version is
  *keep `expected_f`, and stop tuning `assign`*. Neither clears Neal's own 0.003 bar for added
  complexity; both are kept because they are already written and cost nothing at inference. The
  remaining headroom is 0.9532 against a blocking ceiling of 0.9903 — **3.7 points, all of it in the
  matcher**, and concentrated in India. That is where the next hour goes, not here.
- **`loss_ratio` still fires.** 004 recorded it fixed to need a stalled valid loss; at 150k it tripped
  anyway — `fold 20 iter 210: valid/train loss 1.53 > 1.5` — stopping stage-2 fold 0 and rolling back
  to its best iteration. The guard behaved correctly; the threshold in `mlguard.toml` is tuned on 30k
  and wants revisiting before it silently truncates full-sample folds.

Train/valid gaps shrink with sample, as expected: stage 1 0.004-0.017 and stage 2 0.001-0.003 at 150k,
against 0.007-0.014 at 30k and 0.043 at 2k (which mlguard correctly FAILed on `overfit_gap`).
Stage 1 fold 1 hit `best_iter 2000` = `MAX_ROUNDS` without early-stopping, so that fold was still
improving when the cap cut it off — worth raising MAX_ROUNDS for full-sample runs.

### No submission yet: the run hung in the test phase

After blocking finished, the run deadlocked — 0% CPU, no children, 10.2 GB resident, silent.
`split_stats` called `Pool(workers)` unconditionally (unlike `record_table` beside it, which guards on
`workers > 1`). On the test split it runs with the 52M-pair frame resident, so the parent was at ~10 GB
when it spawned; a worker died, the pool could not replace it (`PermissionError: [WinError 5]` from
`DuplicateHandle`), and `pool.map` waited forever. It stalled rather than raised, which is the worse
failure: 57 minutes of finished blocking sat on disk while the process held 10 GB doing nothing.
Fixed in `c701d01`. Everything expensive was already cached, so the retry skips all blocking.

**A second wall sits behind that one**, and the retry would have hit it. The test phase builds one
feature frame for all 51,974,499 pairs at once: 46 float64 columns over 52M rows is ~19 GB, and
stage 2's `concat` doubles it, against 23.7 GB total. `predict_test_chunked` (`9aad360`) streams it in
entity-aligned chunks instead — two passes, because `claim_rank` / `claim_gap` / `n_claims` /
`n_strong_claims` group by `cand_id` and one candidate can be claimed from different chunks, so stage 2
is built once globally while only the wide stage-1 matrix is chunked. Cuts land only where `s1_id`
changes, so every per-entity aggregate matches the unchunked result exactly. Peak drops from ~19 GB to
roughly 1.5 GB per chunk. `AMLC_TEST_CHUNK` overrides the 4M default.

Full narrative for this run, written for someone picking it up cold:
[`runs/007_v2_full/CONTEXT.md`](../runs/007_v2_full/CONTEXT.md).

### Windows portability — the branch could not start at all before this

`nealstuff` had never run on a Windows box. Four fixes (`6a91330`, `8264b83`):
- Pools sized `os.cpu_count() - 1` = 15 workers. Free under fork, fatal under spawn (each worker
  re-imports pandas/polars/numpy/scipy, 250-400 MB). Died in MemoryError during pool startup and
  orphaned the workers. Now `config.WORKERS`, capped at 4 on spawn, `AMLC_WORKERS` to override.
- `_pick_data_dir` tested the drive letter, not the folder, so a mounted-but-empty `D:\amlc_data`
  shadowed the real dataset and every path silently pointed at nothing.
- `train_guarded.sh` ran a bare `python` — here the system 3.12 with pandas 3.0.3, the major version
  requirements.txt pins against, and missing sparse_dot_topn and anyascii. It now prefers the venv and
  prints which interpreter it chose.
- That script was stored CRLF and `core.autocrlf=true` restores it, so bash choked on the `\r`.
  `.gitattributes` pins `*.sh` to LF.

New deps needed installing: polars, anyascii, sparse_dot_topn.


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

## 008 — first leaderboard submission (lb-20260926-1)

Scored the full test set from run 007's model with `src/score_test.py` (no retraining), and uploaded.

```
offline OOF macro F0.5   0.9532
LEADERBOARD              0.943     rank 934, leader ~0.988
```

**About one point of optimism, and France is the leading suspect.** It is 15% of the test set with
zero training labels, so no OOF number covers it; a weak 15% slice costs roughly this much. The
alternative explanations are weaker: the output matched OOF closely on every shape statistic
(singleton rate 6.25% vs 6.28%, links/entity 3.125 vs 3.138), which rules out a gross
train/inference mismatch, and blocking recall is measured on train only, so a France blocking hole
would show up here too.

**This is measurable without labels** and has not been done: per-country orphan rate and
top-candidate similarity distribution from the cached test candidates. If France's distribution
looks like India's rather than the US's, we know where the point went. Doing that before tuning
anything else avoids optimizing the 85% we can already see.

The gap to the leader is 4.5 points, and our own blocking ceiling is 0.9903 — so the headroom is
real and in the matcher, not in blocking.

Scoring cost 117 min for 51,974,499 pairs (52 chunks, two passes); test blocking was cached from
run 007, saving 52 min. Artifacts in `submissions/001/`.

## 009 — laya vs e5 band reranker, head to head (branch `laya`)

Neal's hypothesis: the India gap is native-script names, e5-small handles them poorly, and mmBERT-base
(322M, 100+ languages) should do better. Tested as a drop-in swap -- same band, same training pairs,
same `entities.txt` guard, same 30k GBDT sample. Only the model changes.

Both rerankers trained here on identical data (514,335 pairs, same entity-grouped valid split, both
excluding run 007's folds). `models/rr_e5s` did not exist on this machine, so it was trained too --
without it there is no A/B, only a laya number with nothing to compare against.

### Intrinsic (reranker's own valid split)

| | rr_laya | rr_e5s |
|---|---|---|
| valid logloss | 0.03636 | **0.03339** |
| valid AUC | 0.99893 | **0.99907** |
| trainable params | 55.1M | ~22M |
| train time | 132 min | **23 min** |

### Downstream (30k pipeline, the number that matters)

| | baseline (004) | laya | e5 |
|---|---|---|---|
| stage 1 | 0.9490 | 0.9490 | 0.9490 |
| stage 2 | 0.9508 | 0.9608 | 0.9607 |
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| US | 0.9606 | 0.9704 | 0.9697 |
| band inference | -- | 55 s | **15 s** |

### Verdict: reject laya, keep e5

laya is **0.0009 worse on India** -- the single metric the hypothesis was built to win -- and 0.0001
better overall, which is noise. It costs 5.8x the training time, 3.7x the inference, 2.5x the
trainable parameters and a 647 MB checkpoint. There is no axis on which it wins.

The premise had a flaw worth recording: the incumbent is `intfloat/multilingual-e5-small`, which is
*already* multilingual. "e5 cannot read Devanagari" was the motivating assumption and it was never
true, so the experiment was testing multilingual-vs-multilingual, not multilingual-vs-English.

### The finding that matters more than the A/B

Both rerankers lift the 30k baseline by the same ~0.0096, and AGENTS.md records the earlier e5
reranker at "+0.009 UNVERIFIED (record-overlap leak audit pending)". Three numbers agreeing to
within 0.0006 across two architectures with different tokenizers, parameter counts and pretraining
is not what genuine model-quality differences look like. It is what a **shared confound** looks
like.

The obvious candidate is the leak Neal already flagged: `entities.txt` guards S1 entities, but S2/S3
*records* can repeat between the reranker's training pairs and the GBDT sample. Both rerankers would
exploit that equally, which is exactly the pattern observed.

**So the leak audit is now the critical path, not reranker selection.** If the +0.010 is a leak it is
fake for both models and must not reach the leaderboard; if it is real it is our largest single gain
and should go in immediately. Nothing else about the reranker is worth tuning until that is settled.
Concretely: measure the S2/S3 record overlap between the reranker's training pairs and the 30k GBDT
sample, then retrain with those records excluded and see whether the gain survives.

## 011 — v4: claim features over the full frame (the contention fix)

Stage 2's competition features are raw counts over whatever Source-1 set is present, and we train
with 150k entities while inferring with 1,732,544. Measured mean n_claims: train 150k **1.957**,
test **5.549** — a 2.8x shift in the second most important feature in stage 2 (`n_strong_claims`,
importance 2.52M behind only p1's 14.35M).

`run_v4.py` computes those counts over the full 2,206,808-entity train frame (66,204,198 pairs)
while still training the GBDT on 150k. Measured on the run: contention went to **6.717**, which
brackets test's 5.549 instead of sitting far below it.

```
                        003 (150k+rr)    011 (+contention)
stage 1                    0.9490            0.9494
stage 2                    0.9626            0.9644
India                      0.9491            0.9514
US                         0.9717            0.9730
```

**+0.0018 OOF, +0.0023 on India.** That was not the predicted outcome. The fix aligns training with
test contention, and OOF is *measured* in the low-contention regime, so the expectation was flat or
slightly worse OOF with the benefit visible only on the leaderboard. Getting a gain anyway means the
full-frame claim counts are genuinely more informative, not merely better matched to test — the
sample-only counts were not just mis-scaled, they were noisy.

**The decision layer changed its mind.** 003 chose `assign=soft, select=expected_f, miss=0.1`; 011
chose a plain global `threshold` at 0.71. With honest claim features a single cut now beats
per-entity expected-F, which suggests the expected-F machinery had been partly compensating for
miscalibrated competition counts rather than adding decision-theoretic value of its own.

Cost on the laptop: 4 h. Blocking all 2.2M train entities was 63 min, stage-1 inference over the
full 66M-pair frame 83 min, the rest training.

One logging defect to fix: the line `claims: full-frame mean n_claims 145.517 vs sample-only 10.797
(test is 5.549)` compares a row-weighted mean (Sum n^2 / Sum n, dominated by popular candidates)
against a plain ratio. Apples to oranges — the like-for-like figures are 6.717 vs 5.549. The
features themselves are correct.
