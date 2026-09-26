# PLAN_985 — from LB 0.943 (rank 933) toward 0.985

Written 2026-09-26 10:45 IST. Deadline 2026-09-27 23:59 IST (~37 h). 5 uploads/day.
Sources: runs/007_v2_full, runs/008_submission, context/challenge/*, four ECC reviews (rules,
code map, MLE review, external research). Nothing here is measured on test beyond the one LB number.

## 1. Where we are

```
TSV -> ingest (polars) -> normalize (translit, skeleton, legal form, FR/IN/US tables)
    -> word TF-IDF blocking on core_name+core_addr, per country, top-30   (pair recall 0.95)
    -> 46 pair features + 5 rank features -> LightGBM x5 (GroupKFold by S1)      OOF 0.9500
    -> stage 2: entity shape + claim competition + peers -> LightGBM x5          OOF 0.9526
    -> isotonic calibrator -> assign=soft -> expected-F0.5 per entity (miss=0.1) OOF 0.9532
    -> matching_results.tsv                                                      LB  0.943
```
Train sample 150k of 2.2M S1. Test: 52.0M candidate pairs, 117 min to score on a laptop.

## 2. Why the LB is 0.010 below OOF

| Suspect | Evidence | Size |
|---|---|---|
| **France** (15% of test, 0 labels) | If US/India transfer 1:1, LB 0.943 implies France ~0.898 | up to ~0.009 |
| Country mix (India 47% of test vs 40% train) | reweighted OOF 0.9526 | 0.0006 |
| Partition not enforced (`assign=soft`) | one S2/S3 id can be emitted under several S1; test has 11.5x the S1 of the OOF sample, so contention is far higher than OOF saw | unmeasured |
| Calibrator + `miss` fit on US/India only | applied verbatim to France | inside France row |

No label leakage found in stage-2 claims, split_stats or folds (MLE review).

## 3. Is 0.985 reachable? The arithmetic

Perfect-matcher ceiling at blocking recall R is 1.25R/(0.25+R): R=0.95 -> **0.9896**.
0.985 at today's blocking means a matcher at 99.5% of the ceiling in all three countries
(today: 96.3% on OOF, ~95% on LB). So 0.985 needs BOTH:
- blocking recall 0.95 -> ~0.975 (ceiling ~0.994), via a union of retrievers, not a bigger K;
- matcher errors cut by ~75%, via graph/transitivity + multilingual cross-encoder + France fix.

Foursquare 2022 (closest public analogue) got its largest jumps from exactly these two:
multi-retriever union (max IoU 0.978 -> 0.99+) and graph post-processing (+0.025, 1st place's key stage).
Honest range for 37 h: **0.955-0.970 likely, 0.975+ possible, 0.985 only if the top of the LB
shows it is being done** (see inputs Q1).

## 4. Plan (each phase ends in an upload; ~3 h per full test score)

### Phase 0 — measure, no retraining (T+0 -> T+4 h)          upload #2
1. `score_test.py`: persist `[s1_id, cand_id, country, p1, p2, p_cal]` to parquet (~1 GB) so every
   decision variant after this is minutes, not 117 min. Fix the `sing:.4f` crash.
2. Per-country test diagnostics: singleton rate, links/entity, p-histogram (France vs US vs India).
3. Count S2/S3 ids emitted under >1 S1 in submission 001.
4. Upload `assign=hard` (+ per-country `miss` if France's p-distribution is shifted).

### Phase 1 — recall + graph (T+4 -> T+16 h)                    uploads #3-#4
5. **Record-record graph.** Within country, top-5 neighbours of every S2/S3 record among S2/S3
   (same TF-IDF machinery, ~10M queries on smaller per-country indices). Two uses:
   - stage-2 features: `sib_max_p1` (best p1 of this candidate's near-duplicates for the same S1),
     `sib_count`, S2<->S3 path score p(S1,a)*sim(a,b)*p(S1,b);
   - candidate expansion: siblings of a candidate enter the entity's list (raises recall).
   Duplicates run 5-6 copies per source, so one confident copy should pull in the rest. This is
   where the 3.14 vs 3.46 links/entity shortfall is.
6. **Retriever union** (each k small): + char 3-gram TF-IDF on transliterated core_name (Indic,
   typos), + name-only pass for empty-address records (7.8x over-represented in misses, run 002),
   + multilingual-e5-small kNN (MIT). Gate: keep a retriever only if ceiling rises >= +0.002.
7. Raise training sample 150k -> 400k+ (fold 1 hit MAX_ROUNDS; more S1 = realistic contention).

### Phase 2 — GPU (in parallel on the RTX 3050, T+4 -> T+20 h)   upload #5
8. multilingual-e5-small bi-encoder: embed ~10M records once (~1 h), add name/addr cosine as
   features for all 52M pairs (a dot product per pair, cheap). Covers Indic and French.
9. Leak-free cross-encoder reranker (exclude every GBDT-sample cand_id from its training
   pairs), scoring only the uncertain band. Keep only if its 30k gain survives the leak fix.

### Phase 3 — France (T+16 -> T+28 h)                           uploads #6-#7
10. Pseudo-labels: French pairs that are mutual-best, p > 0.97, and agreed by US-only and
    India-only models, as positives; clear rejects as negatives. Retrain with them; check the
    French p-distribution moves toward US/India.
11. Final: best config, full mlguard PASS, validate_submission, both TSVs, code zip. Stop
    experimenting at T+32 h (27 Sep ~19:00 IST).

## 5. Gates (no step ships without these)
- Every change measured on OOF (per country) before it touches test; every upload logged in
  docs/EXPERIMENTS.md with its delta.
- Blocking/normalize changes purge the candidate cache (key has no normalizer version).
- One heavy python job per box (run 005 OOM).
