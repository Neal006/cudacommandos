# Experiment Plan

Ordered by expected value per hour. Each result goes in `docs/EXPERIMENTS.md` (the log)
using its template, and every training experiment runs under `train_guarded.sh`.
**Primary metric:** OOF macro F0.5 on train (GroupKFold by S1). **Always also report:** per
country, LOCO, predicted singleton rate, links/entity, runtime, peak RAM.

**Noise floor.** Two runs differ only if |Δ| > max(0.002, 2·fold_std/√5). Below that,
keep the simpler one.

## A. Candidate generation

| id | Question | Setup | Success | Kill / fallback | P |
|---|---|---|---|---|---|
| E01 | Where does test blocking time go, and can it be parallelized? | Profile pass A on 20k India queries: vectorize / matmul / top-k split; ProcessPool 4/8/12 workers | ≥3× throughput at ≤ 20 GB RAM | If RAM-bound, run on AWS | P0 |
| E02 | Recall@K per country (not global) | Existing cache, K ∈ {5,10,20,30,50}, split US/India | Numbers per country | — | P0 |
| ~~E03~~ **DONE** | Where are the misses? | Missed true pairs by slice: native-script name, domain name, empty address, junk prefix, no shared token after pruning | A slice holding > 40% of misses | — | ✅ run 002 |
| E04 | Pass B (address) marginal recall | Add pass B K∈{10,15,20}; union recall and candidates/entity | +2 pts recall at ≤ +15 cands | < +0.5 pt → drop | P0 |
| E05 | Pass C (translit/skeleton) + pass D (reverse) | Add each alone, then both | Recall ≥ 0.985 total at ≤ 60 cands | — | P1 |
| E06 | Does the training sample need neighbours? (Weakest hypothesis #1) | Random 300k vs geo-cluster 300k; evaluate both on a fully blocked held-out set of localities | Geo ≥ random + 0.003 on the full slice | Equal → keep random (simpler) | P1 |

## B. Decision layer (no retraining, on OOF scores)

| id | Question | Setup | Success | Kill | P |
|---|---|---|---|---|---|
| E07 | Does the partition constraint help? | Global threshold vs + hard assignment vs + soft | ≥ +0.003 | < +0.001 | P0 |
| E08 | Expected-F0.5 per entity vs global threshold (Weakest hypothesis #2) | Isotonic-calibrated p; grid `miss` ∈ {0, 0.05, 0.1, 0.2}, `p_min` | ≥ +0.003 over E07's best | Worse on singletons → keep threshold | P0 |

## C. Features and model

| id | Question | Setup | Success | Kill | P |
|---|---|---|---|---|---|
| E09 | Stage-2 (competition + peer) | Stage 2 on OOF p1 with the same folds | ≥ +0.003, and gains on the "same name + locality" slice | No slice gain → drop | P1 |
| E10 | Transfer to unseen countries (France proxy) | LOCO: train US → test India, train India → test US; compare feature sets | LOCO drop ≤ 0.03 | Drop > 0.05 → strip country-specific features | P0 |
| E11 | French normalizer correctness | Unit tests on 50 hand-checked test-France S1/S2 look-alike pairs (no labels, rule checks only) | 100% pass | — | P0 |
| E12 | Feature-group ablation | Drop one group at a time: translit, skeleton, IDF, numbers, blocking-pass, genericity, colocation | Ranked gain table for the methodology document | — | P1 |
| E13 | Train/test feature drift | KS statistic per feature, train candidates vs test candidates per country | Top features KS < 0.1 | KS > 0.2 → find the cause before uploading | P1 |
| E14 | Cross-encoder on the uncertain band | Fine-tune `microsoft/mdeberta-v3-base` (MIT) or `intfloat/multilingual-e5-small` (MIT) on 1M pairs; score only 0.2 < p < 0.8; blend | ≥ +0.003, fits the time budget | Else skip | P2 |
| E15 | Seeds/folds ensemble | 3 seeds × 5 folds mean | fold std ↓, OOF ≥ single | — | P2 |
| E16 | GBDT alternatives | XGBoost / CatBoost on the same features, and a rank average | ≥ +0.002 | — | P2 |
| E17 | Train size | 150k vs 300k vs 600k entities | Curve flat → keep smaller | — | P2 |
| E18 | Fellegi–Sunter EM as features / France fallback | Unsupervised EM weights on the test France candidates (label-free) | France cardinality closer to OOF; LOCO ↑ | — | P3 |

## D. Leaderboard protocol

- An upload is only an experiment when its OOF result is already known. The LB confirms,
  and never selects hyperparameters (it's a public subset → overfitting).
- Record `OOF → LB` for every upload. A widening gap is a drift signal: check mlguard
  drift, especially France.
- Five a day: 1 champion re-confirm (only if the pipeline changed), up to 3 ranked
  challengers, 1 reserve.

## E. Tiebreak rules
1. Better OOF by more than the noise floor wins.
2. Otherwise: better worst-country score wins.
3. Otherwise: better LOCO wins (France proxy).
4. Otherwise: faster / simpler wins.
