# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** CUDA_COMMANDOS
**Team Members:** Priyanshu Doshi, Neal Daftary, Krisha, Krina
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A CPU pipeline of sparse TF-IDF blocking, a LightGBM prune model, a pairwise
LightGBM (stage 1) and a context-aware LightGBM (stage 2), followed by a
decision layer that enforces the one-to-one structure of the ground truth and
chooses, per Source-1 entity, the subset of candidates that maximises expected
F0.5. An optional multilingual cross-encoder adds scores on the ambiguous band.

**Leaderboard score: 0.985.** Measured out-of-fold on held-out train S1
beforehand: **0.98654**.

**Compute.** Everything is CPU work apart from the optional cross-encoder. The
full-data stages ran on a single AWS `c7i.48xlarge` (192 vCPU, 384 GB, Ubuntu
22.04, Python 3.11) with `ER_JOBS=64`; a Windows laptop handled packaging,
validation and the output checks. Peak memory is about 30 GB, so a 48 GB box is
enough — the large instance buys wall-clock, not headroom. `io → prune` takes 86
minutes there and the whole pipeline about 3 hours. No managed or third-party
service is used at any point: the only inputs are the provided TSVs.

---

## 2. Methodology

### 2.1 Problem Analysis

- F0.5 weights precision twice as heavily as recall, and singletons count: an S1
  with no true match scores 1 only if we predict nothing. The decision of *how
  many* candidates to output per S1 is therefore as important as the scoring.
- Ground truth is a partition: an S2/S3 record belongs to at most one S1. How
  strongly *other* S1 entities claim a record is signal.
- Names carry junk prefixes, aliases, domains ("Name | www.x.com"), leet-speak
  and legal suffixes; addresses mix state/region, locality, postcode, house
  number, unit and street in free text.
- A large share of Indian records are in native scripts, so a transliteration
  step is required before any string comparison.
- France has no training labels, so every country-specific decision is checked
  against test-side statistics (pairs per S1 by country) rather than trusted.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — blocking + pruning + two-stage GBDT + decision-theoretic output.

**Core Innovation:** Stage 2 scores each pair in context — its rank and gap within
the S1, competition from other S1 entities for the same record, agreement with
the S1's other top candidates (S2 <-> S3), and "twin" detection (same name,
different house number) — and the output layer picks each S1's subset by
Monte-Carlo expected F0.5 after a one-to-one assignment.

```
TSV -> parquet (io_utils) -> splits (20% hidden S1 + 5 folds)
    -> transliteration + name/address normalisation
    -> sparse TF-IDF blocking (name / name+address / address, reverse pass, exact keys)
    -> prune model -> top-20 per S1 (= candidate_pairs.tsv)
    -> stage-1 LightGBM (~75 pair features)
    -> [optional cross-encoder on the 0.01 < p1 < 0.999 band]
    -> stage-2 LightGBM (context features) -> isotonic calibration
    -> one-to-one assignment + per-S1 expected-F0.5 subset -> matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** within each (country, state) block, sparse top-k
  retrieval on (a) character 3-gram TF-IDF of the name (k=30), (b) character
  3-gram TF-IDF of name + street + locality (k=30), (c) word TF-IDF of address
  tokens and numbers (k=30); a reverse pass from each S2/S3 record to its top S1
  (k=5, k=20 for records without a state/address); exact-key blocks capped at
  300 records. Union of all blockers, keeping each blocker's score and rank.
- **Candidate pairs generated:** the pruned set, at most top-20 per S1 plus the
  per-source and best-S1 keeps below; written to `output/candidate_pairs.tsv`.
  The union holds 101.9 candidates per S1 (recall 0.99069 on Q); pruning brings
  that to **7.72 per S1** (13.37M test pairs) for 0.21% of true pairs lost in
  each of India and US. Pairs per S1 by country: France 10.25, India 7.80,
  US 6.63 — France, which has no training labels, is not pruned harder than the
  countries that do.
- **How you ensured true matches were not lost:** a dedicated prune model scores
  the full union, and a pair is kept if it is in its S1's top K=20 with score
  >= 0.005, **or** in the top 10 from S2 or from S3 separately, **or** it is the
  best S1 for its S2/S3 record. `prune.py` reports recall on the query set before
  and after pruning, lists lost true pairs by reason and segment, and writes a
  K x min-p recall grid (`work/prune_grid.csv`) used to choose the settings.

---

## 4. Matching Model

**Features used:**
- Name features: set overlaps on tokens and character n-grams (numba), rapidfuzz
  ratios, name frequency, alias/domain/legal-suffix flags, transliterated forms.
- Address features: parsed components — state/region, locality, postcode, house
  number (match and distance), unit, street — and their overlaps.
- Other: blocker scores and ranks; stage-2 context features (within-S1 rank/gap/
  share, expected cluster size, same-source counts, competition from other S1s,
  agreement with other top candidates, twin flag, optional cross-encoder logit
  with rank/gap/margin).

**Model type:** LightGBM (prune model, stage 1, stage 2; GroupKFold, 5 folds),
isotonic calibration; optional cross-encoder `intfloat/multilingual-e5-small`.
**Threshold selection method:** no global threshold. Probabilities are adjusted
`p' = sigmoid(a * logit(p) + b)` with (a, b) tuned on out-of-fold predictions
(optionally per country), then a one-to-one assignment and a per-S1 subset that
maximises expected F0.5 (256 Monte-Carlo samples, up to 10 candidates per S1),
including the empty set.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.985 on the leaderboard.** Measured out-of-fold at
  0.98654 on 1.77M held-out train S1 with 20% of S1 hidden as orphans
  (US 0.98731, India 0.98537, singletons 0.98733) — an offline-to-leaderboard
  gap of 0.0015, which is the main evidence that the validation design holds up:
  hiding a fifth of S1 makes the held-out set carry the same share of
  businesses-with-no-match the test set has, and that is the case macro F0.5
  punishes hardest. The test output passes `validate_submission.py --check-ids`.
- **Recall ceiling.** The blocking union holds 0.99069 of the true pairs at 101.9
  candidates per S1; pruning takes that to 0.98862 at 7.72. Retrieval is therefore
  not the binding constraint — the distance between 0.98862 and the final score is
  the matcher's, and Appendix B.2 shows buying more recall costs more than it
  returns under F0.5.
- **Common false positives (wrong merges):** same-name businesses in the same
  locality with different house numbers or units (branches, chains). This is what
  the twin flag and the house-number *distance* feature exist for: 12 vs 14 is
  weaker evidence against a match than 12 vs 890.
- **Common false negatives (missed matches):** records whose name differs by
  script, alias or rebranding and whose address is sparse or missing. India is the
  harder country out-of-fold (0.98537 against US 0.98731), and native-script names
  are why the transliteration dictionary is learnt from the training pairs rather
  than taken from a rule table.
- **Singletons.** 5.6% of entities have no true match, and predicting anything for
  them scores 0 instead of 1. Out-of-fold the singleton segment scores 0.98733 —
  in line with the overall number, which is the evidence that the expected-F0.5
  decoder is choosing the empty set when it should and not merely often.

---

## 6. Conclusion

Treating the output as a decision problem — calibrated probabilities, a
one-to-one constraint and per-entity expected-F0.5 subsets — mattered as much as
the scorer itself. Context features that encode competition between entities
were the main gain of stage 2 over stage 1.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md` for the full run guide):

- `src/run_pipeline.py` — entry point; runs stages
  `io, splits, normalize, sets, blocking, prune, stage1, context, output`
  and can resume with `--from <stage>`.
- `src/write_output.py` — writes `output/matching_results.tsv` and
  `output/candidate_pairs.tsv` and runs the official validator.
- `src/crossencoder.py`, `gpu_steps.sh` — optional GPU cross-encoder round trip.
- `requirements.txt` — pinned dependencies.

Reproduce: place the data at `dataset/{train,test}/`, then
`cd code/business_entity_resolution/src && ER_JOBS=64 python run_pipeline.py`.
`ER_JOBS` also fixes LightGBM's thread count, so pinning it is what makes two
runs line up; `README.md` covers this under "Determinism".

### B. Additional Results

#### B.1 Recall through the pipeline

Every true pair the blocking union contains, and what survives each stage, measured
on the query set Q (1,765,506 train S1 with 20% of S1 hidden as orphans):

| stage | candidates per S1 | recall on Q |
|---|---|---|
| blocking union (3 forward passes + reverse + exact keys) | 101.9 | 0.99069 |
| after the prune model | 7.72 | 0.98862 |

Pruning discards 92% of the union and 0.2% of the recall. The 0.99069 ceiling is
what any matcher downstream is working against; the 3.7 points between that and
the final score sit in the matcher, not in retrieval.

#### B.2 Choosing K and min-p

`prune.py` writes `work/prune_grid.csv`: Q recall and candidate volume across
K ∈ {5…40} × min-p ∈ {0.02…0.0005}, from a single scoring pass, so the rule can
be set without re-running the stage. Representative rows, against 6,109,055 true
pairs in Q:

| K | min-p | pairs per S1 | true pairs kept | recall |
|---|---|---|---|---|
| 5 | 0.005 | 4.42 | 5,713,912 | 0.93532 |
| 10 | 0.005 | 6.00 | 6,032,251 | 0.98743 |
| 20 | 0.02 | 4.60 | 6,001,098 | 0.98233 |
| 20 | 0.01 | 5.75 | 6,031,243 | 0.98726 |
| **20** | **0.005** | **6.42** | **6,039,534** | **0.98862** |
| 20 | 0.002 | 7.87 | 6,047,524 | 0.98993 |
| 20 | 0.0005 | 10.30 | 6,051,247 | 0.99054 |
| 40 | 0.005 | 6.44 | 6,039,716 | 0.98865 |
| 40 | 0.0005 | 10.86 | 6,052,666 | 0.99077 |

Two things fall out of the grid, and they set the operating point:

- **K stops mattering at 20.** Going 20 → 40 at min-p 0.005 recovers **182** more
  true pairs out of six million. The per-S1 cap is not what is binding; the score
  floor is.
- **min-p is the real lever, and it is expensive.** Dropping 0.005 → 0.0005 buys
  0.0019 recall for 60% more candidates — every one of which the ~75-feature
  stage and both GBDTs then have to score. K=5 is the only genuinely bad setting
  in the table, giving up 325,622 true pairs.

`(K=20, min-p=0.005)` is the knee: the last point where recall is still being
bought at a sensible price. Because F0.5 weights precision four times recall,
paying 60% more compute for 0.2% more recall that the matcher must then reject
is the wrong trade.

#### B.3 Where the pruning loss falls

| | share of true pairs lost |
|---|---|
| India | 0.21% |
| US | 0.21% |
| Source 2 | 0.20% |
| Source 3 | 0.22% |

Even across both labelled countries and both noisy sources — no segment is
absorbing the loss on behalf of the others.

Candidates per S1 on the test set, by country: **France 10.25, India 7.80,
US 6.63**. France is the one country with no training labels anywhere, so this
is a label-free check that the per-country floors are not quietly starving it;
it keeps more candidates per entity than either country we can measure.

#### B.4 Reproducibility

Runs are not bit-identical to one another. `ER_JOBS` sets LightGBM's
`num_threads`, which fixes the order histogram bins are summed in, so prune
scores shift slightly with thread count and CPU architecture and pairs sitting
exactly on the top-K / min-p boundary can fall either way. Between two executions
of the pipeline this moved 2,474 of 5,757,784 matched pairs — 0.043%. Pin
`ER_JOBS` to make two runs agree.
