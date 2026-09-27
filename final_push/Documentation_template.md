# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** CUDA_COMMONDOS
**Team Members:** Priyanshu Doshi, Neal Daftary, Krisha, Krina
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A CPU pipeline of sparse TF-IDF blocking, a LightGBM prune model, a pairwise
LightGBM (stage 1) and a context-aware LightGBM (stage 2), followed by a
decision layer that enforces the one-to-one structure of the ground truth and
chooses, per Source-1 entity, the subset of candidates that maximises expected
F0.5. An optional multilingual cross-encoder adds scores on the ambiguous band.
Measured out-of-fold macro F0.5 on held-out train S1: **0.98654**.

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

- **F_0.5 Score (macro):** 0.98654 out-of-fold on 1.77M held-out train S1 with
  20% of S1 hidden as orphans (US 0.98731, India 0.98537, singletons 0.98733).
  The test output passes `validate_submission.py`.
- **Common false positives (wrong merges):** same-name businesses in the same
  locality with different house numbers or units (branches, chains).
- **Common false negatives (missed matches):** records whose name differs by
  script, alias or rebranding and whose address is sparse or missing.

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
`cd code/business_entity_resolution/src && python run_pipeline.py`.

### B. Additional Results

Per-segment F0.5 above; prune recall diagnostics are printed by `prune.py report`.
