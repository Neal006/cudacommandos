# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** cudacommandos
**Team Members:** Priyanshu Doshi, Neal Daftary, Krisha, Krina
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve business entities with a three-layer pipeline: per-country word-level
TF-IDF blocking reduces 1.7×10¹³ possible pairs to 52.0M candidates at 0.950
recall; a pairwise LightGBM over 46 similarity features scores each candidate;
and a second LightGBM re-scores them using *context* — how hard other entities
compete for the same record, and whether a candidate resembles the entity's
other likely matches. A calibrated per-entity expected-F₀.₅ decoder then chooses
each entity's output set rather than applying one global cut.

The two ideas we would highlight are **treating recall as a budget rather than a
goal** — we deliberately chose K=30 over K=50 because the extra 0.002 of ceiling
was not worth 35M extra pairs — and the **competition features**, which encode
the fact that ground truth is a partition and turned out to dominate stage 2's
feature importance.

---

## 2. Methodology

### 2.1 Problem Analysis

Everything below is measured on the provided data, not assumed.

| Observation | Consequence for the design |
|---|---|
| Only **20%** of true pairs share a normalized name | Exact-name blocking is hopeless; needed token-level retrieval |
| **99.985%** of true pairs share at least one name or address token | Token retrieval *is* sufficient — this justified TF-IDF over embeddings |
| **~26%** of Source-2/3 records match no Source-1 business | The model must be able to say "none"; distractors are the majority of candidates |
| **~23%** of India Source-2 names are in native script (9 scripts) | Needed deterministic transliteration; measured 94% token recovery via `anyascii` + a skeleton key |
| India has **no** PIN codes; US ZIP present in only **~10%** of rows | Postal code is unusable as a blocking key — a natural first instinct that the data rules out |
| **80k** Source-1 businesses share both name *and* locality with another | A large irreducible class of hard negatives; precision, not recall, is the binding problem |
| Ground truth is a **partition** — 0 records link to 2 entities | Competition between entities is signal, exploited in stage 2 |
| Sources hold up to **5–6 duplicate copies** of one business | Peer agreement is signal |
| **5.58%** of entities are singletons; the rest average **3.46** links | Under F₀.₅ an entity with matches predicted empty scores 0, so "predict nothing when unsure" is a trap |
| Test is **15% France**, with **zero** French training labels | A whole country is invisible to cross-validation |

The metric shapes everything. F₀.₅ weights precision twice recall, so a wrong
merge costs more than a missed link. Converting a blocking recall R into the
best score any perfect matcher could reach, `1.25R/(0.25+R)`, gave us a single
number to decide retrieval questions with.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — blocking + two-stage supervised classification +
decision-theoretic decoding.

**Core Innovation:** Stage 2 does not look at the pair in isolation. It sees the
pair's *situation*: the rank of this candidate among the entity's options, how
strongly **other** entities claim the same record (`claim_rank`, `claim_gap`,
`n_claims`, `n_strong_claims`), and how similar the candidate is to the entity's
top other candidates (`peer1_sim`, `peer2_sim`). This encodes the partition
constraint and the duplicate structure as features rather than as a
post-processing rule, and in the trained model those competition features carry
more importance than any raw string similarity.

The second contribution is the **decoder**. Rather than one global threshold, we
calibrate scores isotonically and then, per entity, choose the top-k that
maximizes expected F₀.₅ — including k=0 when the entity looks like a true
singleton.

```
TSV → ingest (polars, Arrow strings)
    → normalize (NFKC, casefold, accents, transliteration, legal-form and address tables)
    → blocking: word TF-IDF per country, df-pruned, chunked sparse top-K      52.0M pairs
    → 46 pair features + 5 rank features → LightGBM ×5 (GroupKFold by entity)  OOF 0.9500
    → stage 2: entity shape + competition + peers → LightGBM ×5                OOF 0.9526
    → isotonic calibration → partition assignment → per-entity expected-F₀.₅   OOF 0.9532
    → matching_results.tsv + candidate_pairs.tsv
```

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:** word-level TF-IDF over a concatenation of the normalized
business name and address, cosine top-K per Source-1 entity, computed
independently within each country.

Normalization before blocking: Unicode NFKC, casefolding, accent folding,
possessive stripping (`Orelee's` → `orelee`, not `orelee south`), legal-suffix
removal (`pvt`, `ltd`, `sasu`, `eurl`, …), French function words (`de`, `du`,
`des`, `la`, `le`, `les`, `et`), address abbreviation expansion applied **only**
to addresses, and deterministic transliteration of non-Latin scripts.

Two scale decisions made this tractable on a 24 GB laptop:

- **Document-frequency pruning** (`max_df=0.01`, `min_df=3`). Tokens in more than
  1% of records ("road", "restaurant", "pvt") produce enormous posting lists and
  almost no discriminative signal; tokens appearing once or twice are typos that
  dominate vocabulary size.
- **Sampled vocabulary fit** (1M documents) and **batched transform**. Document
  frequency is a statistical quantity — a 1M sample gives the same answer as all
  10.5M while fitting ~10× faster. Every document is still transformed.

**Candidate pairs generated:** **51,974,499** over 1,732,544 test entities
(K=30). 50 entities (0.003%) produce no candidate because no token of theirs
survives pruning; they are emitted as empty rows, since a missing Source-1 id is
an outright rejection.

**How we ensured true matches were not lost.** We measured the recall ceiling
directly rather than trusting the design, and chose K from the measurement:

| K | pair recall | best achievable F₀.₅ | test pairs |
|---|---|---|---|
| 5 | 0.8340 | 0.983 | ~9M |
| 10 | 0.9187 | 0.988 | ~17M |
| 20 | 0.9411 | 0.990 | ~35M |
| **30** | **0.9499** | **0.9903** | **52M** |
| 50 | 0.9586 | 0.992 | ~87M |

We stopped at K=30 deliberately. Going to 50 buys **+0.002** of ceiling for 35M
extra pairs to featurize, and our matcher sits 3.7 points *below* the K=30
ceiling — so retrieval is not the binding constraint and spending compute there
would be misallocated. Blocking is done per country because zero cross-country
positives exist in training; `country` is compared by equality only and never
one-hot encoded or filtered on, so France flows through unchanged.

A separate diagnostic (experiment 002) profiled what blocking *does* miss:
records with an **empty address** are over-represented among misses by 7.8×, and
native-script names by 3.6×. The empty-address finding is a design flaw in using
`name + " " + address` as one blob, and is the most promising retrieval fix we
have identified.

---

## 4. Matching Model

**Features used** — 46 pair features, plus 5 rank features, plus 12 stage-2
context features.

- **Name features:** token-set / token-sort / partial / plain Levenshtein ratios,
  Jaro–Winkler, Jaccard and containment over token sets, acronym equality,
  acronym-vs-core, first-token equality, length and token-count differences,
  IDF-weighted token overlap, a "genericity" score (how common the name's tokens
  are), transliterated token-set and Jaro–Winkler, and a cross-script *skeleton*
  key.
- **Address features:** token-set / token-sort / plain ratios, Jaccard,
  containment, length difference, house-number agreement and Jaccard over
  numeric tokens, and a house-number edit distance and near-match.
- **Other:** blocking cosine, combined name+address mean and min, domain-stem
  similarity (for records whose name was replaced by a web domain), legal-form
  compatibility, a native-script indicator, and co-location.
- **Rank features:** the candidate's rank within its entity, the entity's best
  score, the gap to it, candidate count, and an is-best flag.
- **Stage-2 context:** `p1`, `p1_rank`, `p1_gap`, `p1_second`, `n_strong`,
  `sum_p1`, `claim_rank`, `claim_gap`, `n_claims`, `n_strong_claims`,
  `peer1_sim`, `peer2_sim`.

**Model type:** LightGBM (MIT licensed), two stages, 5 folds each, `GroupKFold`
grouped by Source-1 entity so no entity's pairs appear in both train and
validation. Stage 2 consumes **out-of-fold** stage-1 scores and reuses the same
fold assignment — without that, stage 2 would be reading its own training
signal.

Stage-1 importances are led by `name_addr_mean`, `block_sim`, `skel_jw`,
`name_partial` and `addr_token_set`. In stage 2 the ordering is striking: after
`p1` itself, **`n_strong_claims` is the single most important feature**, ahead of
every string similarity. Whether a record is strongly wanted by some *other*
entity is the most informative thing to know about it — direct confirmation that
modelling the partition was the right call. Four features carry exactly zero
importance (`country_eq`, `skel_eq`, `addr_empty_any`, `n_candidates`) and are
dead weight.

**Threshold selection method:** not a threshold. We isotonically calibrate
stage-2 scores (cross-fitted, so no fold calibrates on itself), resolve the
partition constraint, then pick each entity's output set by maximizing expected
F₀.₅:

```
E[F0.5] ≈ 1.25 · Σ_{i≤k} p_i / (0.25 · E|T| + k),     E|T| = Σp + miss
```

with the empty set valued at `Π(1−p)`. `miss` estimates true links blocking
never offered. All 288 combinations of (assignment mode, selection rule,
threshold/miss) are swept on out-of-fold predictions and logged.

Honest note on its value: expected-F beats a single global threshold by only
**+0.0006**, and partition assignment beats doing nothing by **+0.00007**. Both
are kept because they cost nothing at inference and are theoretically correct,
but neither is where the score comes from.

---

## 5. Results & Error Analysis

**F₀.₅ Score (macro):** **0.9532** out-of-fold (150k-entity training sample).
Leaderboard: **0.943**.

| Layer | OOF F₀.₅ | Δ |
|---|---|---|
| pre-v2 feature set | 0.9266 | — |
| stage 1 | 0.9500 | +0.0234 |
| stage 2 (context features) | 0.9526 | +0.0026 |
| decision layer | **0.9532** | +0.0006 |

Per country (out-of-fold): **US 0.9620**, **India 0.9399**. France cannot be
measured offline — it has no labels.

Training-sample sensitivity: 30k → 150k entities improved stage 1 by +0.0010,
stage 2 by +0.0018 and the final score by +0.0020. The gain grows down the
stack, which is what one expects when stage 2's competition features need a
densely covered candidate graph to be meaningful. The curve had not flattened.

### Common false positives (wrong merges)

- **Name-and-locality twins.** 80k Source-1 businesses share both a name and a
  locality with another business — franchise branches, chains, and genuinely
  distinct businesses with generic names on the same road. String similarity
  cannot separate these even in principle; `n_strong_claims` and `claim_gap` are
  what stop most of them, by noticing that two entities want the same record
  equally.
- **Generic-token names.** "City Medical Store", "Sri Krishna Traders" — names
  built entirely from high-frequency tokens. The `name_genericity` feature exists
  for this and carries meaningful importance.
- **Shared-address multi-tenant buildings**, where address similarity is a
  perfect 1.0 and carries no information.

### Common false negatives (missed matches)

- **Blocking losses, 5.0% of true pairs.** Two measured drivers: records with an
  **empty address** (7.8× over-represented among misses) and **native-script
  names** (3.6×). The first is the larger effect and is our own doing — the
  blocking blob is `name + " " + address`, so an empty address dilutes the only
  signal present.
- **Systematic under-prediction.** We output **3.13** links per entity against a
  true average of 3.46. This is *not* a defect: under F₀.₅ precision is weighted
  double, so the expected-F decoder correctly declines marginal links. Pushing
  toward 3.46 lowers the score.
- **France.** The largest single loss. Under the observed country shares
  (France 14.97%, India 46.75%, US 38.27%), and assuming India and US transfer at
  their out-of-fold values, the leaderboard's 0.943 implies France scores near
  **0.90** — about six points below the US. Every parameter downstream of the
  model (the calibrator, `miss`) was fitted on US and India and applied verbatim
  to a country with no labels.

### What we tested and rejected

Recording these because the negative results shaped the design as much as the
positive ones:

- **A larger K.** Measured, and rejected on the ceiling arithmetic above.
- **A GPU cross-encoder band reranker** (both `multilingual-e5-small` and
  `laya`/mmBERT-base, trained and A/B'd head to head). Both raised the 30k
  out-of-fold score by the same ~0.0096 — two architectures with different
  tokenizers and parameter counts agreeing to within 0.0006, which is the
  signature of a shared confound rather than a model gain. The suspected cause is
  record overlap between the reranker's training pairs and the classifier's
  training sample. **We did not ship it**, because we could not verify the gain
  in the time available, and an unverified gain in a precision-weighted metric is
  a risk rather than an asset.
- **Logistic-regression blending** with LightGBM: 0.8959 vs 0.9083 on a matched
  sample. The ensemble hurt; dropped.

---

## 6. Conclusion

We built a measured pipeline rather than a maximal one: every scale decision —
K=30, `max_df=0.01`, a 1M-document vocabulary fit, a 150k training sample — was
taken against a number we had measured, and the recall ceiling `1.25R/(0.25+R)`
let us tell retrieval problems from matcher problems instead of guessing. The
result is 0.9532 out-of-fold and 0.943 on the leaderboard, with the gap almost
entirely attributable to France, a country our validation is structurally blind
to.

The clearest lesson was that the metric, not the text, drove the biggest wins:
the two features that matter most in stage 2 both encode the partition
constraint, and the largest single improvement came from adding context about
competing entities rather than from any better string comparison. The clearest
mistake was the opposite of over-engineering — we validated on two countries and
shipped to three, and the one we could not measure is the one that cost us.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships under `code/business_entity_resolution/`, with all
source in `src/`, a `README.md` giving exact reproduction steps, and a pinned
`requirements.txt`.

| Module | Responsibility |
|---|---|
| `config.py` | All paths and knobs (`TOP_K`, df bounds, sample sizes, chunk sizes) |
| `ingest.py` | polars readers, Arrow-backed strings, parallel blocking blobs |
| `normalize.py` | Unicode, transliteration, skeleton key, legal-form and address tables |
| `blocking.py` | TF-IDF vocabulary, batched transform, chunked sparse top-K, orphan check |
| `features.py`, `features_v2.py` | The 46 pair features and 5 rank features (vectorized, `rapidfuzz.cpdist`) |
| `stage2.py` | Entity-shape, competition and peer features |
| `decide.py` | Isotonic calibration, partition assignment, expected-F₀.₅ decoder, sweep |
| `run_v2.py` | Orchestrator: train → out-of-fold → tune → score test |
| `score_test.py` | Score test from a trained run without retraining; caches per-pair scores |
| `redecide.py` | Re-run the decision layer over cached scores in ~1 min |
| `diagnose_country.py` | Label-free per-country diagnosis |
| `data.py` | TSV I/O and the output writer, which enforces the format rules as it writes |
| `tools/mlguard/` | An independent Rust checker (overfitting, fold variance, submission format) run beside training and in CI |

Reproduce:

```bash
export AMLC_DATA_DIR=/path/to/data          # expects dataset/{train,test}
export AMLC_OUTPUT_DIR=/path/to/output
python src/run_v2.py --sample 150000        # trains and writes both TSVs
```

Scoring the test set streams features in entity-aligned chunks so the 52M-pair
frame is never materialized; peak memory stays near 1.5 GB per chunk against a
19 GB whole-frame requirement. Chunk boundaries fall only where `source1_entity_id`
changes, so every per-entity statistic is identical to a whole-frame run, and the
four candidate-grouped stage-2 columns are computed once globally because a
record can be claimed from different chunks. `tests/test_chunk_equiv.py` asserts
this equivalence and includes a negative control showing that naive chunking
corrupts exactly those four columns.

### B. Additional Results

**Measured throughput** (10-core laptop, 24 GB). Blocking rate is a property of
the index being searched, not the machine:

| Partition | Queries | Index | Queries/sec | Wall |
|---|---|---|---|---|
| France | 259,452 | 1,434,993 | 1,066 | 4.1 min |
| US | 663,106 | 3,817,031 | 800 | 13.8 min |
| India | 809,986 | 4,717,565 | 397 | 34.0 min |

US searches *fewer* records than India and is twice as fast — Indian names and
addresses share more tokens, so posting lists are denser. Full test blocking is
52 minutes; scoring all 52M pairs is 117 minutes.

**Reproducibility and guards.** Every training run is gated by `mlguard`, an
independent Rust checker that watches the metrics stream during training
(overfitting gap, fold variance, validation/train loss ratio), verifies the
finished run, and validates submission format. It runs in CI as well. A guard
stop rolls LightGBM back to its best validation iteration rather than keeping the
stopped one.

**Compliance.** No external data, APIs, geocoders, registries or internet
augmentation are used at any stage. The final pipeline uses LightGBM (MIT),
rapidfuzz (MIT), scikit-learn (BSD-3), pandas (BSD-3), polars (MIT) and anyascii
(ISC); no pretrained neural model is used in the shipped system, so the ≤8B
parameter and MIT/Apache-2.0 licensing constraints are satisfied with room to
spare. `country` is treated as an open set throughout — compared by equality or
used as a rule-table key, never one-hot encoded or filtered — and every test
Source-1 entity appears in the output, France included.
