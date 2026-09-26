# Master Plan — Business Entity Resolution (Amazon ML Challenge 2026)

**Merged to `main` on 2026-09-25.** This is the team's plan of record — the
working pipeline, the analysis, and the `mlguard` trust checker now live
together on `main`, and that is where work continues.

Findings from executed experiments are folded back into these documents as
they land; [`../EXPERIMENTS.md`](../EXPERIMENTS.md) is the append-only run log
and the source of truth for any number quoted here.

| Doc | What it answers |
|---|---|
| **This file** | What the problem is, which CS concepts solve it, how they map to our data, and what the solution must deliver |
| [ANALYSIS.md](ANALYSIS.md) | Measured data facts, **the full question register with answers**, the weakest-hypothesis audit, and the Pandora's Box pass |
| [HLD.md](HLD.md) | High-level design: layers, data flow, key decisions |
| [LLD.md](LLD.md) | Low-level design: modules, functions, schemas, algorithms, memory and time budgets |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | End-to-end system and a day-by-day implementation plan (who / what / when) |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Ordered experiments E01–E18, each with a question, setup, success and kill criteria |
| [EDGE_CASES.md](EDGE_CASES.md) | Everything to test on train and test: data, text, blocking, model, output, ops |
| [DATA_SECURITY_AND_LEAKAGE.md](DATA_SECURITY_AND_LEAKAGE.md) | Leakage map, validation protocol, compliance with the fair-play rules, data handling |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | Candidate models (licence-checked) and the champion/challenger decision framework |
| [MLGUARD.md](MLGUARD.md) | The Rust checker: what it enforces, how it runs during training / in CI / at test time |
| [GPU_PLAN.md](GPU_PLAN.md) | RTX 3050 plan after review: what runs on the GPU (band reranker), what was dropped and why |

---

## 1. What the problem is

Three sources describe businesses in the US, India and (test only) France. **Source 1** is a
clean, deduplicated reference list. **Sources 2 and 3** are noisy copies: typos, reordered
words, legal suffixes moved or dropped, addresses abbreviated or reordered, names written
in Hindi/Tamil/Telugu/… script, names replaced by a web domain. There are also ~26%
records that belong to no Source-1 business at all. For every Source-1 business we must
output the list of S2/S3 record ids that are the same real business.

Scoring is **macro F0.5**: F0.5 is computed per Source-1 entity and averaged. Precision
counts twice as much as recall, and an entity with no true match scores 1.0 only if we
predict nothing. The practical difficulty is scale (1.73M × 9.97M possible pairs on test),
noise (only 20% of true pairs have the same normalized name), hard look-alikes (80k S1
businesses share name *and* locality with another), and a whole country (France, 15% of
test) with zero training labels.

## 2. CS concepts that solve it

| # | Concept | What it is (one line) |
|---|---|---|
| C1 | **Entity resolution / record linkage** | Deciding which records refer to the same real-world thing (Fellegi–Sunter framework) |
| C2 | **Blocking / candidate generation** | Cheaply shrinking n×m comparisons to a small candidate set without losing true pairs |
| C3 | **Inverted index + TF-IDF + sparse matrix products** | Information retrieval: records that share rare tokens meet; sparse top-k matmul finds them |
| C4 | **Approximate nearest neighbour (ANN)** | HNSW/IVF search over embeddings when token overlap fails |
| C5 | **String similarity metrics** | Levenshtein, Jaro–Winkler, token-set ratio, Jaccard, char n-gram cosine |
| C6 | **Text normalization / Unicode** | NFKC, accent folding, casefolding, **transliteration** across scripts, phonetic keys |
| C7 | **Supervised pairwise classification** | Gradient-boosted trees over pair features; the labels come from GT |
| C8 | **Probability calibration** | Isotonic/Platt so scores mean probabilities (needed for expected-F decisions) |
| C9 | **Decision theory for F-measures** | Choosing the prediction set that maximizes *expected* F-β, per entity |
| C10 | **Bipartite assignment / constraint satisfaction** | Each S2/S3 record goes to ≤1 S1: a one-to-many matching constraint |
| C11 | **Graph / clustering consistency** | True matches form clusters of mutual duplicates; use peer agreement |
| C12 | **Stacking / two-stage learning** | A stage-2 model on out-of-fold stage-1 outputs plus context features |
| C13 | **Cross-validation with groups** | GroupKFold by entity so no entity leaks across folds |
| C14 | **Domain generalization / covariate shift** | Train on US/India, test on France and an India-heavy mix |
| C15 | **Out-of-core / memory-bounded computing** | Chunked sparse ops, columnar storage (parquet/polars), streaming |
| C16 | **Parallel computing** | Multi-process chunks, SIMD string scorers (rapidfuzz), Rust for checks |
| C17 | **ML reliability engineering** | Leakage tests, overfitting monitors, drift checks, CI gates |

## 3. Concept → problem → data mapping

| Concept | Where it hits this problem | Evidence in our data (ANALYSIS.md) | Where in the design |
|---|---|---|---|
| C1 ER | The task itself | S1 dedup reference, S2/S3 noisy (§1.5) | Whole pipeline |
| C2 Blocking | 1.73M × 9.97M = 1.7×10¹³ pairs | 99.985% of true pairs share a name or address token (§1.3) | L3 multi-pass blocking |
| C3 TF-IDF/sparse | Rare-token co-occurrence is the backbone | 0.95 recall @K30 measured by Priyanshu | L3 pass A (existing) + pass B/C |
| C4 ANN | Pairs sharing no surviving token | 0.015% share none; more lost to pruning/K | L3 pass D (optional, GPU) |
| C5 String sims | Typos, reorders, abbreviations | `Tetlecommunication`, `EVE'S ASSOCIATES HIGHLAND` | L5 features |
| C6 Normalization | Case, accents, 9 Indic scripts, domains, FR abbreviations | 23% of India S2 names in native script (§1.4); injected accents everywhere | L2 normalizer |
| C7 Classifier | Pair decision | 7.6M labelled positives on train | L5 GBDT |
| C8 Calibration | Expected-F needs real probabilities | — | L6 |
| C9 Expected F | Macro F0.5 per entity; singletons score 1.0 only for empty output | 5.58% singletons; mean 3.46 links | L6 decoder |
| C10 Assignment | GT is a partition | 0 S2/S3 ids linked to 2 S1 (F1) | L6 conflict resolution |
| C11 Clusters | S2 holds up to 5 copies of one business | F5, F6 | L5 stage-2 peer features |
| C12 Stacking | Competition + peer features need stage-1 scores | — | L5 stage 2 |
| C13 GroupKFold | Entity-level leakage | Partition means grouping by S1 also groups S2/S3 | Validation |
| C14 Domain shift | France 15% unseen; India 40%→47% | §1.2 | Country-agnostic features, LOCO validation, drift gate |
| C15 Out-of-core | 2.4 GB TSV → 10M-row frames, 25 GB box | Priyanshu measured ~9.7 GB free | Parquet cache, chunked blocking |
| C16 Parallel | ~100M pairs to featurize on test | Python loops = hours | `rapidfuzz.process.cpdist(workers=-1)` |
| C17 Reliability | Overfit/leak/drift must not reach the leaderboard | 5 submissions/day | `mlguard` in background + CI |

## 4. What the solution must deliver (from the problem statement and guidelines)

**Outputs**
- [ ] `output/matching_results.tsv`: header `source1_entity_id<TAB>matched_entity_ids`; one row per test S1 (1,732,544 rows); ids comma-separated, no quoting; empty list for singletons; no duplicate ids in a list; no duplicate rows; only S2-/S3- ids that exist in test.
- [ ] `output/candidate_pairs.tsv`: header `source1_entity_id<TAB>candidate_entity_ids`; the **exact** set the final model scored (after all filtering); same rules; matches ⊆ candidates.
- [ ] Both files pass `validate_submission.py` **and** `mlguard submission`.

**Package** (`<team_name>_submission.zip`)
- [ ] `output/` with both TSVs.
- [ ] `code/business_entity_resolution/src/` (all code), `README.md` (exact data → blocking → matching → output steps), `requirements.txt` (pinned).
- [ ] `Documentation_template.md` filled in: methodology, blocking strategy, model architecture + features, threshold method, results + error analysis (FP/FN).
- [ ] The guidelines' 1–2 page summary: approach, models, experiments, conclusion. Code needs comments describing functions.

**Rules**
- [ ] Final model MIT or Apache-2.0 licensed, ≤ 8B parameters.
- [ ] No external databases, APIs, geocoders, registries, internet augmentation. No LLM API calls on the data.
- [ ] `country` treated as an open set: no hard-coding/filtering/one-hot of {US, India}; every France entity in the output.
- [ ] Read and write TSV with an explicit tab separator.
- [ ] ≤ 5 leaderboard submissions per day; log every one in `docs/EXPERIMENTS.md`.
- [ ] Keep the version history of every submission (git tag per upload: `lb-<date>-<n>`).
- [ ] Challenge window ends 27 Sep 2026, 23:59 IST.

**Quality bar we set ourselves**
- [ ] OOF macro F0.5 reported overall **and per country**, plus leave-one-country-out.
- [ ] Every training run is gated by `mlguard` (background watch + end-of-run check + CI).
- [ ] The full test run is reproducible from a clean checkout in ≤ 6 h on the documented hardware.
