# HLD — High-Level Design

Scope: the path from the released TSVs to `matching_results.tsv` + `candidate_pairs.tsv`,
tuned to the measured properties of *this* dataset (ANALYSIS.md §1). Existing code in
`src/` is the starting point; ✅ = exists, 🆕 = to build, 🔧 = change.

## 1. Design drivers (fact → requirement)

| Fact (ANALYSIS.md) | Requirement it creates |
|---|---|
| 1.7×10¹³ possible pairs on test | Blocking must be sub-quadratic and memory-bounded |
| 99.985% of true pairs share a name or address token | Token-union blocking can reach ~0.99+ recall; the losses are from pruning/K |
| 23% of India S2 names in native Indic scripts; India = 47% of test | Transliteration before blocking and features |
| GT is a partition (F1) | A decision layer that assigns each S2/S3 record to ≤1 S1 |
| Up to 5–6 duplicates per source per entity | Cluster/peer-consistency signal |
| 80k S1 share name + locality; 117k share an address | House number + street must decide; name IDF and co-location features |
| France 15% of test, 0% train; same generator | Country-agnostic features; country-keyed normalizer tables with a generic fallback; drift gate on France cardinality |
| Macro F0.5, singletons 5.6% | Per-entity decision maximizing expected F0.5, not a single global cut |
| 25 GB RAM, 16 threads; AWS available | Columnar (polars/parquet), chunked, parallel; heavy runs on AWS |

## 2. Layered architecture

```
 ┌───────────────────────── L8  mlguard (Rust) — watch · run · split · submission ─────────────────────────┐
 │                                                                                                          │
 TSV ─► L0 Ingest ─► L1 Contract ─► L2 Normalize ─► L3 Blocking (multi-pass ∪) ─► L4 Candidate store ─►   │
        (polars→      (schema,       (NFKC, accent,   A name+addr TF-IDF  ✅          (parquet: s1,cand,     │
         parquet)      counts,        translit 🆕,     B address TF-IDF    🆕           pass flags, ranks,     │
                       prefixes)      skeleton 🆕,     C translit-name     🆕           sims)                 │
                                      domain split 🆕, D reverse S2→S1     🆕                                  │
                                      FR tables 🔧,    E dense ANN (opt)   🆕                                  │
                                      addr parse 🆕)                                                           │
 ─► L5 Scoring ───────────────────────────────► L6 Decision ─────────────────► L7 Output ─────────────────────┘
    stage-1 GBDT on pair features ✅🔧            calibrate (isotonic, OOF) 🆕    writer ✅ + validate_submission ✅
    stage-2 GBDT + competition/peer feats 🆕      assignment: S2/S3 → best S1 🆕  + mlguard submission 🆕
    (optional) cross-encoder on uncertain band    per-entity expected-F0.5 🆕     candidate_pairs = L4 set scored
```

### L0 Ingest ✅🔧
Read each TSV once with an explicit tab separator, assert the row counts from
ANALYSIS.md §1.2, and cache to parquet (`interim/<split>_s{1,2,3}.parquet`). Later stages
never touch TSV again. 🔧 Switch the hot paths from pandas to polars (≈5–10× faster reads,
lower RAM; guess based on typical columnar gains).

### L1 Data contract 🆕
Schema, id prefix per file, unique ids, null counts, and the country label set (logged, never
filtered). Fails loudly on a partial download (Priyanshu hit exactly this: 253k vs 2.2M rows).

### L2 Normalization 🔧
Produces, per record: `name_norm`, `name_core` (legal suffix removed), `legal_form`
(canonical class: LLC / PVT_LTD / SARL / …), `name_translit`, `name_skel` (consonant
skeleton), `name_domain_tokens`, `addr_norm`, `house_nums`, `street_tokens`,
`locality_tokens`, `state_canon`, `postal`. Country-keyed rule tables
(`rules/{US,India,France,_default}.py`) with a **generic fallback for any unseen
country**. That satisfies the open-set rule without learning from `country`.

### L3 Blocking 🔧🆕
A union of cheap passes, all **within country** (F7). Each pass records its rank and similarity.

| Pass | Text | Why | K (start) |
|---|---|---|---|
| A ✅ | name core + address core, word TF-IDF | existing backbone, 0.95 @30 | 30 |
| B 🆕 | street tokens + house numbers + locality | rescues native-script and domain names (addresses share tokens in 95.5% of positives) | 15 |
| C 🆕 | transliterated + skeleton name tokens + locality | rescues cases where the address is empty or different | 15 |
| D 🆕 | reverse: each S2/S3 → top-3 S1 | partition means every record needs its best owner; also yields competition features | 3 |
| E (opt) | multilingual-e5-small embedding, HNSW | what token passes miss; GPU/AWS only | 10 |

Union size target: ≤ 60 candidates/entity average, pair recall ≥ 0.985 (E02–E05).

### L4 Candidate store ✅🔧
Parquet keyed by (split, parameter hash). Columns: `s1_id, cand_id, src(S2|S3),
passA_rank, passA_sim, …, n_passes`. **`candidate_pairs.tsv` is written from exactly this
set after any pre-model filter.** That is the rule in the spec.

### L5 Scoring 🔧🆕
- **Stage 1**: LightGBM on ~60 pair features (LLD §5): name, address, numbers,
  transliteration, legal-form compatibility, IDF/genericity, blocking-pass features, source.
  GroupKFold(5) by S1. Out-of-fold scores are kept.
- **Stage 2**: LightGBM on stage-1 OOF score + **competition** (this candidate's rank among
  all S1s that claim it, gap to the best claimant) + **peer** (similarity to the entity's
  other strong candidates, number of strong peers) + entity context (n candidates, top-k
  score shape). Same folds.
- Optional: cross-encoder (mDeBERTa-v3-base / e5 fine-tuned) only on 0.2 < p < 0.8.

### L6 Decision 🆕
1. **Calibrate** stage-2 scores with isotonic regression fit on OOF.
2. **Assignment**: for each S2/S3 id claimed by several S1 with p ≥ p_min, keep only the
   argmax claimant (hard), or down-weight the others (soft). E07 chooses.
3. **Per-entity expected-F0.5 selection** over the sorted candidates, where k = 0 (empty) is
   also an option. See LLD §7 for the formula. E08 compares it with the global threshold.

### L7 Output ✅🔧
Writer (exists) → `validate_submission.py` (exists) → `mlguard submission --summary` (new),
which also checks per-country predicted singleton rate and links/entity against OOF.

### L8 Trust layer 🆕 (`tools/mlguard`, Rust)
Watches training in the background, gates every run, and runs in GitHub Actions. See MLGUARD.md.

## 3. Key decisions

| # | Decision | Alternatives rejected | Reason |
|---|---|---|---|
| D1 | GBDT (LightGBM) as the core matcher | End-to-end transformer; Siamese nets | Tabular similarity features are strong for ER, train in minutes on CPU, are explainable, and survive domain shift better than a model fitted to US/India vocabulary (guess, E14 checks) |
| D2 | Transliterate deterministically, not with a model | LLM/seq2seq transliteration | Offline, fast, auditable, no licence or compute risk |
| D3 | Multi-pass union blocking | One bigger K | More K adds weak candidates; new passes add *different* candidates |
| D4 | Decision layer as its own stage | Threshold inside the model | Uses the partition and the metric directly; cheap to iterate on OOF |
| D5 | Country only as equality + rule-table key | One-hot / per-country models | Spec forbids hard-coding; France has no labels |
| D6 | Group by S1 in CV | Random pair split | Pair split leaks entity context; the partition makes S1 grouping complete |
| D7 | Heavy runs on AWS | Laptop only | The 4–5 h laptop blocking run leaves no room for iteration in 2 days |
| D8 | Rust checker, not Python | Python asserts | Runs as an independent process during training (it cannot be killed by a Python OOM), fast on 1.7M-row outputs, single binary in CI |

## 4. Non-functional budgets

| Budget | Target | How |
|---|---|---|
| Peak RAM (local) | ≤ 20 GB | polars, drop text after blocking blobs (exists), chunked matmul |
| Test blocking | ≤ 90 min on AWS (≤ 5 h local) | processes per (country, chunk); passes B–D are smaller than A |
| Test featurization | ≤ 60 min for ~100M pairs | `rapidfuzz.process.cpdist(workers=-1)`, numpy vectors, no Python loops |
| Training (stage 1+2) | ≤ 45 min on 300k entities | LightGBM, 5 folds, early stopping |
| Reproducibility | clean checkout → outputs | pinned requirements, seeds, parquet cache keyed by parameter hash |
