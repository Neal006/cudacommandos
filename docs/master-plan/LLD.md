# LLD — Low-Level Design

Concrete modules, schemas, algorithms and budgets. File names match `src/`. ✅ exists ·
🔧 change · 🆕 new.

## 1. Module map

| File | Status | Responsibility | Key symbols |
|---|---|---|---|
| `src/config.py` | 🔧 | paths + knobs | add `BLOCK_PASSES`, `PASS_K`, `DECISION`, `RUN_DIR` |
| `src/data.py` | 🔧 | TSV I/O, parquet cache, writer | `read_source` → polars; `assert_counts()` 🆕 |
| `src/contract.py` | 🆕 | L1 data contract | `check_split(which) -> dict` |
| `src/normalize.py` | 🔧 | L2 text | + `translit()`, `skeleton()`, `split_domain()`, `legal_form()`, `parse_address()` |
| `src/rules/{US,India,France,_default}.py` | 🆕 | country-keyed tables | `ABBREV`, `LEGAL`, `STATES`, `NOISE_WORDS` |
| `src/blocking.py` | 🔧 | L3 passes | `run_pass(name, q, idx, text_fn, k)`, `reverse_pass()`, `union()` |
| `src/features.py` | 🔧 | L5 pair features | vectorized `build_pair_features()`; `stage2_features()` 🆕 |
| `src/train.py` | 🆕 | stage 1/2 training, OOF, calibration | `fit_stage(X, y, groups, folds) -> (models, oof)` |
| `src/decide.py` | 🆕 | L6 | `assign()`, `expected_f_select()` |
| `src/metrics.py` | ✅ | macro F0.5, breakdown, blocking recall | + `per_country()` 🆕 |
| `src/runlog.py` | 🆕 ✅ | metrics/summary for mlguard | `RunLog` |
| `src/run_pipeline.py` | 🔧 | orchestration | stages as functions, cached |
| `tools/mlguard/` | 🆕 ✅ | Rust trust checker | see MLGUARD.md |

## 2. Schemas

**Record parquet** `interim/{split}_s{n}.parquet`:
`entity_id:str, business_name:str, business_address:str, country:cat, src:u8`.

**Normalized parquet** `interim/{split}_norm.parquet` (all sources):
`entity_id, src, country, name_norm, name_core, legal_form:cat, name_translit,
name_skel, is_native_script:bool, is_domain:bool, addr_norm, house_nums:list[str],
street:str, locality:str, state_canon:str, postal:str|null, addr_empty:bool`.

**Candidates parquet** `interim/cands_{split}_{hash}.parquet`:
`s1_id, cand_id, src, A_rank:u16, A_sim:f32, B_rank, B_sim, C_rank, C_sim, D_hit:bool,
D_rank, n_passes:u8`. The rank is null when the pass didn't produce the pair.

**Scores parquet** `runs/<id>/oof.parquet` / `test_scores.parquet`:
`s1_id, cand_id, fold, p1 (stage 1), p2 (stage 2), p_cal, selected:bool`.

## 3. Normalization (L2), in order

1. `NFKC` → casefold → remove zero-width chars (U+200B–U+200D, U+FEFF). Keep ZWJ/ZWNJ
   *until* transliteration (Indic conjuncts need them), then drop.
2. **Script detection** by Unicode block (9 Indic blocks, ANALYSIS.md §1.4).
3. **Transliteration** of non-Latin runs: `indic_transliteration.sanscript` (MIT) for the 9
   scripts → ITRANS/IAST → ASCII, with `anyascii` (ISC) as the fallback. Output
   `name_translit`. Native-script state names map through the India STATES table.
4. **Accent fold** (NFD, drop combining marks). This is mandatory: accents are injected
   noise in every country.
5. Strip **junk prefixes** `^[^\w]+` (`--`, `<<`, `@`); drop bracket characters but keep
   their content (`[EURL]` → `eurl`, `(Limited)` → `limited`).
6. **Domain names** (`\.(com|in|net|org|co|fr)$`): strip the TLD, then segment the stem with
   a word-break DP whose unigram costs come from **S1 name tokens (train+test)**. That is
   in-data only, so no external corpus. `deltatelecommunication` → `delta
   telecommunication`; `heassociates` → `h e associates` (initials kept for the acronym
   feature).
7. **Legal form** extraction to a canonical class with a country-keyed table (`pvt ltd`,
   `private limited` → `PVT_LTD`; `llc` → `LLC`; `sarl`, `sas`, `sasu`, `eurl`, `sci`,
   `snc` → FR classes). Removed from `name_core`, kept as `legal_form`.
8. **Abbreviations** from `rules/<country>.ABBREV`, falling back to `_default` (the current
   `ADDR_ABBREV`). France adds `r`→`rue`, `av`→`avenue`, `bd`→`boulevard`, `pl`→`place`,
   `all`→`allee`, `imp`→`impasse`, `st`/`ste`→`saint`/`sainte` **in names of places only**,
   `b`/`bis`/`ter` as house-number suffixes. These rules are safe *because they are keyed
   on the country string*, which is exactly why `r` was excluded from the global table.
9. **Skeleton** (`name_skel`), **prototype verified**: `tools/eda/skeleton_prototype.py`.
   anyascii → lowercase letters → drop a leading `y` before a vowel → drop post-vowel `gh` →
   `ction→ksn, tion→sn, ph→f, th→t, kh→k, sh→s, ch→s, ck→k` → hard `c` (before a/o/u/k/r/l/t
   or at word end) → `k` → anusvara `m` before a consonant → `n` → fold `g→k b→p d→t v→w z→s j→s
   c→s q→k` → drop non-leading vowels → collapse repeats. Tamil has no voicing contrast and
   writes ச as `c`, hence the classes. `Global Business` and `குளோபல் பிசினஸ்` → `klpl psns`.
   **Measured:** on 20k native-script India positives, 94.4% share ≥1 core skeleton token
   with S1 (raw: ~0%) and 43.1% are identical. It is a feature (`skel_jw`, `skel_eq`) and a
   pass-C blocking key, never a decision on its own. Transliterated legal forms
   (`praivet`, `limitet`, `elelpi` = LLP) go in `rules/India.LEGAL`.
10. **Address parse** (regex, no geocoder):
    - `house_nums`: `\d+[a-z]?(?:\s*(?:/|-)\s*\d+[a-z]?)*` plus `1/2`, `bis|ter`; leading
      `#`, `no.`, `hn`, `h.no`, `plot no`, `shop no` stripped.
    - `postal`: US `\b\d{5}(-\d{4})?\b`; FR `\b\d{5}\b`; India `\b\d{6}\b`. Nearly always
      absent (ANALYSIS Q11), so it is a feature, never a key.
    - Split on commas. The **locality** is the segment set minus street/number/state
      segments, order-free because components get reordered.
    - `state_canon`: US 2-letter ↔ full name; India abbreviations (MH, TN, …) and
      native-script names ↔ English; France region ↔ departments (hand table, 13 regions,
      96 departments). Note the R2 rule question in ANALYSIS.md.
    - Drop `pmb \d+`, `po box \d+`, `unit|suite|fl|ste \w+` into `addr_extra`, not street.

Every function is pure `str -> str|list` so it runs in `polars.map_elements` or,
better, in a `multiprocessing.Pool` over 1M-row chunks (16 workers).

## 4. Blocking (L3)

All passes run **per country partition** and **per chunk of 2,000 S1 rows** (existing
`BLOCK_CHUNK`). Each (country, chunk) is an independent task, so a `ProcessPool` with
N = min(cores, RAM/peak_per_task) workers runs them.

| Pass | Vectorizer | Query side | Index side | K |
|---|---|---|---|---|
| A ✅ | word TF-IDF, `min_df=3, max_df=0.01` | `name_core + addr core` | same | 30 |
| B 🆕 | word TF-IDF over `street + house_nums + locality`, numbers as tokens `#599` | S1 | S2+S3 | 15 |
| C 🆕 | word TF-IDF over `name_translit` tokens ∪ `skel:` tokens ∪ domain-split tokens | S1 | S2+S3 | 15 |
| D 🆕 | pass A matrix transposed: for each S2/S3 row, top-3 S1 | S2+S3 | S1 | 3 |

- The sparse top-k product uses the existing `_topk_from_sparse_rows`. If single-thread
  scipy stays the bottleneck, swap in `sparse_dot_topn` (check its licence first) or
  split chunks across processes.
- Pass D reuses pass A's vectors: `idx_matrix[chunk] @ s1_matrix.T` over S2/S3 chunks
  (10M queries against a 1.7M index is smaller per query than the forward direction).
- `union()`: outer-join on (s1_id, cand_id), fill ranks with null, `n_passes = count(non-null)`.
- Recall is measured **per pass and per slice** (country × script × domain × empty-address)
  on train, so each pass's marginal gain is known (E03, E04).

## 5. Pair features (stage 1)

Vectorized with `rapidfuzz.process.cpdist(a, b, scorer=..., workers=-1)` over aligned
arrays (C++, multi-threaded). The current Python list comprehensions in `features.py` are
the #1 runtime risk at ~100M test pairs.

| Group | Feature | Definition / why |
|---|---|---|
| Name | `name_ratio, name_tsort, name_tset, name_partial` ✅ | rapidfuzz on `name_norm` |
| | `core_ratio, core_tsort, core_jw, core_jaccard, core_containment` ✅ | on `name_core` |
| | `translit_tset, translit_jw` 🆕 | on `name_translit` (Indic rescue) |
| | `skel_jw, skel_eq` 🆕 | consonant skeleton |
| | `name_char3_cos` 🆕 | char 3-gram TF-IDF cosine (fit on S1 names) via row-wise sparse dot |
| | `name_weighted_overlap` ✅ | Σ idf(shared) / Σ idf(union): generic tokens count little (renamed: `_id` substring trips mlguard banned_feature) |
| | `name_min_shared_idf, name_max_shared_idf` 🆕 | a rare shared token is strong evidence |
| | `name_genericity` 🆕 | log count of S1 entities with the same `name_core` (39.6% share names) |
| | `acronym_eq, acronym_vs_core` ✅ | `ICH` vs `Indian Coffee House` |
| | `is_domain, domain_stem_ratio` 🆕 | the domain stem vs `name_core` without spaces |
| | `either_native, both_latin` 🆕 | tells the model which features to trust |
| | `legal_compat` 🆕 | 0 same class / 1 one side missing / 2 conflicting classes |
| | `noise_word_extra` 🆕 | S3 appends `Center`, `Service`: the count of extra tokens that are in the NOISE_WORDS table |
| Address | `addr_ratio, addr_tsort, addr_tset, addr_jaccard, addr_containment` ✅ | on `addr_core` |
| | `hn_exact` 🆕 | first house number equal (83% US / 73% India on positives) |
| | `hn_fuzzy` 🆕 | one number is a prefix/suffix of the other, or edit distance ≤ 1 (`9914`↔`991`) |
| | `num_jaccard, num_any_shared` ✅ | |
| | `street_tset, street_jw` 🆕 | street only, without locality |
| | `locality_overlap` 🆕 | set overlap of locality tokens (order-free) |
| | `state_eq` 🆕 | canonical state equal / unknown |
| | `postal_eq` 🆕 | 1 / 0 / missing (rare) |
| | `addr_empty_any` 🆕 | ~3% of S2/S3 have no address |
| | `colocation` 🆕 | log count of S1 at the same `addr_norm` (117k share one) |
| | `addr_idf_overlap` 🆕 | like the name version |
| Blocking | `A_rank, A_sim, B_rank, B_sim, C_rank, C_sim, D_hit, D_rank, n_passes` 🆕 | pass evidence is signal |
| Context | `n_candidates, rank_in_entity, gap_to_best` ✅ | label-free, within entity |
| Source | `is_s3` 🆕 | S2/S3 noise styles differ (structural, not identity) |
| Country | `country_eq` ✅ | equality only, never a category |

Banned (mlguard `banned_feature`): anything derived from the numeric part of ids, row
order, fold, label.

## 6. Training (L5)

- **Sample**: 300k train S1 entities by **geo-cluster sampling** (E06): group S1 by
  (country, locality key) and draw whole localities until 300k. Neighbours then compete
  inside the sample as they will on test. Blocking runs against the **full** S2/S3 index.
- **Labels**: `y = cand_id ∈ GT[s1_id]`. Also record the blocking-miss count per entity so
  recall loss is measured, not hidden.
- **Folds**: `GroupKFold(5)` on `s1_id`, written to `runs/<id>/folds.tsv` → `mlguard split`.
- **Stage 1** LightGBM: `objective=binary, lr=0.05, num_leaves=127, min_data_in_leaf=100,
  feature_fraction=0.8, bagging 0.8/1, lambda_l2=1, max_bin=255, early_stopping=100 on
  valid logloss, max 3000 rounds`. `valid_sets=[train, valid]` with `RunLog.lgb_callback`
  so `mlguard watch` sees both curves. Keep OOF `p1` for every pair.
- **Stage 2** features (all from OOF `p1`, so no in-fold leakage):
  - entity shape: `p1_rank, p1_gap_to_max, n_p1_gt_05, sum_p1 (expected match count),
    p1_second_best`
  - competition (needs all claimants of `cand_id`): `claimant_rank` (this S1's rank among
    S1s claiming the candidate by p1), `gap_to_best_claimant`, `n_claimants_gt_05`
  - peer: for candidates of the same entity with p1 > 0.5, `peer_max_sim, peer_mean_sim,
    peer_count`, where `sim = mean(tset(name), tset(addr))` between the two S2/S3 records
  - Stage-2 LightGBM with the same folds and params; OOF `p2`.
- **Calibration**: `IsotonicRegression(out_of_bounds="clip")` fit on OOF `p2` vs `y`. On test,
  apply it to the fold-averaged `p2`. Check the averaged-vs-OOF score histogram
  (DATA_SECURITY §3.6).

## 7. Decision (L6)

```python
def assign(df, p_min=0.05, mode="hard"):
    """GT is a partition: each cand_id belongs to <=1 S1."""
    best = df.group_by("cand_id").agg(pl.col("p_cal").max().alias("p_best"))
    df = df.join(best, on="cand_id")
    if mode == "hard":
        return df.filter((pl.col("p_cal") == pl.col("p_best")) & (pl.col("p_cal") >= p_min))
    # soft: share of the claim mass
    return df.with_columns((pl.col("p_cal") * pl.col("p_cal") / pl.col("p_cal").sum().over("cand_id")).alias("p_cal"))

def expected_f_select(p, miss=0.0, beta2=0.25):
    """p: calibrated probs of one entity's candidates, sorted desc. Returns k to keep.
    F_beta = (1+b2)·TP / (b2·|T| + |P|).  E|T| ≈ Σp + miss (true links blocking missed).
    k=0 is correct only if T is empty: P(T=∅) ≈ Π(1-p)·exp(-miss)."""
    ET = p.sum() + miss
    best_k, best_v = 0, np.prod(1 - p) * np.exp(-miss)
    tp = 0.0
    for k, pk in enumerate(p, 1):
        tp += pk
        v = (1 + beta2) * tp / (beta2 * ET + k)
        if v > best_v:
            best_k, best_v = k, v
    return best_k
```

`miss` is one scalar per country: `(1 − R_country)/R_country · mean Σp`, from the measured
blocking recall. E08 tunes `p_min`, the mode and `miss` on OOF, and compares against the
global-threshold baseline (the current `tune_threshold`).

## 8. Output (L7)

`D.write_outputs` ✅ writes both files with `\n` line endings (explicit `newline="\n"` on
Windows), ids sorted by descending score, no quoting. After that:
`python validate_submission.py …` and
`mlguard submission --matching … --candidate … --test-dir … --summary runs/<id>/summary.json`.

## 9. mlguard hook points in `run_pipeline.py`

Seven small edits. Not applied in this branch, because the 6 h pipeline shouldn't change without a run.

```python
from runlog import RunLog
log_run = RunLog()                                  # MLGUARD_RUN_DIR set by train_guarded.sh
...
folds = np.empty(len(pairs), int)
for fold, (tr, va) in enumerate(GroupKFold(C.N_FOLDS).split(X, y, groups)):
    folds[va] = fold
    dtr, dva = lgb.Dataset(X.iloc[tr], y[tr]), lgb.Dataset(X.iloc[va], y[va])
    m = lgb.train(params, dtr, 3000, valid_sets=[dtr, dva], valid_names=["train", "valid"],
                  callbacks=[lgb.early_stopping(100, verbose=False), log_run.lgb_callback(fold)])
    fold_rows.append(dict(fold=fold, best_iter=m.best_iteration, max_iter=3000,
        train_score=entity_f05(pairs.iloc[tr], m.predict(X.iloc[tr]), thr_guess),   # in-sample F0.5
        valid_score=entity_f05(pairs.iloc[va], oof[va], thr_guess)))
log_run.write_folds(pairs["s1_id"].drop_duplicates(), ...)   # one row per S1 with its fold
...
log_run.write_summary(run_id=run_id, oof_score=cv, threshold_source="oof", folds=fold_rows,
    blocking_recall=rec["pair_recall"], per_country=per_country_f05,
    oof_pred_singleton_rate=..., true_singleton_rate=..., oof_pred_links_per_entity=...,
    feature_importance=dict(zip(X.columns, models[0].feature_importance("gain"))))
log_run.end()
```

## 10. Config knobs (new)

```python
BLOCK_PASSES = ("A", "B", "C", "D")
PASS_K = {"A": 30, "B": 15, "C": 15, "D": 3}
TRAIN_SAMPLING = "geo"          # "random" | "geo"
STAGE2 = True
DECISION = {"calibrate": "isotonic", "assign": "hard", "select": "expected_f", "p_min": 0.05}
RUN_DIR = os.environ.get("MLGUARD_RUN_DIR", "runs/adhoc")
```
