# Changes: Sentence-Transformer Features + LR/LightGBM Ensemble

## Summary

On top of the base pipeline (`config.py` → `normalise.py` → `blocking.py` → `features.py` → `run_pipeline.py`, word-level TF-IDF blocking at `TOP_K=30`, features = Jaro-Winkler / token overlap / Jaccard / numeric, single LightGBM model), two changes were made:

1. Added a new feature block: sentence-transformer cosine similarity.
2. Changed the training/scoring path from single LightGBM to a fixed 50/50 blend of LightGBM + Logistic Regression.

Note: base model is **LightGBM** (`lgb.train`, binary objective, `GroupKFold` CV grouped by `s1_id`), not XGBoost — correcting an earlier mischaracterization.

No changes were made to blocking, normalisation, or config.

## 1. New feature: `build_embedding_features`

Added to `features.py`.

- Model: `all-MiniLM-L6-v2` (sentence-transformers), frozen/pretrained, no fine-tuning, CPU-only inference. Loaded once via a module-level singleton (`_get_st_model`) so it's not reloaded per call.
- Encodes `_core_name` and `_core_addr` once per unique record across S1 and the candidate pool (`others`), not per pair — embeddings are computed once and looked up by ID for each pair, avoiding redundant encoding of duplicated candidates across pairs.
- Embeddings are L2-normalized at encode time (`normalize_embeddings=True`), so cosine similarity reduces to a dot product: `np.sum(L * R, axis=1)`.
- Output: two new columns per pair —
  - `st_name_cosine`
  - `st_addr_cosine`

These get appended to the existing feature set (Jaro-Winkler, token overlap, Jaccard, numeric) before training.

## 2. Training/scoring: ensemble (LightGBM + Logistic Regression)

`run_pipeline.py` training and scoring logic now fits both models per fold, on the same feature set (baseline features + the two new ST cosine features):

- **LightGBM**: unchanged params (`binary` objective, `binary_logloss`, `lr=0.05`, `num_leaves=63`, `min_data_in_leaf=50`, feature/bagging fraction 0.8, early stopping on 100 rounds).
- **Logistic Regression**: `StandardScaler` fit on the training fold, then `LogisticRegression(max_iter=1000, C=1.0)` on the scaled features. Scaling matters here specifically because LR is sensitive to feature magnitude and the new ST cosine features (bounded [-1, 1]) sit on a different scale than raw Jaro-Winkler/Jaccard/token-overlap features — LightGBM doesn't need this since it splits on thresholds, not magnitudes.
- **Combination**: fixed, unweighted average — `oof = 0.5 * oof_lgb + 0.5 * oof_lr` — same blend applied at test time (`0.5 * lgb_scores + 0.5 * lr_scores`). Not a learned/stacked blend.
- Both models are trained per `GroupKFold` fold (grouped by `s1_id` so no entity leaks across train/val), and test-time scores are the average of all fold models for each of LightGBM and LR before the 50/50 blend.
- Decision threshold is tuned by sweeping macro F_0.5 (precision-weighted 2x) on the blended OOF scores — same `tune_threshold` logic as before, just now sweeping the ensemble's output instead of a single model's.

### Confound to flag

Two things changed at once versus the baseline: the **features** (added ST cosine) and the **model** (single LightGBM → LightGBM+LR blend). If the ensemble scores higher than baseline, it won't be possible to attribute the gain to the embeddings vs. the blending vs. both, unless a third config is also run: LightGBM-only *with* the ST features added. Worth doing that as a middle data point before drawing conclusions on "worth it or not."

## Results (training-side only — OOF, sample=2000)

Run: `python src/run_pipeline.py --sample 2000`, stopped after CV/threshold-tuning (before test blocking) via Ctrl-C.

- Features built: `(60000, 34)` — 60,000 pairs, 34 columns (baseline features + 2 ST cosine features + rank features)
- 5-fold GroupKFold, LightGBM best iterations: 180, 148, 127, 166, 172
- **Best threshold: 0.56 → OOF macro F_0.5 = 0.8959**
- Breakdown: `n_entities=2000`, `n_singletons=120`, `singleton_f05=0.8667`, `matched_f05=0.8978`, `mean_pred_size=2.9565`, `mean_true_size=3.462`

**Top-12 features by LightGBM gain (fold 0):**
`name_addr_mean`, `block_sim`, `name_addr_min`, `addr_num_jaccard`, `name_token_sort`, `name_partial`, `addr_num_agree`, `core_jw`, `addr_token_set`, `name_ratio`, `core_len_diff`, `addr_containment`

**Neither `st_name_cosine` nor `st_addr_cosine` appears in the top 12.** LightGBM's gain-based importance ranks both new embedding features below all baseline hand-crafted features at this sample size. This doesn't rule out a contribution — LR may weight them differently than LightGBM does, and gain importance can undersell features that are correlated with stronger existing ones — but it's a real result to report as-is, not a reason to assume the embeddings are pulling their weight.

### Caveat: no baseline number yet

This run only tested the full config (LightGBM+LR ensemble + ST features). **No plain-LightGBM-only run (without ST features) has been done yet** — so `0.8959` has nothing to be compared against so far. Per the "next steps" below, at minimum a LightGBM-only baseline (no ST, no LR) on the same 2,000-sample candidate set is needed before claiming any improvement.

## Blocking (unchanged, for reference)

```
python src/run_pipeline.py --blocking-only --sample 2000
```

- S1=2,000 (sampled), S2=5,034,616, S3=5,285,603, gt=2,000
- TF-IDF vocab: 126,023 tokens (max_df=0.01, min_df=3), fit on 1M of 10,320,219 docs
- India: 773 queries scored against 4,133,346 records
- US: 1,227 queries scored against 6,186,873 records
- 60,000 candidate pairs cached (K=30)
- **Blocking recall ceiling @K=30: 0.9506** (mean 30.0 candidates/entity, 0 orphans)

This confirms blocking (unchanged from baseline) is healthy and caps achievable match recall at ~95% regardless of downstream model — this ceiling applies equally to both pipelines being compared, since blocking wasn't touched.

The new feature function and ensemble logic have **not** been exercised yet — this run used `--blocking-only`, which skips `features.py` and training entirely.

## Next steps, before comparing against the other pipeline

1. Run full pipeline (no `--blocking-only`) on the same sample so `build_embedding_features` and the LightGBM+LR ensemble actually execute. Note: the un-flagged run also does full test-set blocking + inference afterward, which the pipeline's own docstring estimates at ~4-5 hours. If only the training-side comparison (OOF macro F_0.5) is needed, it prints right after CV/threshold-tuning, before test blocking starts — safe to stop there for a quick comparison.
2. Hold blocking fixed (same cached candidate parquet) and compare at minimum:
   - baseline features + LightGBM only
   - baseline features + ST cosine features + LightGBM+LR ensemble
   - (recommended middle point) baseline features + ST cosine features + LightGBM only
   to isolate whether any gain is from the features, the ensembling, or both.
3. Score against the rust-based leak-checked holdout rather than in-pipeline OOF metrics for the final call, since that's the agreed-upon comparison standard.
4. Report results alongside the 0.9506 recall ceiling so match-rate numbers are read in that context.

## Known caveats

- CPU-only ST inference: encoding cost scales with number of unique S1 + candidate records, not number of pairs — should be fine at this sample size but worth timing at full scale.
- The 50/50 blend weight is fixed, not tuned or learned — if LightGBM and LR turn out to disagree a lot, a different fixed weight (or a learned stack) might beat the naive average; not explored yet.
- Test-set inference at full scale is bottlenecked by blocking (~4-5 hrs per the pipeline's own estimate), independent of anything changed here — this is a shared cost for both pipelines being compared, not specific to the ensemble.
