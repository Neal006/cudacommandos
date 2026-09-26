# Data Security, Leakage and ML Failure Modes

Three kinds of risk: **(1)** breaking the fair-play rules (disqualification), **(2)**
leakage/overfitting that inflates OOF and then collapses on the leaderboard, **(3)** losing
or exposing the data and credentials.

## 1. Fair-play compliance (disqualification risk)

| Rule (problem statement) | What it forbids for us | Our control |
|---|---|---|
| No external databases, APIs or services to resolve entities | Geocoders, company registries, search engines, **LLM APIs (Claude/GPT/Gemini) called on records**, commercial ER tools | No network calls in `src/`. The CI "fair-play gate" in `mlguard.yml` fails on imports of `requests`, `httpx`, `urllib`, `aiohttp`, `openai`, `anthropic`, `google.generativeai`, `googlemaps`, `geopy` inside `src/` (passes today) |
| No external data augmentation | Downloaded gazetteers, postal-code tables, business lists, web word-frequency lists | The domain-split dictionary is built from **S1 names in the data**; no internet corpora |
| Final model MIT/Apache-2.0, ≤ 8B | GPL/CC-BY-NC/Llama-licence models; Qwen2.5-3B (not Apache) | MODEL_SELECTION.md licence column. We apply it to *every* model (rule R4 default) |
| Country is an open set | Filtering/one-hot on {US, India} | Country only as equality + rule-table key; UT with an unseen country |

**Grey zone. Decide explicitly, document, and ask via the Google Form (ANALYSIS R2/R3):**
- Offline transliteration libraries (anyascii ISC, indic-transliteration MIT) and
  hand-written abbreviation/state/department tables. These are code and linguistic
  knowledge, not identity lookup. **We use them and document them** in the methodology.
- Using **unlabeled test text** (IDF fit, dictionaries). Standard transductive practice
  with no labels. **Used, and documented.**
- Synthetic training pairs generated from test-France S1 records. **Not used** unless the
  organizers say yes.

## 2. Leakage map

| # | Leak | How it happens here | Symptom | Prevention | Detector |
|---|---|---|---|---|---|
| L1 | **Entity split leak** | Pairs of one S1 in both train and valid folds | OOF too optimistic | GroupKFold by `s1_id`; the partition means S2/S3 follow automatically | `mlguard split --folds` (group_leak) |
| L2 | **ID leak** | Numeric part of `entity_id` as a feature, or row order | Suspicious single-feature dominance | Ids never enter `X` | `banned_feature`, `feature_dominance` |
| L3 | **Threshold in-sample** | Threshold/decision params tuned on in-fold predictions | Optimistic OOF, worse LB | Tune only on OOF | `threshold_source` must be `oof`/`holdout` |
| L4 | **Stacking leak** | Stage-2 features from stage-1 predictions made by a model that saw the pair's label | Stage 2 looks miraculous | Stage-2 features from **OOF** p1 only, same folds | `overfit_gap` on stage-2 folds; `too_good` |
| L5 | **Label-derived features computed globally** | Target encoding of tokens, or "how often this token appears in true pairs", over all train | OOF inflated | Any label-derived statistic is computed inside the training folds | code review + `too_good` |
| L6 | **Competition features across folds** | Claimant ranks use p1 of other entities, possibly from other folds | Mild; usually fine because every p1 is OOF | Only OOF p1 anywhere | — |
| L7 | **Blocking tuned on GT of the eval entities** | K/df tuned on the same entities we report OOF on | Slight optimism | Tune blocking on a separate 50k slice; report OOF on the rest | — |
| L8 | **Score beats the blocking ceiling** | Any of the above | Impossible OOF | — | `beats_ceiling` |
| L9 | **Public-LB overfitting** | Choosing among near-ties by LB score | Private LB drop | LB confirms and never selects (EXPERIMENT_PLAN §D) | OOF→LB log |
| L10 | **Vocabulary fit on train only, used on test** | IDF/vocab missing French and India-test tokens | France/India features degraded | Fit the vectorizer on train+test text (unlabeled) | E13 drift |

## 3. Other ML failure modes and their guard

| # | Problem | Why it's likely *here* | Guard |
|---|---|---|---|
| 3.1 | Overfitting in boosting | 100M+ pairs, deep trees, easy positives | Early stopping on valid; `mlguard watch` (diverging_valid, loss_ratio); `overfit_gap` |
| 3.2 | Fold instability | Clustered data (localities) | `fold_instability` (std > 0.01) |
| 3.3 | **Covariate shift** | France unseen; India 40→47% | Country-agnostic features; LOCO (E10); per-country report |
| 3.4 | **Label shift in cardinality** | A normalizer bug on France → 0 matches | `test_singleton_drift`, `test_links_drift` (label-free, on test outputs) |
| 3.5 | Sampling bias | 150k random S1 → rival owners missing (weakest hypothesis #1) | E06 geo-cluster sampling |
| 3.6 | Ensemble-vs-OOF calibration shift | Test p = mean of 5 fold models (smoother than any single OOF model) | Compare the test p histogram to OOF; if shifted, calibrate on the OOF of the averaged-seeds model or use a single refit |
| 3.7 | Class imbalance | ~3.5 positives per ~60 candidates | GBDT copes; do **not** resample (it distorts calibration); isotonic on OOF |
| 3.8 | Silent NaN | Empty address features | UT + `nan_loss` |
| 3.9 | Non-determinism | Threads in LightGBM, dict order | Fixed seeds, `deterministic=True` for the final run |
| 3.10 | Train/test pipeline skew | Different code paths for test | One `featurize()` for both; the same cache keys |

## 4. Validation protocol (the one number we trust)

1. Fix the evaluation entities: GroupKFold(5) over the sampled S1, seed 42, fold file committed.
2. Blocking parameters are tuned on a disjoint 50k-entity slice.
3. Report: OOF macro F0.5, per-country, LOCO (US→India, India→US), blocking recall
   ceiling, predicted singleton rate vs 5.58%, links/entity vs 3.46.
4. `mlguard run --champion runs/champion.json` must pass before an upload.

## 5. Data handling and security

| Asset | Rule |
|---|---|
| Dataset (~2.4 GB) | Never committed (`.gitignore` covers `data/ dataset/ *.tsv *.parquet`). Lives in `AMLC_DATA_DIR` and the team S3 bucket only |
| S3 bucket | `amazon-cuda-commandos-2026` (ap-south-1), **owned by a teammate's account**, cross-account read+write granted per member. We do not hold the bucket policy, so "no `Principal: *`" is unverified from our side — ask the owner to confirm Block Public Access is ON. Access is via `aws login` SSO, not static keys ([`../TEAM_BUCKET.md`](../TEAM_BUCKET.md)) |
| Credentials | Never in the repo (`.gitignore`: `.aws-credentials`, `*.pem`, `.env`). Use `aws configure` profiles; rotate keys after the challenge |
| Outputs | `output/*.tsv` ignored; the uploaded versions are kept as S3 objects named by git tag |
| Third-party services | No record text is sent to any external service: no LLM chat, no pastebins, no online TSV viewers |
| Logs | `runs/*/pipeline.log` may contain sample records. Committing them is OK (private repo). **If the repo is public, don't commit logs with record text** |
| Repo visibility | Checked 2026-09-25: **PRIVATE**. Keep it private until the challenge ends |
| Local disk | The OneDrive-synced repo means data under the repo would sync to the cloud. Keep the dataset outside the repo (config already does) |
| Cleanup | After the challenge: delete the AWS instances + bucket and revoke the teammate policy |
