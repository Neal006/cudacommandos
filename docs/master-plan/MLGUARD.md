# mlguard: the Rust trust checker

`tools/mlguard` is a single Rust binary (deps: serde, serde_json, toml). It runs in three
places:

| When | How | What happens on a problem |
|---|---|---|
| **During training** (background) | `train_guarded.sh` starts `mlguard watch runs/<id>/metrics.jsonl` next to the pipeline | Writes `runs/<id>/MLGUARD_STOP`; the LightGBM callback in `src/runlog.py` raises `EarlyStopException` at the next eval, and the run is flagged |
| **After training / before any upload** | `mlguard split`, `mlguard run --champion`, `mlguard submission --summary` | Non-zero exit → don't upload |
| **GitHub Actions** (`.github/workflows/mlguard.yml`) on every push/PR | unit tests, fixtures self-test, fair-play import gate, gate every committed `runs/*/summary.json` (champion comparison for changed runs) | The check goes red |

Build: `cargo build --release --manifest-path tools/mlguard/Cargo.toml`
Test: `cargo test --release --manifest-path tools/mlguard/Cargo.toml` (7 tests) and
`python tools/mlguard/test_runlog.py`.

## Commands

```
mlguard run <summary.json> [--champion runs/champion.json] [--config mlguard.toml]
mlguard watch <metrics.jsonl> [--once] [--stop-file PATH]
mlguard split --folds runs/<id>/folds.tsv
mlguard split --train train_ids.txt --valid valid_ids.txt
mlguard submission --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
                   --test-dir <dataset/test> [--summary runs/<id>/summary.json]
```
Exit codes: `0` pass (warnings allowed) · `1` at least one FAIL · `2` usage/IO error.
Thresholds live in `tools/mlguard/mlguard.toml` (compiled in as the default). Change them
by PR, with a reason.

## Rules

| Rule | Level | Trigger | Catches |
|---|---|---|---|
| `threshold_source` | FAIL | summary says the threshold was not tuned on `oof`/`holdout` | L3 in-sample tuning |
| `overfit_gap` | FAIL | any fold train F0.5 − valid F0.5 > 0.03 | overfitting |
| `fold_instability` | FAIL | std of fold valid F0.5 > 0.010 | unstable split/model |
| `no_early_stop` / `early_collapse` | WARN | best_iter hits the cap / < 20 | early stopping not wired / broken features |
| `too_good` | FAIL | OOF > 0.995 | leakage |
| `beats_ceiling` | FAIL | OOF > 1.25R/(0.25+R) + 0.005 for blocking recall R | leakage (mathematically impossible otherwise) |
| `banned_feature` | FAIL | a feature name contains `entity_id`, `_id`, `row_idx`, `index`, `label`, `target`, `fold` | ID/label leakage |
| `feature_dominance` | WARN | one feature > 60% of total gain | leak smell |
| `country_gap` | WARN | a country's OOF < overall − 0.05 | shift / weak slice |
| `singleton_rate` | WARN | \|OOF predicted − true singleton rate\| > 0.05 | decision miscalibrated |
| `test_singleton_drift` | FAIL | a test country's predicted singleton rate differs from OOF by > 0.10 | France collapse / over-merge |
| `test_links_drift` | FAIL | test links/entity ÷ OOF links/entity outside [0.6, 1.4] per country | same, label-free |
| `champion_regression` | FAIL | OOF < champion − 0.002 | regressions |
| `diverging_valid` (watch) | FAIL | valid loss rises 5 evals in a row while train loss falls | overfitting live |
| `loss_ratio` (watch) | FAIL | valid/train loss > 1.5 after 10 evals | overfitting live |
| `nan_loss` (watch) | FAIL | non-finite loss | NaN features / divergence |
| `group_leak` (split) | FAIL | an S1 id in more than one fold | entity leakage |
| `split_overlap` (split) | FAIL | train ∩ valid ids ≠ ∅ | leakage |
| `fold_balance` (split) | WARN | largest/smallest fold > 1.25 | skewed CV |
| `header`/`columns`/`format`/`missing_rows` (submission) | FAIL | the spec rules | rejection |
| `not_subset` (submission) | FAIL | a matched id is not in candidates | pipeline bug |
| `partition` (submission) | WARN (configurable FAIL) | an S2/S3 id matched to more than one S1 | GT never does this |

## `summary.json` schema (written by `RunLog.write_summary`)

```json
{
  "run_id": "007_stage2",                       // required
  "oof_score": 0.9123,                          // required, macro F0.5 on OOF
  "threshold_source": "oof",                    // required: oof | holdout
  "folds": [{"fold":0,"train_score":0.93,"valid_score":0.912,"best_iter":840,"max_iter":3000}],
  "blocking_recall": 0.962,
  "per_country": {"US":0.921,"India":0.903},
  "oof_pred_singleton_rate": 0.066, "true_singleton_rate": 0.0558,
  "oof_pred_links_per_entity": 3.21,
  "test_pred_singleton_rate": {"US":0.06,"India":0.07,"France":0.08},   // or computed by `mlguard submission --summary`
  "test_pred_links_per_entity": {"US":3.3,"India":3.1,"France":2.9},
  "feature_importance": {"core_token_sort": 1234.5}
}
```
`train_score` is the fold model's F0.5 on its own training entities, which makes it the
overfitting signal. Computing it on a 20k-entity subsample of the training fold is fine.

`metrics.jsonl` line: `{"fold":0,"iter":120,"train_loss":0.081,"valid_loss":0.087}`; the last
line is `{"event":"end"}`.

## Wiring into the pipeline
`src/runlog.py` is in the branch. The seven hook lines for `run_pipeline.py` are in LLD §9.
They are deliberately **not applied** here, because the 6 h pipeline should change in the
same PR that runs it (Priyanshu's branch). Until then, `mlguard submission` already works
on today's outputs.
