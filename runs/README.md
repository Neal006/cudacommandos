# runs/

One folder per training run, named `<nnn>_<what>`. The number matches the
experiment id in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md), which
stays the append-only narrative; this folder holds the machine-written
artifacts that back it.

## What lands here

| File | Written by | Committed? |
|---|---|---|
| `summary.json` | `runlog.RunLog.finish` | yes — the run's headline numbers, gated by mlguard in CI |
| `metrics.jsonl` | `runlog` lgb callback | yes — per-iteration train/valid loss, what `mlguard watch` reads |
| `decision_table.csv` | `decide.tune` | yes — every (assign, select, thr, miss) combination and its OOF F0.5 |
| `pipeline.log` | `run_v2.log` | yes — timings, the only record of how long each stage actually took |
| `mlguard_watch.log` | `train_guarded.sh` | yes — PASS/FAIL and which rule fired |
| `folds.tsv` | `runlog` | **no** — `*.tsv` is gitignored; regenerate or pull from S3 |
| `model.pkl` | `run_v2.main` | **no** — 35 MB; share via `./aws/s3.sh` |
| `MLGUARD_STOP` | `mlguard watch` | **no** — a live control file, not a result |

A run that is interesting enough to explain also gets a `CONTEXT.md`: what it
was for, what it produced, what broke, and what someone picking it up needs
to know. See [`007_v2_full/CONTEXT.md`](007_v2_full/CONTEXT.md).

## Reproducing one

```bash
PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh <run_id> --sample 150000
```

Caches in `<AMLC_DATA_DIR>/interim/` are keyed by blocking parameters, so a
rerun with the same `TOP_K` / `BLOCK_MAX_DF` / `BLOCK_MIN_DF` reuses the
candidates instead of rebuilding them. That is the difference between a
70-minute rerun and a 3-hour one.

## Runs so far

| Run | Sample | OOF F0.5 | Note |
|---|---|---|---|
| `003_v2_30k` | 30k | 0.9512 | first v2 pipeline end to end |
| `004_v2_30k_fixstop` | 30k | 0.9512 | same, with the guard-stop rollback fixed |
| `007_v2_full` | 150k | **0.9532** | first full sample, first test phase — see its CONTEXT.md |

`smoke_2k_v2/` is a 2k-entity smoke test used to check the pipeline runs at
all before committing hours to it. Kept out of git; rebuild with
`--sample 2000` in about a minute.
