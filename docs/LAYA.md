# Laya band reranker — test guide (branch `laya`)

**What:** [laya](https://github.com/NandhaKishorM/laya) (`pip install laya`, Apache-2.0) is a
multilingual encoder with a small "decision head" that answers typed questions in one forward pass.
We use its `laya-multilingual` checkpoint (mmBERT-base, 322M params, 100+ languages including the
9 Indic scripts in India S2) as a **drop-in challenger for the e5-small band reranker**.

**Why:** the gap between our 0.9532 OOF and the ~0.99 blocking ceiling is mostly **India**, and
e5-small handles native-script names poorly. mmBERT was trained on them.

**Not a new stage.** Nothing else in the pipeline changes:

```
stage-1 LightGBM -> pairs with p1 in [0.2, 0.8] (~1.2%) -> RERANKER -> feature `rr` -> stage-2 LightGBM -> decision layer
                                                           ^ e5 (reranker.py)  or  laya (laya_rr.py)
```

`run_v2.py --rerank <dir>` picks the backend from `<dir>/meta.json` `kind` (`e5` | `laya`).
Training pairs, record text (`reranker.serialize`), the `entities.txt` leak guard and the band are
the same for both, so an A/B comparison changes only the model.

## Commands

```bash
pip install laya==0.3.20                     # no other dependency changes (torch>=2.0, transformers>=4.48)
python tests/test_laya_rr.py                 # data-free unit checks, no GPU, no download
python src/gpu/laya_rr.py fetch              # -> models/laya_ml_base (640 MB, Windows-safe: no symlinks)
python tests/smoke_laya_rr.py                # GPU end-to-end on synthetic records, ~1 min
python src/gpu/laya_rr.py bench --train      # pairs/s + peak VRAM on YOUR GPU

# the real experiment (needs the dataset; same --exclude as the e5 reranker)
python src/gpu/laya_rr.py train --exclude runs/<id>/folds.tsv --entities 40000 --out models/rr_laya
python src/run_v2.py --sample 30000 --train-only --rerank models/rr_laya        # OOF with laya
python src/run_v2.py --sample 30000 --train-only --rerank models/rr_e5s         # same, e5: the A/B
```

Knobs: `--train-layers N` (top N of mmBERT's 22 encoder layers + the head train; default 8 fits
4 GB; `-1` trains everything except the embedding table, for a 16 GB T4/Kaggle), `--bs`, `--lr`
(3e-5), `--epochs`.

## Measured (RTX 3050 Laptop 4 GB, bf16, synthetic records)

| | laya (top 8 layers) | e5-small (EXPERIMENTS 004) |
|---|---|---|
| inference | 369 pairs/s → ~620k test band pairs in **~28 min** | 1,870 pairs/s (~6 min) |
| training | 153 pairs/s at bs 32 | 390 pairs/s |
| trainable / peak VRAM | 55M / 2.75 GiB | 22M / — |
| synthetic smoke | valid AUC 0.996, held-out AUC 0.997 | — |
| zero-shot (no fine-tune) | **useless**: an unrelated pair scored 0.90 "same" | — |

Profiling: 97% of inference time is the GPU forward (rows average 80 tokens: a 32-token question
prefix + both records), so it is compute-bound. More speed means a narrower `--band`, not code changes.

## What to report back

1. `meta.json` of `models/rr_laya` vs `models/rr_e5s`: `valid_auc`, `valid_logloss`. Both use the
   same entity-grouped valid split, so they are directly comparable.
2. The run_v2 log lines `stage 2 (global threshold)` and `decision layer best`, plus `per_country`
   from `summary.json`, for both rerankers. **India** is the number that matters.
3. Any `laya loaded on cpu, not cuda` error. It means the GPU ran out of memory; lower `--train-layers`.

## Caveats (read before trusting a gain)

- **Leak audit still pending, for both rerankers:** S2/S3 records can repeat across the reranker's
  training pairs and the GBDT sample. `entities.txt` only guards S1 entities. Do not quote a laya
  gain as final until that audit is done (same status as the e5 +0.009).
- The checkpoint weights are saved in bf16 (as laya ships them), so reloaded scores differ from
  the in-memory valid metrics by rounding only.
- `models/` is gitignored (640 MB per checkpoint). Share trained dirs through S3, not git.
- A trained dir is also a normal laya checkpoint: `laya.load("models/rr_laya")` works, with the
  fitted temperature in `rl_agent_config.json`.
