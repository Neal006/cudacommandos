# Fine-tuning options for the band reranker — runbook (branch `finetune`)

Everything here trains the **band reranker**, the model that re-scores the ~1.2% of candidate pairs
stage 1 is unsure about (p1 in 0.2–0.8) and hands stage 2 the feature `rr`. Nothing else in the
pipeline changes, and `run_v2.py --rerank <dir>` reads every model built here as-is.
Builds on the laya reranker (docs/LAYA.md). Target box: Priyanshu's RTX 3050 (4–6 GB VRAM,
~7–9 GB free RAM). **Never run these next to CPU blocking or test featurization.**

| # | Option | Flag / command | What it changes | Cost on a 3050 |
|---|---|---|---|---|
| 1 | Band-matched pairs | `python src/gpu/ft_data.py band --run runs/<id>` then `--pairs <parquet>` | trains on the pairs stage 1 actually finds hard | one blocking + featurize pass over the pool (RAM-heavy, like a run_v2 train phase) |
| 2 | In-data augmentation | `--augment 0.5` | +50% train rows: native-only / Latin-only name, dropped name or address token, shuffled address, stripped legal form | +50% train time |
| 3 | LoRA over all layers | `--lora 16` (laya only) | adapters on every mmBERT layer instead of unfreezing the top 8 | 67 pairs/s, 1.9 GiB (vs 153 pairs/s, 2.75 GiB) |
| 4 | Listwise loss | `--listwise 1.0` (laya only, needs option-1 pairs) | the S1 candidates of one S2/S3 record compete in a softmax with a "none" slot, added to BCE | ~free |
| 5 | Distillation laya → e5 | `python src/gpu/distill.py --teacher models/rr_laya --pairs ... --out models/rr_e5_kd` | e5-small learns laya's scores: laya quality at e5 speed | teacher scoring + one e5 training (~10–20 min) |

Not built: RL (RLCD/GRPO/DPO: with exact 0/1 labels the proper-scoring RL optimum is BCE), embedder
fine-tuning (killed by the blocking-ceiling measurement), and a QLoRA small LLM (3–10x slower,
usually weaker than a fine-tuned encoder on pair classification). See the discussion in PR notes.

## Commands (in order)

```bash
pip install -r requirements.txt                     # adds laya 0.3.20; peft is already required
python tests/test_finetune.py                       # data-free, CPU, ~1 min
python src/gpu/laya_rr.py fetch                     # laya base -> models/laya_ml_base (once)
python tests/smoke_finetune.py                      # GPU end to end on synthetic records, ~5 min

# 0. a finished GBDT run is the input to everything: runs/<id>/{folds.tsv,model.pkl}
# 1. band-matched pairs for entities OUTSIDE that run's sample (cached in INTERIM)
python src/gpu/ft_data.py band --run runs/<id> --entities 150000
#    -> prints "band pairs -> ft_band_<id>_n150000_s7_b0.05-0.95_o0.05.parquet: N of M pairs ..."
P=$AMLC_DATA_DIR/interim/ft_band_<id>_n150000_s7_b0.05-0.95_o0.05.parquet

# the A/B grid: one change at a time against the laya baseline
python src/gpu/laya_rr.py train --exclude runs/<id>/folds.tsv --out models/rr_laya              # baseline
python src/gpu/laya_rr.py train --pairs $P --out models/rr_laya_band                          # +1
python src/gpu/laya_rr.py train --pairs $P --augment 0.5 --out models/rr_laya_band_aug        # +1+2
python src/gpu/laya_rr.py train --pairs $P --augment 0.5 --lora 16 --out models/rr_laya_lora  # +1+2+3
python src/gpu/laya_rr.py train --pairs $P --augment 0.5 --listwise 1.0 --out models/rr_laya_lw  # +1+2+4
python src/gpu/reranker.py train --pairs $P --augment 0.5 --out models/rr_e5_band_aug         # e5 gets 1+2 too

# quick comparison on ONE frozen held-out file (another seed = entities no model trained on);
# each model's own meta.json valid split is a different draw and is NOT comparable across runs
python src/gpu/ft_data.py band --run runs/<id> --entities 30000 --seed 9          # -> H (held-out)
python src/gpu/ft_data.py eval --pairs $H --models models/rr_laya models/rr_laya_band models/rr_laya_band_aug \
       models/rr_laya_lora models/rr_laya_lw models/rr_e5_band_aug     # auc, logloss, auc in the 0.2-0.8 band

# the decision: each one in run_v2, same sample and folds as the GBDT run it was built against
python src/run_v2.py --sample 30000 --train-only --rerank models/<dir>

# 5. only after laya wins: distil it into e5 on pairs the teacher did NOT train on
python src/gpu/ft_data.py band --run runs/<id> --entities 150000 --seed 8
python src/gpu/distill.py --teacher models/<best laya> --pairs <seed-8 parquet> --out models/rr_e5_kd
```

`--band 0.05 0.95` / `--outside 0.05` control option 1's training band. It is wider than run_v2's
0.2–0.8 serving band because the serving band alone is ~1.2% of pairs, too few to train on.
`--group-by s1_id` switches the listwise groups to entities (ranking within an entity) instead of
records competing for one entity.

## Measured (RTX 3050 Laptop 4 GB, bf16, synthetic records)

| setup | train pairs/s | peak VRAM | smoke held-out AUC |
|---|---|---|---|
| laya, top-8 layers (default) | 153 | 2.75 GiB | 0.997 (easy synthetic set) |
| laya, LoRA r=16 + checkpointing | 67 | 1.88 GiB | 0.937 (harder set: competing look-alikes, Devanagari) |
| laya, LoRA r=16, no checkpointing, bs 32 | 29 (spills to shared RAM) | 3.81 GiB | — |
| e5 distilled from the LoRA laya (on a separate set) | 634 | small | 0.936 (teacher 0.937) |

The two smoke sets differ, so compare options on **real data** only (next section).
On a 6 GB card LoRA without checkpointing would fit, but checkpointing costs only ~10% and is always on.

## What to report back (per trained dir)

1. The `ft_data.py eval` table (one row per model, same held-out rows): `auc_band` is the number
   that matters, since only band pairs reach stage 2.
2. From each `run_v2 --train-only --rerank` run: `stage 2 (global threshold)`, `decision layer best`
   and `per_country` (India first). This is the real decision; the eval table only ranks candidates.
3. `meta.json` `seconds` and the option fields, and the `band pairs -> ...` line from option 1
   (how many pairs, positives and % in band).

## Caveats

- **Option 1 has not run on real data** (the dataset drive was not mounted when this was built).
  Its selection logic is unit-tested; the orchestration (`band_pairs`) reuses run_v2's own
  featurize/predict and fails loudly if the run's `feat1` columns are not produced by this code.
  It uses the run's averaged fold models, i.e. the test-time stage 1. Stage 2's OOF rows are selected
  by per-fold OOF p1 instead; the gap is unmeasured, so before trusting option 1 compare how many of
  the GBDT sample's pairs fall in the band under OOF p1 vs the averaged models (label-free check).
- **Guards that fail loudly:** `distill.py` drops pairs from entities the teacher trained on (and stops
  if none are left); `ft_data.py eval` refuses a model that trained on any evaluation entity;
  `--listwise` refuses pairs without a group key (mixing `--pairs` files with and without `cand_id`);
  a listwise group larger than `--bs` becomes its own batch and is reported at start.
- **Leak audit still pending** for every reranker: S2/S3 records can repeat across reranker pairs and
  the GBDT sample. `entities.txt` guards S1 only; distillation writes the union with the teacher's.
- Augmentation assumes label preservation (Ditto): dropping a token rarely flips a match. Check
  `--augment` on India specifically; if it hurts, `native_only`/`latin_only` are the ops to keep.
- LoRA defaults to lr 2e-4 (vs 3e-5); pass `--lr` to override. The saved checkpoint is merged, so it
  scores at normal laya speed (369 pairs/s), not LoRA speed.
