# Business Entity Resolution: reproducible pipeline

Given S1 (reference) and S2/S3 (noisy) business records, predict for every S1 entity which S2/S3
records refer to the same business. Metric: macro F0.5 over S1 entities, singletons included.

## Layout

```
code/business_entity_resolution/
├── README.md, requirements.txt
├── tests/test_normalize.py      # parser checks on real examples from the data
└── src/
    ├── config.py         paths (ER_DATA / ER_WORK / ER_OUT env overrides), seeds, K values
    ├── io_utils.py       TSV -> parquet, integer ids, ground-truth pairs
    ├── splits.py         hidden set H (20% of train S1) + 5 GBDT folds over the query set Q
    ├── translit.py       Indic -> Latin: token dictionary learnt from train pairs + ITRANS fallback
    ├── normalize.py      names (junk prefixes, aliases, domains, leet, legal suffixes) and addresses
    │                     (component parser: state/region, locality, postcode, house number, unit, street)
    ├── blocking.py       sparse TF-IDF top-k per (country, state): name / name+address / address-only,
    │                     reverse pass, exact keys; union with per-blocker scores and ranks
    ├── features.py       ~75 pairwise features (numba set overlaps, rapidfuzz, house numbers, name frequency)
    ├── gbdt.py           LightGBM GroupKFold helper (out-of-fold predictions)
    ├── prune.py          streaming prune model -> top-20 per S1 (+ per-source and best-S1 keeps)
    │                     (= candidate_pairs.tsv); recall diagnostics and K / min-p grid
    ├── stage1.py         pairwise LightGBM on the pruned set
    ├── context.py        context features (within-S1, competition, S2/S3 agreement, twins, CE)
    │                     -> stage-2 LightGBM -> isotonic -> decision tuning
    ├── decide.py         one-to-one assignment + per-S1 expected-F0.5 subset (numba Monte-Carlo)
    ├── evaluate.py       exact macro F0.5 + breakdowns
    ├── crossencoder.py   optional GPU cross-encoder: export / train / infer / France pseudo-labels
    ├── write_output.py   output TSVs + official validator
    └── run_pipeline.py   orchestrator
```

## Environment

CPU side. The pipeline runs on one Linux box with enough RAM; we used an AWS
`c7i.48xlarge` (192 vCPU, 384 GB, Ubuntu 22.04, Python 3.11) for the full-data
stages, and a Windows laptop for packaging, validation and the output checks.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Measured peak through `prune` is 41 GB resident, so 64 GB is a sensible floor;
the large instance buys wall-clock, not headroom. `ER_JOBS` controls parallelism and
defaults to `cpu_count() - 2`. We set it to **64**: the per-worker copies in the
normalize pool make 192 workers a memory risk with no speed gain, and it also
fixes LightGBM's `num_threads`, which is worth pinning if you want runs to line
up with each other (see the determinism note below).

GPU side (cross-encoder only; design target 1× A10G 24 GB): `torch` (CUDA build),
`transformers`, `polars`, `pyarrow`, `scikit-learn`.

## Run

Data is expected at `<repo>/dataset/{train,test}/*.tsv` (override with `ER_DATA`). Intermediate files
go to `<repo>/work/` (`ER_WORK`) and outputs to `<repo>/output/` (`ER_OUT`).

```bash
cd code/business_entity_resolution/src
ER_JOBS=64 ../../../.venv/bin/python run_pipeline.py     # full CPU pipeline -> output/*.tsv
```
Stages resume with `run_pipeline.py --from prune`; earlier stages reuse their parquet
in `work/`.

Wall-clock on the `c7i.48xlarge` at `ER_JOBS=64`, reading the data from local NVMe:

| stage | time | |
|---|---|---|
| io | 6 s | TSV → parquet, 12.53M train and 11.70M test records |
| splits | < 1 s | 441,315 hidden S1, 1,765,506 in the query set |
| normalize | 88 s | transliteration dictionary, name and address parsing |
| sets | 57 s | token and n-gram sets for the feature stage |
| blocking | 23 min | three sparse passes + reverse pass + exact keys, both splits |
| prune | 61 min | 5 fold models (≈155 s each), then the streaming scoring pass |
| stage1 | ~58 min | pairwise LightGBM on the pruned set |
| context | ~19 min | stage-2 LightGBM, isotonic calibration, decision tuning |
| output | 20 s | TSVs + validator |

`io` through `prune` is **86 minutes measured on this instance**, peaking at
41 GB resident. The three model stages are marked `~` because they were timed on
comparable hardware rather than on this box; budget about three hours end to end.

Measured out-of-fold on 1.77M held-out train S1, with 20% of S1 hidden as orphans:
macro F0.5 **0.98654** (US 0.98731, India 0.98537, singletons 0.98733). On the
competition leaderboard this output scored **0.985**. The test output passes
`validate_submission.py --check-ids`.

### Optional cross-encoder (adds the `ce*` features to the stage-2 model)
```bash
python crossencoder.py export                                  # CPU box: work/ce/*.parquet
# copy work/ce/ to the GPU box, then there:
python crossencoder.py train --model intfloat/multilingual-e5-small --out ce_e5s
python crossencoder.py infer --model ce_e5s --band band_train.parquet --out ce_train.parquet
python crossencoder.py infer --model ce_e5s --band band_test.parquet  --out ce_test.parquet
# copy ce_train.parquet / ce_test.parquet into work/, then back on the CPU box:
python run_pipeline.py --from context
```
France pseudo-labels (after a first full run): `python crossencoder.py pseudo`. Then fine-tune on the GPU with
`train --model ce_e5s --extra pseudo_fr.parquet --epochs 1 --out ce_e5s_fr`, re-infer, and rerun `--from context`.

The whole round trip is scripted in `gpu_steps.sh` (`GPU=user@host bash gpu_steps.sh`).

### Prune settings and diagnostics
The prune rule and model are configured in `config.py`. Each value can be overridden with an environment variable:

| Variable | Default | Effect |
|---|---|---|
| `ER_PRUNE_K` | 20 | keep a pair if it is in its S1's top K ... |
| `ER_PRUNE_MIN_P` | 0.005 | ... and its prune score is at least this |
| `ER_PRUNE_MIN_P_COUNTRY` | `{}` | per-country score floor, e.g. `'{"France": 0.002}'` |
| `ER_PRUNE_K_PER_SRC` | 10 | also keep the top k from S2 and, separately, from S3 (0 = off) |
| `ER_PRUNE_KEEP_CAND_BEST` | 1 | add back each S2/S3 record's best S1 (score ≥ `ER_PRUNE_CAND_BEST_MIN_P`, default 0.001) |
| `ER_PRUNE_EXTRA_FEATS` | 1 | prune model also gets name flags, name frequency, house-number distance and postcode |
| `ER_PRUNE_HARD_NEG_RANK` | 5 | training keeps every negative ranked ≤ this by any blocker (0 = off) |

Setting the last four to 0 reproduces the original prune step exactly.

After a run, `prune.py` prints (and `python prune.py report` reprints without rescoring):
- recall on Q after blocking and after pruning;
- the true pairs pruning lost, and why (score below the floor, rank below K, or both), broken down by country,
  source, record type (Indic / domain / alias / empty address) and union size;
- a grid of Q recall and pairs per S1 for K ∈ {5…40} × min-p ∈ {0.02…0.0005} (per-S1 rule only),
  saved as `work/prune_grid.csv`. Use it to choose `ER_PRUNE_K` / `ER_PRUNE_MIN_P` without rerunning;
- test pairs per S1 by country, to check that France is not cut harder than the training countries.

To compare settings end-to-end: `run_pipeline.py --from prune` with the variables set (earlier stages are reused).

### Leaderboard probes
`write_output.py --params probe.json --tag NAME` writes `output/matching_results_NAME.tsv` using
per-country decision parameters, for example
`{"default": {"method": "ef", "a": 1.0, "b": 0.0, "delta": 0.1}, "France": {"b": -0.5}}`.

## What is in `src/`
`src/` holds the sixteen modules `run_pipeline.py` executes and nothing else. Probe and
experiment scripts written during development are not part of the pipeline and are left
out of this package, so every file here is on the path from the raw TSVs to the two
output files.

## Models and licences
LightGBM (MIT). Optional cross-encoder: `intfloat/multilingual-e5-small` (MIT, 118M parameters).
Transliteration fallback: `indic-transliteration` (MIT). No model exceeds 8B parameters.
No external data, API, geocoder or registry is used: every signal, including the
transliteration dictionary, is derived from the provided TSVs.

## `output/candidate_pairs.tsv`

`prune.py` writes the candidate set that every later model scores, and that file
is what ships as `candidate_pairs.tsv`. Measured on the run above:

| | |
|---|---|
| blocking union | 101.9 candidates per S1, recall 0.99069 on Q |
| after pruning | 13,371,630 test pairs, **7.72 per S1**, recall 0.98862 on Q |
| true pairs lost to pruning | India 0.21%, US 0.21% |
| test pairs per S1 by country | France 10.25, India 7.80, US 6.63 |

Pruning costs 0.2% of the recall the union had, evenly across both labelled
countries. France — which has no training labels at all — keeps more candidates
per entity than either, so the per-country floors are not quietly starving it.
`prune.py report` reprints these without rescoring, and `work/prune_grid.csv`
holds the K × min-p grid the settings were chosen from.

### Determinism, and how the two output files relate

The package's two outputs come from two executions: the model stages that
produced `matching_results.tsv`, and the `io → prune` execution timed above that
produced `candidate_pairs.tsv`.

Runs of this pipeline are not bit-identical to each other. `ER_JOBS` sets
LightGBM's `num_threads`, which fixes the order histogram bins are summed in, so
the prune model's scores shift slightly with thread count and CPU architecture,
and pairs sitting on the top-K / min-p boundary can fall either side. Across the
two executions here that affected 2,474 of 5,757,784 matched pairs — 0.043%.

Those 2,474 are included in `candidate_pairs.tsv`. `write_output.py` asserts that
every match lies inside the candidate set and refuses to write otherwise, so a
matched pair is by construction a pruned pair; including them keeps the shipped
files consistent with that invariant. Pin `ER_JOBS` to reproduce a given run.

The file holds 13,374,104 pairs over 1,732,544 entities, 7.72 per entity, with
2,546 entities keeping no candidate. Every matched id lies inside its entity's
candidate list, and `validate_submission.py --check-ids` passes on the pair.

