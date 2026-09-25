# Design EDA

`design_eda.py` reproduces the numbers in `docs/master-plan/ANALYSIS.md` §1
(~40 s, polars, ~8 GB RAM):

    AMLC_DATASET_DIR=/path/to/dataset python tools/eda/design_eda.py

`dataset/` must contain `train/` and `test/` exactly as released. The script-coverage,
token-overlap and house-number numbers (ANALYSIS.md §1.4–1.6) were follow-up queries of
the same shape; each is described next to its number so it can be re-derived.
