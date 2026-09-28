#!/usr/bin/env bash
# Cross-encoder round trip between the CPU box and the GPU box and the A10G box.
# Usage:   GPU=user@a10g-host  bash gpu_steps.sh
# Needs:   a completed CPU run through `stage1` (work/s1_train.parquet, work/s1_test.parquet) on the CPU box,
#          and on the GPU box a Python env with torch (CUDA), transformers, polars, pyarrow, scikit-learn.
set -euo pipefail
: "${GPU:?set GPU=user@host of the A10G box}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
REMOTE_DIR="${REMOTE_DIR:-~/er}"
MODEL="${MODEL:-intfloat/multilingual-e5-small}"

# 1) CPU box: export H training pairs + ambiguous bands for Q and test
( cd "$HERE/src" && "$PY" crossencoder.py export )

# 2) ship code + data (only the columns the GPU needs)
ssh "$GPU" "mkdir -p $REMOTE_DIR/ce"
rsync -av "$HERE/src/crossencoder.py" "$GPU:$REMOTE_DIR/"
rsync -av "$REPO/work/ce/" "$GPU:$REMOTE_DIR/ce/"

# 3) GPU: train on H, score both bands (ER_CE_DIR tells the script where the parquet files are)
ssh "$GPU" "cd $REMOTE_DIR && export ER_CE_DIR=$REMOTE_DIR/ce && \
  python crossencoder.py train --model $MODEL --out ce_model --epochs 2 --bs 128 --lr 5e-5 && \
  python crossencoder.py infer --model ce_model --band band_train.parquet --out ce/ce_train.parquet && \
  python crossencoder.py infer --model ce_model --band band_test.parquet  --out ce/ce_test.parquet"

# 4) bring scores back and rerun the context stage + output on the CPU box
rsync -av "$GPU:$REMOTE_DIR/ce/ce_train.parquet" "$GPU:$REMOTE_DIR/ce/ce_test.parquet" "$REPO/work/"
( cd "$HERE/src" && "$PY" run_pipeline.py --from context )
