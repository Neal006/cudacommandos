"""SageMaker wrapper for src/gpu/laya_rr.py — GPU fine-tuning entry point.

Wraps the fetch + train subcommands via subprocess (same reason as the CPU
train_entry.py: SageMaker's automatic hyperparameter->CLI conversion doesn't
know about laya_rr.py's argparse subcommands).
"""
import os
import subprocess
import sys

# SageMaker mounts the "training" channel (same dataset S3 path reused from
# stage 1) at this path, and the "runinfo" channel (folds.tsv) here:
os.environ["AMLC_DATA_DIR"] = os.environ["SM_CHANNEL_TRAINING"]
FOLDS_PATH = os.path.join(os.environ["SM_CHANNEL_RUNINFO"], "folds.tsv")

MODEL_DIR = os.environ["SM_MODEL_DIR"]
OUT_DIR = os.path.join(MODEL_DIR, "rr_laya")

CODE_DIR = "/opt/ml/code"  # SageMaker unpacks source_dir here


def run(cmd):
    print(f"[train_entry_gpu] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=CODE_DIR)


if __name__ == "__main__":
    # 1. fetch the base laya checkpoint (~640MB, idempotent)
    run([sys.executable, "src/gpu/laya_rr.py", "fetch"])

    # 2. the actual fine-tune. --train-layers -1 per your call; entities=40000
    #    matches docs/LAYA.md's example command.
    run([
        sys.executable, "src/gpu/laya_rr.py", "train",
        "--exclude", FOLDS_PATH,
        "--entities", "40000",
        "--out", OUT_DIR,
        "--train-layers", "-1",
        "--lr", "1e-5",
    ])

    print("[train_entry_gpu] done.", flush=True)