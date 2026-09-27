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
    # pass the environment through so AMLC_AMP reaches the child
    subprocess.run(cmd, check=True, cwd=CODE_DIR, env={**os.environ})


def gpu_report():
    """Print what we are actually on, and pick a safe autocast dtype.

    This is the difference between the boxes. An RTX 3050 is sm_86 and has
    bf16, whose exponent range matches fp32 -- activations cannot overflow it.
    A T4 (ml.g4dn.xlarge) is sm_75: no bf16, so the code falls back to fp16,
    which tops out at 65504. Long sequences in a length-bucketed validation
    batch are exactly where that ceiling gets hit, which is why a run that is
    fine on Ampere produces NaN on Turing at the first validation.

    The clamp in laya_rr._logit_diff makes fp16 survivable. AMLC_AMP=fp32 is
    the belt-and-braces option: slower, more memory, cannot overflow.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("[train_entry_gpu] WARNING: no CUDA device", flush=True)
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[train_entry_gpu] GPU {name} sm_{cap[0]}{cap[1]} "
          f"{total:.1f} GB  bf16={bf16}", flush=True)
    if not bf16 and "AMLC_AMP" not in os.environ:
        os.environ["AMLC_AMP"] = "fp16"
        print("[train_entry_gpu] no bf16 on this GPU -> fp16 autocast with logit "
              "clamping. If you still see non-finite losses, rerun with "
              "AMLC_AMP=fp32 (slower, cannot overflow).", flush=True)


if __name__ == "__main__":
    gpu_report()

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
        # A T4 is 16 GB, but --train-layers -1 makes all 322M parameters
        # trainable: fp32 master weights + grads + two AdamW moments is ~5 GB
        # before activations. Keep the batch modest; raise it only if the run
        # reports headroom.
        "--bs", os.environ.get("AMLC_BS", "16"),
    ])

    print("[train_entry_gpu] done.", flush=True)