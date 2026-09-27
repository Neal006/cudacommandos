"""SageMaker GPU entry point: fine-tune the e5 band reranker on a T4.

WHY e5 AND NOT laya
-------------------
We ran them head to head -- same band, same 514,335 training pairs, same valid
split, same 30k GBDT sample, only the model swapped:

                    baseline     laya       e5
    decision         0.9512     0.9608    0.9607
    India            0.9369     0.9460    0.9469     <- e5 wins
    train time          --      132 min    23 min

laya (mmBERT-base, 322M) lost on India -- the metric its hypothesis was built
to win -- at 5.8x the training cost. Its premise was that e5 handles
native-script names poorly, but the incumbent is intfloat/multilingual-e5-small,
already multilingual, so the A/B was multilingual against multilingual.

The reranker itself is worth having: submission 002 (30k GBDT + e5 reranker)
scored 0.951 on the leaderboard against 001's 0.943 without it, using a
*weaker* GBDT. So fine-tuning e5 is the sensible use of GPU hours.

WHY THIS FILE IS SHAPED FOR A T4
--------------------------------
ml.g4dn.xlarge is a T4: sm_75 Turing, no bfloat16. bf16 carries fp32's
exponent range and cannot overflow; fp16 tops out at 65504. Everything was
developed on an RTX 3050 (sm_86, bf16), so that ceiling was never hit and the
code silently relied on it.

The scoring path length-buckets its batches, which puts the longest sequences
together -- the largest activations the model ever sees. That is why a run
trains fine for 200 random batches and then dies on the first validation pass.
Full explanation in docs/T4_FIX.md.

This entry point reports the GPU it actually got, pins the autocast dtype, and
sizes the batch for 16 GB.
"""
import os
import subprocess
import sys

# SageMaker channels: the dataset (reused from the CPU stage) and folds.tsv
# from the stage-1 run. folds.tsv is the leak guard -- the reranker must not
# train on any entity the GBDT will be evaluated on.
os.environ["AMLC_DATA_DIR"] = os.environ["SM_CHANNEL_TRAINING"]
FOLDS_PATH = os.path.join(os.environ["SM_CHANNEL_RUNINFO"], "folds.tsv")

MODEL_DIR = os.environ["SM_MODEL_DIR"]
OUT_DIR = os.path.join(MODEL_DIR, "rr_e5s")

CODE_DIR = "/opt/ml/code"  # SageMaker unpacks source_dir here

# Overridable as SageMaker hyperparameters or plain env vars.
ENTITIES = os.environ.get("AMLC_ENTITIES", "40000")
BS = os.environ.get("AMLC_BS", "32")
LR = os.environ.get("AMLC_LR", "3e-5")
EPOCHS = os.environ.get("AMLC_EPOCHS", "2")
AUGMENT = os.environ.get("AMLC_AUGMENT", "0.3")


def run(cmd):
    print(f"[train_entry_gpu] running: {' '.join(cmd)}", flush=True)
    # pass the environment through so AMLC_AMP reaches the child
    subprocess.run(cmd, check=True, cwd=CODE_DIR, env={**os.environ})


def gpu_report():
    """Report the GPU and pin the autocast dtype for the child process."""
    try:
        import torch
    except ImportError:
        print("[train_entry_gpu] WARNING: torch not importable here", flush=True)
        return
    if not torch.cuda.is_available():
        print("[train_entry_gpu] WARNING: no CUDA device -- this will be very slow", flush=True)
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported()
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[train_entry_gpu] GPU {name}  sm_{major}{minor}  {gb:.1f} GB  bf16={bf16}", flush=True)
    if not bf16 and "AMLC_AMP" not in os.environ:
        os.environ["AMLC_AMP"] = "fp16"
        print("[train_entry_gpu] no bf16 on this GPU (Turing) -> fp16 autocast, with logit "
              "clamping and non-finite-batch skipping. If the run reports many skipped "
              "batches, rerun with AMLC_AMP=fp32 (slower, cannot overflow).", flush=True)


if __name__ == "__main__":
    gpu_report()
    print(f"[train_entry_gpu] leak guard: {FOLDS_PATH}", flush=True)
    if not os.path.exists(FOLDS_PATH):
        raise SystemExit(
            f"folds.tsv not found at {FOLDS_PATH}.\n"
            f"The GPU stage needs the CPU stage's folds.tsv on the 'runinfo' channel: it is "
            f"what stops the reranker training on entities the GBDT is scored on. Without it "
            f"any gain it shows is untrustworthy, so this refuses rather than guessing."
        )

    run([
        sys.executable, "src/gpu/reranker.py", "train",
        "--exclude", FOLDS_PATH,
        "--entities", ENTITIES,
        "--out", OUT_DIR,
        "--bs", BS,
        "--lr", LR,
        "--epochs", EPOCHS,
        # ft_data.augment_train perturbs TRAIN rows only, never valid ones.
        # e5-small has 22M trainable parameters against a lot of pairs, so a
        # little regularisation is cheap insurance against memorising the band.
        "--augment", AUGMENT,
    ])

    print(f"[train_entry_gpu] done -> {OUT_DIR}", flush=True)
    print("[train_entry_gpu] compare meta.json against our current rr_e5s "
          "(valid_auc 0.99907, valid_logloss 0.03339) before adopting it.", flush=True)
