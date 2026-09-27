"""Submit the GPU reranker fine-tune to SageMaker.

Runs on your laptop; it packages the repo and starts the job. The thing that
runs *inside* the container is train_entry_gpu.py.

Target is e5 (intfloat/multilingual-e5-small), not laya. We A/B'd them and e5
won on India -- the metric laya's hypothesis was built to win -- at 5.8x less
training time. See docs/T4_FIX.md and docs/KRISHA_GPU_RUNBOOK.md.
"""
import os
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch

ROLE = os.environ.get(
    "AMLC_SM_ROLE",
    "arn:aws:iam::112334179737:role/service-role/AmazonSageMaker-ExecutionRole-20260923T221246",
)
BUCKET = "amazon-cuda-commandos-2026"
PREFIX = os.environ.get("AMLC_SM_PREFIX", "krisha")

estimator = PyTorch(
    entry_point="train_entry_gpu.py",
    source_dir=".",
    dependencies=[],
    role=ROLE,
    instance_type="ml.g4dn.xlarge",       # T4: sm_75, no bf16 -- see train_entry_gpu.py
    instance_count=1,
    framework_version="2.3",              # verify against the current PyTorch DLC list
    py_version="py311",
    # The file lives under src/, and SageMaker resolves this relative to
    # source_dir. The bare filename silently resolves to nothing at the repo
    # root, so no pin installs and the job dies later on an import instead.
    requirements_file="src/requirements-gpu.txt",
    max_run=10 * 60 * 60,
    environment={
        # Turing has no bf16, so autocast falls back to fp16 (max 65504) where
        # the laptop got bf16 (max 3.4e38). Switch to fp32 if the run reports
        # skipped batches -- it cannot overflow.
        "AMLC_AMP": os.environ.get("AMLC_AMP", "fp16"),
        "AMLC_BS": os.environ.get("AMLC_BS", "32"),
        "AMLC_EPOCHS": os.environ.get("AMLC_EPOCHS", "2"),
        "AMLC_ENTITIES": os.environ.get("AMLC_ENTITIES", "40000"),
        "AMLC_AUGMENT": os.environ.get("AMLC_AUGMENT", "0.3"),
        # "AMLC_RR_DIAG": "1",   # prints the text of any non-finite row
    },
)


def warn_about_payload():
    """source_dir='.' tars the WHOLE directory on every submit.

    submissions/ alone is ~38 MB of gzipped TSVs the training job has no use
    for, and anything sitting in models/ (a laya checkpoint is 647 MB) would
    go up too. Harmless, just slow, and easy to not notice.
    """
    big = []
    for name in ("submissions", "models", "output", "runs"):
        d = Path(name)
        if not d.is_dir():
            continue
        mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
        if mb > 5:
            big.append((name, mb))
    if big:
        print("NOTE: source_dir='.' uploads these every submit; the job needs none of them:")
        for name, mb in big:
            print(f"        {name}/  {mb:,.0f} MB")
        print("      Move them aside if launches feel slow.\n")


if __name__ == "__main__":
    warn_about_payload()
    print(f"role  {ROLE}")
    print(f"env   {estimator.environment}\n")
    estimator.fit({
        # same dataset as the CPU stage
        "training": f"s3://{BUCKET}/{PREFIX}/sm-input",
        # folds.tsv: the leak guard. train_entry_gpu.py refuses to start
        # without it rather than training a model whose gain we could not
        # trust.
        "runinfo": f"s3://{BUCKET}/{PREFIX}/sm-runinfo/",
    })
    print("\ndone. Fetch the model artifact and check meta.json:")
    print("  valid_auc      must beat 0.99907")
    print("  valid_logloss  must beat 0.03339")
    print("Anything less is not an improvement on the reranker we already have.")
