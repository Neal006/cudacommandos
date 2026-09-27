"""Submit the PIPELINE_98 fleet as SageMaker Processing jobs.

    python launch98.py --box a --dry-run
    python launch98.py --box a          # hopeso alone          -> submission A
    python launch98.py --box rr         # bge fine-tune on GPU  -> models/rr_bge
    python launch98.py --box e5         # hopeso + e5 reranker  -> submission B'
    python launch98.py --box bge --rr-s3 s3://.../rr_bge/   -> submission B

WHY PROCESSING AND NOT TRAINING
-------------------------------
`ml.m7i.48xlarge for training job usage` is 0 in this account, and so is the
c7i. `... for processing job usage` is 2 for both. The two quota pools are
separate, which is the single fact PIPELINE_98's fleet depends on.

WHY THE RERANKER BOX IS A GPU
-----------------------------
PIPELINE_98 puts bge-reranker-v2-m3 on c7i.48xlarge with AMX bf16 because it
was written assuming no GPU quota. ml.g5.2xlarge (A10G) is available, is bf16
native, and costs $1.82/h against the c7i's $10.28/h. Strictly better.
"""
import argparse
import os

BUCKET = "sagemaker-ap-south-1-654479364872"
PREFIX = os.environ.get("AMLC_SM_PREFIX", "amlc")
ROLE = os.environ.get("AMLC_SM_ROLE",
    "arn:aws:iam::654479364872:role/service-role/AmazonSageMaker-ExecutionRole-20260925T010466")
S3 = f"s3://{BUCKET}/{PREFIX}"

# hourly on-demand, ap-south-1, from the pricing API on 2026-09-27
# Container 2.5.1/py311 matches the local known-good stack (torch 2.5.1,
# transformers 5.17, numpy 2.1). On the 2.3 container, pip resolves
# transformers>=4.40 to 5.x, whose torch-availability check rejects torch 2.3
# and then reports "AutoModelForSequenceClassification requires the PyTorch
# library but it was not found" -- on a PyTorch image. That cost a 2-hour
# r5.24xlarge run and the first GPU box.
PRICE = {"ml.m7i.48xlarge": 12.217, "ml.c7i.48xlarge": 10.282, "ml.g5.2xlarge": 1.819}

BOXES = {
    # name: (instance, entry, env, extra inputs, est hours)
    "a":   ("ml.m7i.48xlarge", "sm_proc_hopeso.py",
            {"AMLC_RR": "none", "AMLC_RUN_ID": "013_hopeso_a"}, [], 2.5),
    # c7i rather than a second m7i: same 192 vCPU for $2/h less, 384 GB is
    # plenty at chunk 8M, and it leaves both m7i slots free for the bge box.
    "e5":  ("ml.c7i.48xlarge", "sm_proc_hopeso.py",
            {"AMLC_RR": "e5", "AMLC_RUN_ID": "014_hopeso_e5", "AMLC_BAND": "0.2 0.8"},
            [("rerank", f"{S3}/rr_e5s/")], 3.0),
    "bge": ("ml.c7i.48xlarge", "sm_proc_hopeso.py",
            {"AMLC_RR": "bge", "AMLC_RUN_ID": "015_hopeso_bge", "AMLC_BAND": "0.05 0.95"},
            [("rerank", None)], 3.0),
    "rr":  ("ml.g5.2xlarge", "sm_proc_rr.py", {}, [], 1.5),
}


def launch_training(a, inst, entry, env, hours, sess):
    """GPU boxes go through the Training API: their Processing quota is 0 while
    their Training quota is >= 1. The CPU 48xlarge boxes are the exact
    opposite. Two separate pools, inverted."""
    from sagemaker.pytorch import PyTorch
    est = PyTorch(
        entry_point=entry, source_dir=".", role=ROLE,
        instance_type=inst, instance_count=1,
        framework_version="2.5.1", py_version="py311",
        volume_size=a.volume, max_run=int(a.max_hours * 3600),
        base_job_name="amlc-" + a.box, sagemaker_session=sess,
        environment={**env, "AMLC_MEMBER": "priyanshu"},
        output_path=S3 + "/out/amlc-" + a.box + "-p98/",
    )
    est.fit({"rrpairs": S3 + "/rr-pairs/", "runinfo": S3 + "/sm-runinfo/"},
            wait=False, logs=False)
    print("submitted (training): " + est.latest_training_job.job_name)


def main(a):
    inst, entry, env, extra, hours = BOXES[a.box]
    import boto3
    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.pytorch.processing import PyTorchProcessor

    # The default credential chain has no region here, and Session() refuses
    # without one. Pin both explicitly rather than relying on the environment.
    boto = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "amlc"),
                         region_name=os.environ.get("AWS_REGION", "ap-south-1"))
    sess = sagemaker.Session(boto_session=boto, default_bucket=BUCKET)

    inputs = [ProcessingInput(source=f"{S3}/sm-runinfo/", destination="/opt/ml/processing/input/runinfo",
                              input_name="runinfo")]
    if entry == "sm_proc_hopeso.py":
        inputs.append(ProcessingInput(source=f"{S3}/sm-input/dataset/",
                                      destination="/opt/ml/processing/input/dataset",
                                      input_name="dataset"))
        # hopeso.build does NOT block: it unions its passes onto an existing
        # base frame and exits if that frame is missing. Shipping the caches
        # also keeps every box on byte-identical candidates, so their holdout
        # scores are actually comparable.
        inputs.append(ProcessingInput(source=f"{S3}/interim/",
                                      destination="/opt/ml/processing/input/interim",
                                      input_name="interim"))
    else:
        inputs.append(ProcessingInput(source=f"{S3}/rr-pairs/",
                                      destination="/opt/ml/processing/input/rrpairs",
                                      input_name="rrpairs"))
    for name, src in extra:
        src = src or a.rr_s3
        if not src:
            raise SystemExit(f"box {a.box} needs --rr-s3 for the {name} channel")
        inputs.append(ProcessingInput(source=src, destination=f"/opt/ml/processing/input/{name}",
                                      input_name=name))

    job = f"amlc-{a.box}-{os.environ.get('AMLC_TAG', 'p98')}"
    print(f"box       {a.box}")
    print(f"instance  {inst}  (${PRICE[inst]}/h, est {hours} h -> ~${PRICE[inst]*hours:.0f})")
    print(f"entry     {entry}")
    print(f"env       {env}")
    for i in inputs:
        print(f"input     {i.input_name:8} {i.source}")
    print(f"output    {S3}/out/{job}/")
    if a.dry_run:
        print("\n--dry-run: nothing submitted")
        return

    if a.box == "rr":
        return launch_training(a, inst, entry, env, hours, sess)

    proc = PyTorchProcessor(
        framework_version="2.5.1", py_version="py311", role=ROLE,
        instance_type=inst, instance_count=1,
        volume_size_in_gb=a.volume, max_runtime_in_seconds=int(a.max_hours * 3600),
        base_job_name=job, env={**env, "AMLC_MEMBER": "priyanshu"},
        sagemaker_session=sess,
    )
    proc.run(code=entry, source_dir=".", inputs=inputs,
             outputs=[ProcessingOutput(source="/opt/ml/processing/output",
                                       destination=f"{S3}/out/{job}/", output_name="out")],
             wait=False, logs=False)
    print(f"\nsubmitted: {proc.latest_job.job_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, choices=list(BOXES))
    ap.add_argument("--rr-s3", default=None, help="s3 uri of the trained bge reranker dir")
    ap.add_argument("--volume", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
