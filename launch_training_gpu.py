import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::112334179737:role/service-role/AmazonSageMaker-ExecutionRole-20260923T221246"

estimator = PyTorch(
    entry_point="train_entry_gpu.py",
    source_dir=".",                       # ships the whole repo (src/, requirements-gpu.txt, etc.)
    dependencies=[],
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.3",              # verify against current SageMaker PyTorch DLC list
    py_version="py311",                   # ditto — check docs if the launch errors on image resolution
    requirements_file="requirements-gpu.txt",
    max_run=10 * 60 * 60,                 # 10h ceiling, per your call — costs nothing if it finishes sooner
)

estimator.fit({
    "training": "s3://amazon-cuda-commandos-2026/krisha/sm-input",       # same dataset as stage 1, reused
    "runinfo": "s3://amazon-cuda-commandos-2026/krisha/sm-runinfo/",     # folds.tsv lives here
})