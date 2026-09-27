import sagemaker
from sagemaker.sklearn.estimator import SKLearn

role = "arn:aws:iam::112334179737:role/service-role/AmazonSageMaker-ExecutionRole-20260923T221246"

estimator = SKLearn(
    entry_point="rerank_entry.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    framework_version="1.2-1",   # match whatever your working CPU launch_training.py already used
    max_run=4 * 60 * 60,          # 4h ceiling — full run incl. test blocking, budget more than train-only
)

estimator.fit({
    "training": "s3://amazon-cuda-commandos-2026/krisha/sm-input",
})