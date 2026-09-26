import sagemaker
from sagemaker.sklearn.estimator import SKLearn

role = "arn:aws:iam::112334179737:role/service-role/AmazonSageMaker-ExecutionRole-20260923T221246"

estimator = SKLearn(
    entry_point="train_entry.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={},
    environment={
        "AMLC_DATA_DIR": "/opt/ml/input/data/training",
    },
    max_run=3*60*60,   # train-only should land well under 1.5h per repo's own timing table
)

estimator.fit({"training": "s3://amazon-cuda-commandos-2026/krisha/sm-input"})