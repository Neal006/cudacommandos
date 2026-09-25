#!/usr/bin/env bash
# Push a model artifact to the shared bucket and print a 24h presigned URL.
# Presigned URL is the zero-IAM fallback if the bucket policy fights you.
set -euo pipefail

ARTIFACT=${1:?usage: 03_share_artifact.sh <local-file> <team-name>}
TEAM=${2:?usage: 03_share_artifact.sh <local-file> <team-name>}
BUCKET="hackathon-${TEAM}"
KEY="models/$(basename "$ARTIFACT")"

aws s3 cp "$ARTIFACT" "s3://${BUCKET}/${KEY}"
aws s3 presign "s3://${BUCKET}/${KEY}" --expires-in 86400
