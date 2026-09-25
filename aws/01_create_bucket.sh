#!/usr/bin/env bash
# Run ONCE per teammate. Creates the team bucket in a fixed region.
set -euo pipefail

TEAM=${1:?usage: 01_create_bucket.sh <team-name> [region]}
REGION=${2:-ap-south-1}
BUCKET="hackathon-${TEAM}"

aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --create-bucket-configuration LocationConstraint="$REGION"

aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false"

echo "created s3://${BUCKET} in ${REGION}"
