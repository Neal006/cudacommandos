#!/usr/bin/env bash
# Team bucket helper — amazon-cuda-commandos-2026 (ap-south-1, cross-account).
#
#   ./aws/s3.sh ls                     list the whole bucket
#   ./aws/s3.sh ls krina               list one member's prefix
#   ./aws/s3.sh push <file> [subdir]   upload into YOUR prefix
#   ./aws/s3.sh pull <member> <path>   download someone else's file
#   ./aws/s3.sh share-cache            upload the candidate cache (see below)
#   ./aws/s3.sh get-cache <member>     pull someone's candidate cache
#
# Everyone writes under their own prefix so concurrent uploads cannot
# overwrite each other.
#
# Auth: uses the `aws login` browser session on the amlc profile (12h,
# renewable 90 days). Do NOT create static access keys for this — a long-lived
# secret in ~/.aws/credentials is strictly worse than rotating SSO creds.
# If a command fails with "session has expired", run:
#     aws login --region ap-south-1 --profile amlc
set -euo pipefail

BUCKET="${AMLC_BUCKET:-amazon-cuda-commandos-2026}"
REGION="${AWS_REGION:-ap-south-1}"
PROFILE="${AWS_PROFILE:-amlc}"
ME="${AMLC_MEMBER:-priyanshu}"

# Windows installs the CLI outside PATH more often than not.
AWS_BIN="$(command -v aws || echo "/c/Users/$USER/AppData/Local/Programs/Amazon/AWSCLIV2/aws.exe")"
aws_() { "$AWS_BIN" --profile "$PROFILE" --region "$REGION" "$@"; }

# The candidate cache is the artifact actually worth sharing: test blocking is
# 4-5 hours of compute and the parquet reproduces it exactly. One person runs
# it, everyone else pulls the result.
CACHE_DIR="${AMLC_DATA_DIR:-$HOME/amlc_data}/interim"

usage() { sed -n '2,20p' "$0"; exit 1; }
[ $# -ge 1 ] || usage

case "$1" in
  ls)
    if [ $# -ge 2 ]; then aws_ s3 ls "s3://$BUCKET/$2/" --human-readable --recursive
    else aws_ s3 ls "s3://$BUCKET/" --human-readable; fi
    ;;
  push)
    [ $# -ge 2 ] || usage
    dest="s3://$BUCKET/$ME/${3:+$3/}"
    aws_ s3 cp "$2" "$dest"
    echo "-> $dest$(basename "$2")"
    ;;
  pull)
    [ $# -ge 3 ] || usage
    aws_ s3 cp "s3://$BUCKET/$2/$3" "./$(basename "$3")"
    ;;
  share-cache)
    shopt -s nullglob
    files=("$CACHE_DIR"/cands_*.parquet)
    if [ ${#files[@]} -eq 0 ]; then
      echo "no candidate cache in $CACHE_DIR — run the pipeline first" >&2
      exit 1
    fi
    for f in "${files[@]}"; do
      echo "uploading $(basename "$f") ($(du -h "$f" | cut -f1))"
      aws_ s3 cp "$f" "s3://$BUCKET/$ME/cache/"
    done
    ;;
  get-cache)
    [ $# -ge 2 ] || usage
    mkdir -p "$CACHE_DIR"
    aws_ s3 cp "s3://$BUCKET/$2/cache/" "$CACHE_DIR/" --recursive --exclude "*" --include "cands_*.parquet"
    echo "-> $CACHE_DIR"
    ;;
  *) usage ;;
esac
