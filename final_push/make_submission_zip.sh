#!/usr/bin/env bash
# Build <team>_submission.zip in the structure required by the README of the challenge.
# Usage: bash make_submission_zip.sh TEAM_NAME
set -euo pipefail
TEAM="${1:?usage: bash make_submission_zip.sh TEAM_NAME}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/output" "$STAGE/code"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
rsync -a --exclude "__pycache__" "$ROOT/code/business_entity_resolution" "$STAGE/code/"
cp "$ROOT/Documentation_template.md" "$STAGE/"
( cd "$STAGE" && zip -qr "$ROOT/${TEAM}_submission.zip" output code Documentation_template.md )
rm -rf "$STAGE"
echo "wrote $ROOT/${TEAM}_submission.zip"; unzip -l "$ROOT/${TEAM}_submission.zip" | tail -n +1 | head -40
