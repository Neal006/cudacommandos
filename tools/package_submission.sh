#!/usr/bin/env bash
# Build the final submission zip: cudacommandos_submission.zip
#
#   tools/package_submission.sh [output_dir]
#
# output_dir defaults to $AMLC_OUTPUT_DIR, then ./output. It must hold the two
# TSVs; everything else is assembled from the repo.
#
# The layout is fixed by docs/SUBMISSION.md §7 and is graded, so this script
# builds it rather than leaving it to be hand-zipped at 23:00 on the last day:
#
#   cudacommandos_submission.zip
#   |-- output/{matching_results,candidate_pairs}.tsv
#   |-- code/business_entity_resolution/{src/,README.md,requirements.txt}
#   `-- Documentation_template.md
#
# `code/` must be self-contained: someone with only that folder and the
# dataset must be able to regenerate both TSVs. So this copies the whole of
# src/ rather than a hand-picked subset -- a missing module is a silent
# reproducibility failure that nobody notices until a reviewer tries it.
#
# It refuses to build a zip the organisers would reject: the validator has to
# pass, the row count has to be exact, and Documentation_template.md has to be
# filled in. A broken package is worse than a late one, because it looks done.
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM=cudacommandos
OUT=${1:-${AMLC_OUTPUT_DIR:-output}}
MATCHING="$OUT/matching_results.tsv"
CANDIDATE="$OUT/candidate_pairs.tsv"
EXPECTED_ROWS=1732544          # every test Source-1 entity, France included

die() { echo "ERROR: $*" >&2; exit 1; }

for f in "$MATCHING" "$CANDIDATE"; do
  [ -f "$f" ] || die "missing $f -- run src/score_test.py first"
done

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in .venv/Scripts/python.exe .venv/bin/python; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
fi
PY=${PY:-python}

# --- row count, before anything slower. +1 for the header.
for f in "$MATCHING" "$CANDIDATE"; do
  n=$(( $(wc -l < "$f") - 1 ))
  [ "$n" -eq "$EXPECTED_ROWS" ] || die "$f has $n rows, expected $EXPECTED_ROWS"
  echo "rows ok: $f ($n)"
done

# --- the official checker. This is the one that matches the grader.
TEST_DIR=${AMLC_TEST_DIR:-}
if [ -z "$TEST_DIR" ]; then
  TEST_DIR=$("$PY" -c 'import sys; sys.path.insert(0,"src"); import config; print(config.TEST_DIR)')
fi
[ -d "$TEST_DIR" ] || die "test dir not found: $TEST_DIR (set AMLC_TEST_DIR)"
echo "validating against $TEST_DIR ..."
"$PY" validate_submission.py --matching "$MATCHING" --candidate "$CANDIDATE" \
  --test-dir "$TEST_DIR" || die "validate_submission.py failed -- do not upload this"

# --- the graded document must actually be written.
# Anchoring to line start misses the real placeholders: the template writes
# them mid-line ("**Team Name:** [Your Team Name]") and leaves each section as
# a standalone italic prompt ("*Outline your high-level approach.*"). Both
# shapes have to be gone, or the zip looks complete and scores nothing.
ph=$(grep -ciE '\[your |\[list all|\[date\]|\[team|<TODO|TBD' Documentation_template.md || true)
prompts=$(grep -cE '^\*[^*]+\*\s*$' Documentation_template.md || true)
if [ "$ph" -gt 0 ] || [ "$prompts" -gt 0 ]; then
  die "Documentation_template.md is unfilled ($ph placeholders, $prompts template prompts) -- it is graded; see docs/SUBMISSION.md §7"
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/${TEAM}_submission"
CODE="$ROOT/code/business_entity_resolution"
mkdir -p "$ROOT/output" "$CODE"

cp "$MATCHING" "$CANDIDATE" "$ROOT/output/"
cp Documentation_template.md "$ROOT/"
cp requirements.txt "$CODE/"
cp validate_submission.py "$CODE/"

# whole src/, minus caches -- see the note above about self-containment
( cd src && find . -name '__pycache__' -prune -o -type f -print0 \
    | tar --null -cf - -T - ) | ( mkdir -p "$CODE/src" && tar -xf - -C "$CODE/src" )

# README.md in code/ is the reproduction guide, not the repo's front page.
cat > "$CODE/README.md" <<'MD'
# Business Entity Resolution — cudacommandos

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
the competition dataset. No external data, no network calls, no API lookups.

## Environment

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
```

Python 3.11-3.13. `requirements.txt` pins `pandas<3` — pandas 3.x changes
behaviour this code depends on.

## Data

Point `AMLC_DATA_DIR` at the folder holding `dataset/`:

```
$AMLC_DATA_DIR/dataset/train/train_source{1,2,3}.tsv
$AMLC_DATA_DIR/dataset/train/train_ground_truth.tsv
$AMLC_DATA_DIR/dataset/test/test_source{1,2,3}.tsv
```

Outputs go to `AMLC_OUTPUT_DIR` (default `./output`).

## Run

```bash
export AMLC_DATA_DIR=/path/to/data
export AMLC_OUTPUT_DIR=/path/to/output

python src/run_v2.py --sample 150000
```

That trains and writes both TSVs. To score the test set from an
already-trained run without retraining:

```bash
python src/score_test.py --run runs/<run_id> --chunk 2000000
```

## Pipeline

1. **Ingest** (`ingest.py`) — polars, Arrow-backed strings.
2. **Normalize** (`normalize.py`) — casefold, NFKC, accent folding,
   transliteration of 9 Indic scripts, legal-suffix and address-abbreviation
   tables. Two speeds: a cheap blocking blob for all 10M records, heavy forms
   only for records that survive blocking.
3. **Blocking** (`blocking.py`) — word-level TF-IDF per country, document
   frequency pruned (`max_df=0.01`, `min_df=3`), chunked sparse top-k.
   K=30, measured recall ceiling 0.9498.
4. **Features** (`features_v2.py`) — ~46 pair features: rapidfuzz string
   similarities, token overlaps, transliteration and skeleton keys, address
   house numbers, IDF-weighted name overlap.
5. **Stage 1** (`run_v2.py`) — LightGBM, GroupKFold by Source-1 entity.
6. **Stage 2** (`stage2.py`) — stacker over out-of-fold stage-1 scores plus
   entity-shape, competition and peer-agreement features.
7. **Decision** (`decide.py`) — isotonic calibration, then per-entity
   expected-F0.5 selection with soft conflict resolution.
8. **Write** (`data.py`) — enforces row count, duplicate and subset rules.

## Memory

The test phase scores 52M candidate pairs. Features are built in chunks cut
only where `s1_id` changes, so every per-entity statistic matches a whole-frame
run; the four `cand_id`-grouped stage-2 columns are computed once globally
because a candidate can be claimed from different chunks. Lower `--chunk` if
you have less than ~12 GB free.

## Tests

```bash
python tests/test_decide.py
python tests/test_features_v2.py
python tests/test_chunk_equiv.py     # needs the train candidate cache
```

## Licences

LightGBM (MIT), rapidfuzz (MIT), scikit-learn (BSD-3), pandas (BSD-3),
polars (MIT), anyascii (ISC). No pretrained model is used in the final
pipeline.
MD

ZIP="${TEAM}_submission.zip"
rm -f "$ZIP"
( cd "$STAGE" && zip -qr "$OLDPWD/$ZIP" "${TEAM}_submission" )

echo
echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
unzip -l "$ZIP" | tail -n +4 | head -20
echo
echo "checklist (docs/SUBMISSION.md §9): rows exact, validator PASS, docs filled."
