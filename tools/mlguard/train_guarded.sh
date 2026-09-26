#!/usr/bin/env bash
# Train with mlguard watching in the background, then gate the finished run.
#   tools/mlguard/train_guarded.sh <run_id> [run_pipeline.py args...]
#   CHAMPION=runs/champion.json tools/mlguard/train_guarded.sh 004_lgbm_translit
#   PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh 005_v2 --sample 150000
# Works in Git Bash (Windows) and on Linux/AWS.
set -euo pipefail
RUN=${1:?usage: train_guarded.sh <run_id> [pipeline args]}; shift
DIR="runs/$RUN"; mkdir -p "$DIR"; rm -f "$DIR/MLGUARD_STOP"
cargo build --release -q --manifest-path tools/mlguard/Cargo.toml
MLG=tools/mlguard/target/release/mlguard

# Prefer the repo venv. A bare `python` picks up the system interpreter,
# which on this box is pandas 3.x -- the version requirements.txt pins
# against -- and has none of the v2 deps, so the run dies in an import or,
# worse, on a pandas 3 behaviour change. Override with PYTHON=...
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for cand in .venv/Scripts/python.exe .venv/bin/python; do
    [ -x "$cand" ] && { PY="$cand"; break; }
  done
fi
PY=${PY:-python}
echo "interpreter: $PY ($("$PY" -c 'import pandas,sys;print("py",sys.version.split()[0],"pandas",pandas.__version__)'))"

export MLGUARD_RUN_DIR="$DIR"
"$MLG" watch "$DIR/metrics.jsonl" > "$DIR/mlguard_watch.log" 2>&1 &
WATCH=$!

set +e
"$PY" "${PIPELINE:-src/run_pipeline.py}" "$@" 2>&1 | tee "$DIR/pipeline.log"
RC=${PIPESTATUS[0]}
set -e
# a crashed run never writes the end event; write it so the watcher exits
grep -q '"event": "end"' "$DIR/metrics.jsonl" 2>/dev/null || echo '{"event": "end"}' >> "$DIR/metrics.jsonl"
wait "$WATCH" || echo "!! mlguard watch flagged training: $(cat "$DIR/MLGUARD_STOP" 2>/dev/null)"
[ "$RC" -eq 0 ] || { echo "pipeline exited $RC"; exit "$RC"; }

[ -f "$DIR/folds.tsv" ] && "$MLG" split --folds "$DIR/folds.tsv"
"$MLG" run "$DIR/summary.json" ${CHAMPION:+--champion "$CHAMPION"}
if [ -f output/matching_results.tsv ] && [ -n "${AMLC_TEST_DIR:-}" ]; then
  "$MLG" submission --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir "$AMLC_TEST_DIR" --summary "$DIR/summary.json"
fi
