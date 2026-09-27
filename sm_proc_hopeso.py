"""Processing job: hopeso frames -> run_v4 -> score_test, i.e. one whole submission.

Runs on a 192-vCPU box (ml.m7i.48xlarge / ml.c7i.48xlarge, Processing quota 2
each -- the Training quota for both is 0, which is the trap: the two pools are
separate and only Processing has these sizes).

Three env-selected variants, so one script covers every box in the fleet:
    AMLC_RR=none   hopeso alone                      -> submission A, the safe one
    AMLC_RR=e5     hopeso + the e5-small reranker    -> independent of the bge box
    AMLC_RR=bge    hopeso + bge-reranker-v2-m3       -> needs the rr channel

Everything is written to the output channel: matching_results.tsv,
candidate_pairs.tsv and runs/<id>/summary.json, whose `holdout_score` is what
decides which submission gets uploaded.

AMLC_DATA_DIR must be writable -- config.py puts interim/ inside it and hopeso
caches its union frames there -- so the read-only dataset channel is symlinked
into a writable work dir rather than used directly.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IN = Path("/opt/ml/processing/input")
OUT = Path("/opt/ml/processing/output")
WORK = Path("/opt/ml/processing/work")
CODE = Path("/opt/ml/processing/code") if Path("/opt/ml/processing/code").is_dir() else Path.cwd()

RUN_ID = os.environ.get("AMLC_RUN_ID", "013_hopeso")
SAMPLE = os.environ.get("AMLC_SAMPLE", "150000")
HOLDOUT = os.environ.get("AMLC_HOLDOUT", "50000")
ROUNDS = os.environ.get("AMLC_ROUNDS", "4000")
CHUNK = os.environ.get("AMLC_CHUNK", "8000000")
RR = os.environ.get("AMLC_RR", "none")
BAND = os.environ.get("AMLC_BAND", "0.05 0.95").split()

T0 = time.time()


def log(m):
    print(f"[hopeso-box {(time.time()-T0)/60:6.1f}m] {m}", flush=True)


def sh(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(CODE), env={**os.environ})


def setup():
    (WORK / "interim").mkdir(parents=True, exist_ok=True)
    ds = IN / "dataset"
    link = WORK / "dataset"
    if not link.exists():
        # symlink keeps the 2.4 GB where it landed instead of copying it
        link.symlink_to(ds if (ds / "train").is_dir() else IN, target_is_directory=True)
    os.environ["AMLC_DATA_DIR"] = str(WORK)
    os.environ["AMLC_OUTPUT_DIR"] = str(OUT)

    n = os.cpu_count() or 1
    # blocking's sparse matmul and rapidfuzz scale with cores; the pool itself
    # does not, past a point -- every worker re-imports pandas/numpy, and on a
    # fork platform they share pages but still contend for memory bandwidth.
    os.environ.setdefault("AMLC_WORKERS", str(min(64, max(1, n - 1))))
    os.environ.setdefault("AMLC_BLOCK_THREADS", str(n))
    try:
        import psutil
        gb = psutil.virtual_memory().total / 1e9
    except Exception:
        gb = float("nan")
    log(f"{n} vCPU, {gb:.0f} GB RAM, AMLC_WORKERS={os.environ['AMLC_WORKERS']}, "
        f"variant RR={RR}, chunk={CHUNK}")

    # any candidate/stats caches we shipped, so blocking is skipped
    cache = IN / "interim"
    if cache.is_dir():
        for f in cache.iterdir():
            shutil.copy2(f, WORK / "interim" / f.name)
            log(f"staged cache {f.name} ({f.stat().st_size/1e6:.0f} MB)")


def rerank_dir():
    if RR == "none":
        return None
    ch = IN / "rerank"
    if not ch.is_dir():
        sys.exit(f"[hopeso-box] AMLC_RR={RR} but no rerank channel mounted")
    # the channel may mount the model dir itself or its parent
    if (ch / "meta.json").exists():
        return ch
    inner = [d for d in ch.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    if len(inner) != 1:
        sys.exit(f"[hopeso-box] cannot find a single model dir under {ch}: {list(ch.iterdir())}")
    return inner[0]


def main():
    setup()
    rr = rerank_dir()
    if rr:
        log(f"reranker: {rr}  band {BAND}")

    # ---- hopeso union frames. Both splits, before anything else needs them.
    # No --n: run_v4 loads the train frame with --frame (default None -> "nall"),
    # so the tagged frame has to be the full 2.2M-entity one or load_frame misses
    # the cache name and exits. --sample only picks the training rows out of it.
    sh([sys.executable, "src/hopeso.py", "build", "--split", "train"])
    sh([sys.executable, "src/hopeso.py", "build", "--split", "test"])

    # ---- train stages 1-2 on the hopeso frame, decide on the test-like holdout
    cmd = [sys.executable, "src/run_v4.py", "--sample", SAMPLE, "--holdout", HOLDOUT,
           "--decide-on", "holdout", "--rounds", ROUNDS, "--chunk", CHUNK,
           "--cands-tag", "hopeso", "--run-id", RUN_ID, "--train-only"]
    if rr:
        cmd += ["--rerank", str(rr), "--band", *BAND]
    sh(cmd)

    # ---- score test with the very models that produced that holdout number
    cmd = [sys.executable, "src/score_test.py", "--run", f"runs/{RUN_ID}",
           "--chunk", CHUNK, "--cands-tag", "hopeso"]
    if rr:
        cmd += ["--rerank", str(rr), "--band", *BAND]
    sh(cmd)

    src = CODE / "runs" / RUN_ID
    if src.is_dir():
        shutil.copytree(src, OUT / "runs" / RUN_ID, dirs_exist_ok=True)
        log(f"run artifacts -> {OUT / 'runs' / RUN_ID}")
    for f in sorted(OUT.glob("*.tsv")):
        log(f"output {f.name}  {f.stat().st_size/1e6:.1f} MB")
    log("done")


if __name__ == "__main__":
    main()
