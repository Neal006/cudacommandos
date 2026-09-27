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
    # A model trained by a SageMaker TRAINING job arrives as output.tar.gz, not
    # a directory -- the GPU boxes have to be training jobs because their
    # processing quota is 0. Unpack in place rather than round-tripping 1.6 GB
    # through a laptop to re-upload it as loose files.
    tars = sorted(ch.glob("*.tar.gz"))
    if tars and not any(ch.rglob("meta.json")):
        import tarfile
        dest = WORK / "rerank_unpacked"
        dest.mkdir(parents=True, exist_ok=True)
        for t in tars:
            log(f"unpacking {t.name} ({t.stat().st_size/1e6:.0f} MB)")
            with tarfile.open(t) as tf:
                tf.extractall(dest)
        ch = dest
    metas = sorted(ch.rglob("meta.json"))
    if len(metas) == 1:
        return metas[0].parent
    if (ch / "meta.json").exists():
        return ch
    inner = [d for d in ch.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    if len(inner) != 1:
        sys.exit(f"[hopeso-box] cannot find a single model dir under {ch}: {list(ch.iterdir())}")
    return inner[0]



def preflight(need_torch=True):
    """Fail in the first minute, not the third hour.

    Both of the first cloud failures were the same thing: transformers could be
    imported fine, so nothing complained until the code actually asked for a
    torch-backed class -- which on the reranker path is after blocking, stage 1
    and most of stage 2. Check it up front instead.
    """
    import importlib.metadata as md
    for pkg in ("torch", "transformers", "numpy", "pandas", "lightgbm"):
        try:
            print(f"[preflight] {pkg} {md.version(pkg)}", flush=True)
        except Exception:
            print(f"[preflight] {pkg} MISSING", flush=True)
    if not need_torch:
        return
    import torch  # noqa: F401
    from transformers.utils import is_torch_available
    if not is_torch_available():
        raise SystemExit(
            "[preflight] transformers cannot see torch -- this is the 2.3-container "
            "failure: transformers 5.x rejects torch 2.3. Use framework_version 2.5.1 "
            "or pin transformers<5.")
    from transformers import AutoModelForSequenceClassification as _A
    _ = _A  # touching the lazy class is what actually raised before
    print("[preflight] torch + transformers OK", flush=True)


def band_gate(rr):
    """A big cross-encoder is only affordable here if the CPU keeps up.

    With hopeso the frames are 84.2M train / 66.4M test pairs, and the 0.05-0.95
    band is 2.4% of them -- about 3.6M pairs to score across the run. The A10G
    benched bge-reranker-v2-m3 at 1,081 pairs/s (55 min for that volume). This
    box has no GPU, so measure it here and refuse rather than find out four
    hours in: under 500 pairs/s the scoring alone outlasts the deadline.
    """
    need = 3_600_000
    floor = float(os.environ.get("AMLC_RR_FLOOR", "500"))
    out = subprocess.run([sys.executable, "src/gpu/reranker.py", "bench", "--model", str(rr)],
                         cwd=str(CODE), env={**os.environ}, capture_output=True, text=True)
    print(out.stdout, out.stderr, flush=True)
    out.check_returncode()
    rate = float([l for l in out.stdout.splitlines() if "pairs/s" in l][0]
                 .split("inference:")[1].split("pairs/s")[0].strip().replace(",", ""))
    hours = need / rate / 3600
    log(f"band gate: {rate:,.0f} pairs/s -> {need:,} band pairs in {hours:.1f} h")
    if rate < floor:
        sys.exit(f"[hopeso-box] {rate:.0f} pairs/s is under the {floor:.0f} floor "
                 f"({hours:.1f} h of scoring). Aborting; the e5 and A submissions stand.")


def main():
    preflight(RR != "none")
    setup()
    rr = rerank_dir()
    if rr:
        log(f"reranker: {rr}  band {BAND}")
        if RR == "bge":
            band_gate(rr)

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
