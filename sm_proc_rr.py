"""Processing job: fine-tune BAAI/bge-reranker-v2-m3 as the band reranker, on a GPU.

WHY A GPU AND NOT c7i.48xlarge
------------------------------
PIPELINE_98 puts this on a 192-vCPU Sapphire Rapids box using AMX bf16, because
it was written believing the account had no GPU training quota. It does:
ml.g5.2xlarge (A10G, 24 GB) is available and costs $1.82/h against $10.28/h for
the c7i. The A10G is Ampere sm_86, so torch.cuda.is_bf16_supported() is True and
reranker._dev() picks bf16 on its own -- no AMLC_CPU_BF16, and none of the fp16
overflow that put NaNs in the T4 run (T4 is Turing sm_75, 5-bit exponent).

NO DATASET CHANNEL
------------------
make_training_pairs() returns early on a cached
    interim/rr_pairs_n{entities}_s{seed}_ex{len(exclude)}.parquet
so shipping that 57 MB file means this box never loads the 2.4 GB dataset --
which matters, because g5.2xlarge has 32 GB of RAM. The cache key includes
len(exclude_ids), so folds.tsv must be the same 150,000-entity one used to
build it, or the name misses and the job tries to block the whole train split.

GATE
----
bench first. The rate decides --entities, exactly as PIPELINE_98 specifies, and
a rate under the floor aborts rather than spending the box. Then the trained
valid AUC must beat e5-small's 0.99907 or the model is still written but marked
`beats_e5: false` in meta, and the caller must not promote it.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# This runs as a TRAINING job, not a Processing one: every GPU instance has
# `... for processing job usage` = 0 while `... for training job usage` >= 1.
# The CPU 48xlarge boxes are the exact opposite. Two separate pools, inverted.
# Paths are resolved for either mode so the script does not care which it got.
_TRAIN_MODE = "SM_MODEL_DIR" in os.environ

if _TRAIN_MODE:
    OUT = Path(os.environ.get("SM_OUTPUT_DATA_DIR") or os.environ["SM_MODEL_DIR"])
    WORK = Path("/opt/ml/work")
    CODE = Path("/opt/ml/code")
else:
    OUT = Path("/opt/ml/processing/output")
    WORK = Path("/opt/ml/processing/work")
    CODE = Path("/opt/ml/processing/code") if Path("/opt/ml/processing/code").is_dir() else Path.cwd()
IN = Path("/opt/ml/processing/input")


def channel(name, sub):
    """Training mounts each channel at SM_CHANNEL_<NAME>; Processing puts them
    all under a shared input dir."""
    env = os.environ.get(f"SM_CHANNEL_{name.upper()}")
    return Path(env) if env else (IN / sub)

E5_AUC = 0.99907   # models/rr_e5s/meta.json -- the bar bge has to clear

# interim/ has to live somewhere writable that config.py will point INTERIM at
(WORK / "interim").mkdir(parents=True, exist_ok=True)
os.environ["AMLC_DATA_DIR"] = str(WORK)
os.environ.setdefault("AMLC_WORKERS", str(max(1, (os.cpu_count() or 2) - 1)))


def sh(cmd, **kw):
    print(f"[rr] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, cwd=str(CODE), env={**os.environ}, **kw)



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


def main():
    preflight(True)
    # the pre-built pair cache, so we never touch the dataset
    src = channel("rrpairs", "rrpairs")
    if src.is_dir():
        for f in src.iterdir():
            shutil.copy2(f, WORK / "interim" / f.name)
            print(f"[rr] staged {f.name} ({f.stat().st_size/1e6:.0f} MB)", flush=True)

    folds = next(channel("runinfo", "runinfo").glob("folds.tsv"), None)
    if folds is None:
        sys.exit("[rr] no folds.tsv on the runinfo channel -- cannot honour the leak guard")

    import torch
    print(f"[rr] torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  "
          f"bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}",
          flush=True)

    model = os.environ.get("AMLC_RR_BASE", "BAAI/bge-reranker-v2-m3")

    # ---- bench: speed only, untrained head
    out = subprocess.run([sys.executable, "src/gpu/reranker.py", "bench", "--model", model],
                         cwd=str(CODE), env={**os.environ}, capture_output=True, text=True)
    print(out.stdout, out.stderr, flush=True)
    out.check_returncode()
    rate = float([l for l in out.stdout.splitlines() if "pairs/s" in l][0]
                 .split("inference:")[1].split("pairs/s")[0].strip().replace(",", ""))

    # PIPELINE_98's table, applied rather than eyeballed
    if rate >= 500:
        entities, band = 40000, (0.05, 0.95)
    elif rate >= 150:
        entities, band = 20000, (0.1, 0.9)
    else:
        (OUT / "VERDICT.json").write_text(json.dumps(
            {"bench_pairs_per_s": rate, "decision": "abort", "reason": "under 150 pairs/s"}, indent=2))
        sys.exit(f"[rr] {rate:.0f} pairs/s is under the 150 floor -- aborting, submission A stands")
    print(f"[rr] bench {rate:,.0f} pairs/s -> --entities {entities}, band {band}", flush=True)

    mdir = WORK / "rr_bge"
    sh([sys.executable, "src/gpu/reranker.py", "train",
        "--base", model, "--exclude", str(folds), "--entities", str(entities),
        "--bs", os.environ.get("AMLC_RR_BS", "32"), "--lr", os.environ.get("AMLC_RR_LR", "2e-5"),
        "--out", str(mdir), "--run-dir", str(OUT / "rr_bge_run")])

    meta = json.loads((mdir / "meta.json").read_text()) if (mdir / "meta.json").exists() else {}
    auc = float(meta.get("valid_auc") or 0.0)
    meta.update(bench_pairs_per_s=rate, entities=entities, band=list(band),
                e5_auc=E5_AUC, beats_e5=auc > E5_AUC)
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))

    shutil.copytree(mdir, OUT / "rr_bge", dirs_exist_ok=True)
    (OUT / "VERDICT.json").write_text(json.dumps(
        {"bench_pairs_per_s": rate, "entities": entities, "band": list(band),
         "valid_auc": auc, "e5_auc": E5_AUC, "beats_e5": auc > E5_AUC,
         "decision": "promote" if auc > E5_AUC else "do-not-promote"}, indent=2))
    print(f"[rr] valid_auc {auc:.5f} vs e5 {E5_AUC} -> "
          f"{'PROMOTE' if auc > E5_AUC else 'DO NOT PROMOTE'}", flush=True)


if __name__ == "__main__":
    main()
