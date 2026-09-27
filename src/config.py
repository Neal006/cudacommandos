"""Paths and knobs for the Business Entity Resolution challenge.

Point DATA_DIR at wherever you unpacked `student_resource`. Everything below
assumes the layout the problem statement describes:

    <DATA_DIR>/dataset/train/train_source1.tsv
    <DATA_DIR>/dataset/train/train_source2.tsv
    <DATA_DIR>/dataset/train/train_source3.tsv
    <DATA_DIR>/dataset/train/train_ground_truth.tsv
    <DATA_DIR>/dataset/test/test_source{1,2,3}.tsv
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Repo lives in OneDrive, so the dataset stays out of the repo tree. D: and E:
# are removable here and vanish when unplugged, which silently breaks every
# path below — so fall back to a local folder whenever the preferred drive is
# not mounted rather than failing deep inside a run.
def _pick_data_dir() -> Path:
    env = os.environ.get("AMLC_DATA_DIR")
    if env:
        return Path(env)
    # Test the candidate FOLDER, not just its drive letter. Checking the
    # drive root selects D:\amlc_data the moment anything is mounted as D:,
    # even when the dataset is somewhere else entirely -- which is exactly
    # what happened here: an external disk mounted as D:, an empty
    # D:\amlc_data on it, and 2.4 GB of data sitting in the home fallback.
    # Every path below then pointed at nothing, and the run failed several
    # stages in, reading as a data problem rather than a path problem.
    for candidate in (Path(r"D:\amlc_data"), Path(r"E:\amlc_data")):
        if (candidate / "dataset").is_dir():
            return candidate
    # Dataset is on neither removable drive: prefer one that is at least
    # mounted and could receive it, otherwise the home fallback.
    for candidate in (Path(r"D:\amlc_data"), Path(r"E:\amlc_data")):
        if candidate.is_dir():
            return candidate
    return Path.home() / "amlc_data"


DATA_DIR = _pick_data_dir()
TRAIN_DIR = DATA_DIR / "dataset" / "train"
TEST_DIR = DATA_DIR / "dataset" / "test"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# The repo lives in OneDrive, so a full test run's outputs land in a synced
# folder on the system drive -- candidate_pairs.tsv alone is ~1 GB at 1.73M
# entities x K=30. AMLC_OUTPUT_DIR moves them off it. The default is
# unchanged, so nothing differs for anyone who does not set it.
OUTPUT = Path(os.environ.get("AMLC_OUTPUT_DIR") or (ROOT / "output"))
INTERIM = DATA_DIR / "interim"
for _d in (OUTPUT, INTERIM):
    _d.mkdir(parents=True, exist_ok=True)

MATCHING_RESULTS = OUTPUT / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT / "candidate_pairs.tsv"

# --- columns (fixed by the problem statement) ---
ID = "entity_id"
NAME = "business_name"
ADDR = "business_address"
COUNTRY = "country"
GT_S1 = "source1_entity_id"
GT_MATCH = "matched_entity_ids"
OUT_CAND = "candidate_entity_ids"

SEED = 42

# --- blocking ---
# Candidates kept per Source-1 entity. Ground truth averages 3.46 true links
# per entity (mode 3), so 50 leaves plenty of headroom. Raise for recall, lower
# for speed; check the recall ceiling before tuning anything downstream, since
# no matcher can beat what blocking hands it.
#
# Measured on train (150k sample, 10.3M index), recall ceiling by K:
#   K=5 0.8340 | K=10 0.9187 | K=20 0.9411 | K=30 0.9499 | K=50 0.9586
#
# Recall is not the score. With F_0.5 weighting precision 2x, a perfect
# matcher at recall R scores 1.25R/(0.25+R) — so those ceilings translate to
# F_0.5 of 0.983 / 0.988 / 0.990 / 0.992. Going 30 -> 50 buys +0.002 of
# ceiling and costs ~35M extra pairs to featurize on test. Not worth it:
# blocking is not the bottleneck, matcher precision is.
TOP_K = 60

# Drop tokens appearing in more than this FRACTION of records. This is the
# lever that keeps the sparse similarity product tractable at 10M records:
# common tokens ("road", "pvt", "restaurant") produce huge posting lists and
# almost no discriminative signal. Lower = faster and leaner, but prunes more
# signal. Raise it if blocking recall is short and you have the RAM.
BLOCK_MAX_DF = 0.01

# Source-1 rows per chunk of the sparse matmul. Peak memory scales with this
# times the average number of records sharing a token. Drop it if you OOM.
BLOCK_CHUNK = 2000

# Documents sampled to FIT the TF-IDF vocabulary. Document-frequency estimates
# are a statistical question — a 1M-document sample gives essentially the same
# answer as all 10.5M while fitting ~10x faster and holding a far smaller
# vocabulary dict. Every document is still transformed; only the fit samples.
BLOCK_FIT_SAMPLE = 1_000_000

# Minimum document frequency. Tokens appearing once or twice across 10M records
# are almost always typos, and they dominate vocabulary size.
BLOCK_MIN_DF = 3

# Records transformed per batch when building the index matrix. Bounds peak
# memory during the transform, which is otherwise a single huge allocation.
BLOCK_INDEX_BATCH = 500_000

# Drop business_name / business_address after computing the blocking blob.
# Those two columns are ~4GB of Python strings across 10.3M records and are
# not needed again until the feature stage, which re-reads just the rows that
# survived blocking. Set False only if you have RAM to spare.
BLOCK_DROP_TEXT = True

# Train the matcher on a random subsample of Source-1 entities. 2.2M entities
# is far more than a pairwise matcher needs, and the full set will not fit in
# the ~10GB of RAM available here. Blocking and inference still run over the
# FULL test set — only matcher training is subsampled. None = use everything.
TRAIN_SAMPLE = 150_000

# Restrict candidate generation to records sharing the same country string.
# Big reduction in comparisons and almost certainly correct — but it is an
# assumption. If blocking recall looks low, set this False and re-measure.
# NOTE: country is an OPEN set. Test contains France, which never appears in
# training. Nothing here may hard-code {US, India}.
BLOCK_WITHIN_COUNTRY = True

# Threads for sparse_dot_topn's top-k product (-1 = all cores but one). Results
# are identical to the scipy path; only speed changes.
BLOCK_THREADS = int(os.environ.get("AMLC_BLOCK_THREADS", "-1"))

# Worker PROCESSES for the multiprocessing pools in ingest.py and
# features_v2.py. This is not the same knob as BLOCK_THREADS, which is
# threads inside one process and costs no extra memory.
#
# Windows and macOS start workers with `spawn`, so each one is a fresh
# interpreter that re-imports pandas, polars, numpy, scipy and rapidfuzz
# -- roughly 250-400 MB resident before it does any work. cpu_count() also
# reports logical cores (16 here for 10 physical), so the old
# `cpu_count() - 1` default asked for 15 of those and the pool died in
# MemoryError during startup, before the first chunk was processed. Linux
# forks instead and shares those pages, which is why the same default is
# free there and fatal here.
#
# So: cap on spawn platforms, leave fork alone. Raise AMLC_WORKERS if the
# box has RAM to spare -- peak is roughly workers x (400 MB + chunk).
def _default_workers() -> int:
    import multiprocessing
    n = max(1, (os.cpu_count() or 2) - 1)
    if multiprocessing.get_start_method(allow_none=False) == "spawn":
        return min(n, 4)
    return n


WORKERS = int(os.environ.get("AMLC_WORKERS") or _default_workers())

# --- matcher ---
N_FOLDS = 5
# F_0.5 weights precision 2x over recall, so the decision threshold sits well
# above 0.5. Tuned on out-of-fold predictions; this is only the starting point.
DEFAULT_THRESHOLD = 0.70

# Pairs per chunk when scoring the test set. The unchunked path builds one
# frame of 1.73M entities x K=30 = ~52M rows: 19 GB in float64, doubled by
# stage 2's concat, against 23.7 GB of RAM on this box. Chunks are cut only
# where s1_id changes, so every per-entity statistic matches the whole-frame
# result. Lower it if the test phase still runs tight.
TEST_CHUNK_PAIRS = int(os.environ.get("AMLC_TEST_CHUNK") or 4_000_000)

# --- team S3 bucket (cross-account; owned by another team member's account) ---
AWS_REGION = "ap-south-1"
AWS_PROFILE = os.environ.get("AWS_PROFILE", "amlc")
TEAM_BUCKET = "amazon-cuda-commandos-2026"

# Everyone writes under their own prefix so concurrent uploads cannot clobber
# each other. Override per machine with AMLC_MEMBER.
TEAM_MEMBER = os.environ.get("AMLC_MEMBER", "priyanshu")

# The candidate cache is the artifact actually worth sharing: test blocking is
# 4-5 hours of compute, and the parquet reproduces it exactly. One person runs
# it, everyone else pulls the result.
def s3_uri(*parts, member=None):
    """s3://<bucket>/<member>/<parts...> — omit parts for the member's prefix."""
    member = TEAM_MEMBER if member is None else member
    tail = "/".join(str(p).strip("/") for p in parts if p)
    base = f"s3://{TEAM_BUCKET}/{member}"
    return f"{base}/{tail}" if tail else f"{base}/"
