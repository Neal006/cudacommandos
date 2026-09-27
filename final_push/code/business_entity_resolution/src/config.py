"""Paths and hyper-parameters shared by every stage."""
import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ER_DATA", _REPO / "dataset"))
WORK = Path(os.environ.get("ER_WORK", _REPO / "work"))
OUT = Path(os.environ.get("ER_OUT", _REPO / "output"))
VALIDATOR = Path(os.environ.get("ER_VALIDATOR", _REPO / "utils" / "validate_submission.py"))
for _p in (WORK, OUT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_JOBS = int(os.environ.get("ER_JOBS", max(1, (os.cpu_count() or 4) - 2)))

# splits
HIDDEN_FRAC = 0.20          # train S1 never queried: orphan simulation + neural training set
N_FOLDS = 5

# blocking
TOPK_NAME = 30              # sparse char-3gram on name
TOPK_NA = 30                # sparse char-3gram on name + street + locality
TOPK_ADDR = 30              # sparse word tf-idf on address tokens + numbers
TOPK_REVERSE_NOADDR = 20    # reverse pass for pool records without a state / address
TOPK_REVERSE = 5            # pool record -> top S1
SPARSE_THRESHOLD = 0.10
EXACT_BLOCK_CAP = 300       # skip exact-key blocks with more pool records than this

# pruning (every value can be overridden with an ER_PRUNE_* environment variable; see prune.py)
PRUNE_K = int(os.environ.get("ER_PRUNE_K", 20))
PRUNE_MIN_P = float(os.environ.get("ER_PRUNE_MIN_P", 0.005))
PRUNE_MIN_P_COUNTRY = json.loads(os.environ.get("ER_PRUNE_MIN_P_COUNTRY", "{}"))  # e.g. {"France": 0.002}
PRUNE_K_PER_SRC = int(os.environ.get("ER_PRUNE_K_PER_SRC", 10))   # also keep top-k from S2 and from S3 (0 = off)
PRUNE_KEEP_CAND_BEST = os.environ.get("ER_PRUNE_KEEP_CAND_BEST", "1") == "1"   # keep each pool record's best S1
PRUNE_CAND_BEST_MIN_P = float(os.environ.get("ER_PRUNE_CAND_BEST_MIN_P", 0.001))
PRUNE_EXTRA_FEATS = os.environ.get("ER_PRUNE_EXTRA_FEATS", "1") == "1"   # flags / name frequency / house-number diff
PRUNE_HARD_NEG_RANK = int(os.environ.get("ER_PRUNE_HARD_NEG_RANK", 5))  # keep all negatives ranked <= this (0 = off)

# cross-encoder band
CE_MIN_P1 = 0.01
CE_MAX_PER_S1 = 8
CE_MAX_P1 = 0.999          # above this stage-1 is (almost) never wrong: 36% of pairs, 0.3% of errors

# decision
MC_SAMPLES = 256
DECIDE_MAX_M = 10

SPLITS = ("train", "test")


def wpath(name: str) -> Path:
    return WORK / name
