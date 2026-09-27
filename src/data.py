"""Loading and writing the challenge's TSV files.

Everything is tab-separated because names and addresses contain commas, and the
id-list columns use commas as their own separator. Reading without an explicit
sep="\\t" silently yields one column holding the whole line — the problem
statement calls this out, and it is an easy hour to lose.

Memory matters here. Train is ~12.5M records across three sources and only
~10GB of RAM is available, so loading is column-restricted and the heavy
normalized forms are attached separately (see normalize.py) rather than
materialized for every record up front.
"""
import sys
from pathlib import Path

import pandas as pd

import config as C
from metrics import parse_id_list, format_id_list

USECOLS = [C.ID, C.NAME, C.ADDR, C.COUNTRY]


def read_source(path, nrows=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing.\n"
            f"Every source file listed in the problem statement is required — "
            f"without it, no candidate from that source can ever be predicted."
        )
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=lambda c: c in USECOLS, nrows=nrows,
    )
    missing = set(USECOLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}; got {list(df.columns)}")
    # country has a handful of distinct values across millions of rows
    df[C.COUNTRY] = df[C.COUNTRY].astype("category")
    return df


def read_ground_truth(path, keep_ids=None) -> dict:
    """source1_entity_id -> set of matched ids (empty set for singletons).

    `keep_ids` restricts to a subsample without materializing the full 2.2M-row
    mapping twice.
    """
    keep = set(keep_ids) if keep_ids is not None else None
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if keep is not None and sid not in keep:
                continue
            out[sid] = parse_id_list(parts[1] if len(parts) > 1 else "")
    return out


def load_split(which: str, sample=None, seed=C.SEED):
    """which: 'train' or 'test'. Returns (s1, s2, s3) with blocking columns.

    `sample` subsamples Source 1 only — S2/S3 must stay complete, because a
    true match can live anywhere in them.
    """
    from normalize import add_blocking_columns

    paths = {
        "train": (C.TRAIN_S1, C.TRAIN_S2, C.TRAIN_S3),
        "test": (C.TEST_S1, C.TEST_S2, C.TEST_S3),
    }[which]

    s1 = read_source(paths[0])
    if sample and sample < len(s1):
        s1 = s1.sample(n=sample, random_state=seed).reset_index(drop=True)

    frames = [s1]
    for p in paths[1:]:
        try:
            frames.append(read_source(p))
        except FileNotFoundError as e:
            # Fail loudly but let the caller decide: a missing SOURCE file is
            # recoverable for a dry run, never for a real submission.
            print(f"\n*** MISSING SOURCE FILE ***\n{e}\n", file=sys.stderr)
            frames.append(pd.DataFrame(columns=USECOLS))

    return [add_blocking_columns(f) for f in frames]


def load_records_by_id(path, ids, chunksize=1_000_000) -> pd.DataFrame:
    """Read only the rows whose entity_id is in `ids`, streaming the file.

    The feature stage needs business_name / business_address, but only for
    records that survived blocking — a small fraction of the 10M. Streaming
    and filtering per chunk keeps peak memory to one chunk instead of the
    whole file, which is what lets blocking drop those columns entirely.
    """
    ids = set(ids)
    if not ids:
        return pd.DataFrame(columns=USECOLS)
    keep = []
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=lambda c: c in USECOLS, chunksize=chunksize,
    ):
        hit = chunk[chunk[C.ID].isin(ids)]
        if len(hit):
            keep.append(hit)
    if not keep:
        return pd.DataFrame(columns=USECOLS)
    return pd.concat(keep, ignore_index=True)


def source_paths(which: str):
    """(source1, source2, source3) paths for 'train' or 'test'."""
    return {
        "train": (C.TRAIN_S1, C.TRAIN_S2, C.TRAIN_S3),
        "test": (C.TEST_S1, C.TEST_S2, C.TEST_S3),
    }[which]


def check_test_files():
    """Confirm every test source exists before a run that ends in a submission."""
    missing = [p.name for p in (C.TEST_S1, C.TEST_S2, C.TEST_S3) if not Path(p).exists()]
    return missing


def write_submission(path, s1_ids, mapping, id_col, list_col):
    """Write matching_results.tsv or candidate_pairs.tsv.

    Enforces the format rules directly, because a rejected file costs one of
    only five daily submissions:
      - exactly one row per Source-1 entity, in test-file order
      - empty cell for no matches
      - no duplicates inside a list
    """
    rows = [
        {id_col: sid, list_col: format_id_list(mapping.get(sid, set()))}
        for sid in s1_ids
    ]
    df = pd.DataFrame(rows, columns=[id_col, list_col])
    if df[id_col].duplicated().any():
        raise ValueError("duplicate source1_entity_id rows — submission would be rejected")
    df.to_csv(path, sep="\t", index=False)
    return df


def write_outputs(test_s1_ids, matches: dict, candidates: dict):
    """Write both required output files and report a sanity summary."""
    m = write_submission(C.MATCHING_RESULTS, test_s1_ids, matches, C.GT_S1, C.GT_MATCH)
    c = write_submission(C.CANDIDATE_PAIRS, test_s1_ids, candidates, C.GT_S1, C.OUT_CAND)
    if len(c) != len(m):
        raise ValueError(f"candidate_pairs has {len(c)} rows, matching_results {len(m)}")

    # Final matches must be a subset of candidates — the official validator
    # warns otherwise, and it always means a pipeline bug.
    leaked = sum(
        1 for sid in test_s1_ids
        if not matches.get(sid, set()) <= candidates.get(sid, set())
    )
    n_match = m[C.GT_MATCH].map(lambda s: len(parse_id_list(s)))
    return {
        "matching_results": str(C.MATCHING_RESULTS),
        "candidate_pairs": str(C.CANDIDATE_PAIRS),
        "rows": len(m),
        "entities_with_matches": int((n_match > 0).sum()),
        "predicted_singletons": int((n_match == 0).sum()),
        "predicted_singleton_rate": float((n_match == 0).mean()),
        "mean_matches": float(n_match.mean()),
        "entities_with_matches_outside_candidates": leaked,
    }


def frame_fingerprint(pairs: pd.DataFrame) -> str:
    """Order-sensitive hash of a candidate frame's (s1_id, cand_id) rows.

    A cached score array means "score of row i". Two arrays of the same length
    over differently ordered frames are silently misaligned, and a length check
    cannot see that, so score caches carry this beside them.
    """
    import hashlib
    h = pd.util.hash_pandas_object(pairs[["s1_id", "cand_id"]], index=False)
    return hashlib.sha1(h.to_numpy().tobytes()).hexdigest()[:16]


def write_score_meta(npy_path, pairs: pd.DataFrame, **info) -> None:
    """Sidecar `<scores>.json`: row count + frame fingerprint (+ provenance)."""
    import json
    meta = {"rows": int(len(pairs)), "frame": frame_fingerprint(pairs), **info}
    Path(npy_path).with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def check_score_meta(npy_path, pairs: pd.DataFrame) -> None:
    """Refuse a score array cached over a different frame.

    A missing sidecar only warns: caches written before sidecars existed
    (e.g. submission 003's testp_f550ffb02552.npy) stay usable.
    """
    import json
    side = Path(npy_path).with_suffix(".json")
    if not side.exists():
        print(f"WARNING: {side.name} missing -- cannot prove {Path(npy_path).name} "
              "is aligned with this candidate frame (length check only)", flush=True)
        return
    meta = json.loads(side.read_text(encoding="utf-8"))
    got = frame_fingerprint(pairs)
    if meta.get("rows") != len(pairs) or meta.get("frame") != got:
        raise SystemExit(f"{Path(npy_path).name} was scored over a different candidate frame "
                         f"(rows {meta.get('rows')} vs {len(pairs)}, "
                         f"frame {meta.get('frame')} vs {got})")
