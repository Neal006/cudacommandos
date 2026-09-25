"""Lean, parallel loading of S2/S3 for blocking.

`data.load_split` holds every S2/S3 name and address as pandas object strings
and then copies the frame to add the blob, which peaks at 6-8 GB and is what
makes blocking and anything else unable to share a 16-25 GB machine. Here polars
reads the file (Arrow strings, ~3x smaller), the blob is built in worker
processes, and only [entity_id, country, _blob] survive.

The blob is produced by the very same `normalize.core_name/core_addr`, so the
result is identical to `add_blocking_columns` and candidate caches built by
either path are interchangeable.
"""
import os
from multiprocessing import Pool

import pandas as pd
import polars as pl

import config as C
from normalize import core_addr, core_name

_CHUNK = 250_000


def _blob_chunk(args):
    names, addrs = args
    return [f"{core_name(n)} {core_addr(a)}".strip() for n, a in zip(names, addrs)]


def read_polars(path) -> pl.DataFrame:
    """All four columns as strings; empty cells stay empty strings, like read_source."""
    df = pl.read_csv(path, separator="\t", infer_schema=False, quote_char='"',
                     columns=[C.ID, C.NAME, C.ADDR, C.COUNTRY])
    return df.with_columns(pl.col(C.NAME, C.ADDR, C.COUNTRY).fill_null(""))


def blocking_frame(paths, workers=None) -> pd.DataFrame:
    """[entity_id, country, _blob] for one or more source files, blobs built in parallel."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    frames = []
    for p in paths:
        df = read_polars(p)
        names, addrs = df[C.NAME].to_list(), df[C.ADDR].to_list()
        jobs = [(names[i:i + _CHUNK], addrs[i:i + _CHUNK]) for i in range(0, len(names), _CHUNK)]
        del names, addrs
        with Pool(workers) as pool:
            blobs = [b for part in pool.map(_blob_chunk, jobs) for b in part]
        del jobs
        # Arrow-backed strings: ~40% of the RAM of Python str objects at 10M rows.
        frames.append(pd.DataFrame({
            C.ID: pd.array(df[C.ID].to_list(), dtype="string[pyarrow]"),
            C.COUNTRY: pd.Categorical(df[C.COUNTRY].to_list()),
            "_blob": pd.array(blobs, dtype="string[pyarrow]"),
        }))
        del df, blobs
    out = pd.concat(frames, ignore_index=True)
    out[C.COUNTRY] = out[C.COUNTRY].astype("category")
    return out


def load_split_lean(which: str, sample=None, seed=C.SEED):
    """Same contract as data.load_split (s1, s2, s3 with `_blob`, text dropped),
    with S1 sampled exactly like it so cache keys and contents match."""
    import data as D
    from normalize import add_blocking_columns

    p1, p2, p3 = D.source_paths(which)
    s1 = D.read_source(p1)
    if sample and sample < len(s1):
        s1 = s1.sample(n=sample, random_state=seed).reset_index(drop=True)
    s1 = add_blocking_columns(s1)
    return s1, blocking_frame([p2]), blocking_frame([p3])
