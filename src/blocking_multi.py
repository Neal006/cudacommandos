"""A union of retrievers, instead of one.

WHY
---
Blocking never offers the right record for **5.02% of true pairs**, and no
matcher can recover those. That is the largest structural loss in the
pipeline and it caps everything downstream: a perfect matcher on our
candidates scores 0.9903, which after the measured -0.010 leaderboard offset
is ~0.980.

Experiment 002 profiled the misses and found two concentrations:

    empty-address records    7.8x over-represented
    native-script names      3.6x over-represented

The empty-address case is self-inflicted. Pass A searches a single blob:

    _blob = normalized_name + " " + normalized_address

A record with no address is competing in a TF-IDF space where every other
document carries both fields. Its similarity is diluted by the missing half,
it falls below the top-K cut, and it is never seen again. The feature set even
has an `addr_empty_any` flag, and the trained model gives it **exactly 0.0
importance** -- because by the time a pair reaches the model, the records that
mattered were never retrieved. You cannot fix a retrieval failure with a
feature.

The fix is more retrievers, each good at something different, unioned. This
was rejected earlier purely on cost: four passes over the test set was ~4
hours on a 10-core laptop. On 96 vCPU it is ~25 minutes, so the reason not to
do it has gone.

THE PASSES
----------
    A  name + address, word      the current retriever, unchanged
    B  name only, word           empty-address records, and addresses so
                                 noisy they hurt
    C  transliterated name,      typos and cross-script names: char n-grams
       char 3-5 grams            match "Tetlecommunication"/"Telecommunication"
                                 and survive transliteration wobble where whole
                                 word tokens do not
    D  address only, word        records whose name is a web domain, which the
                                 name passes cannot match on at all

Each pass contributes its own top-k. The union is deduplicated per
(s1_id, cand_id).

WHAT COMES OUT
--------------
`block_sim` is kept as **pass A's** similarity, not the max across passes,
because the downstream model was trained on that scale and cosines from a
char-n-gram space are not comparable to cosines from a word space. Mixing
them would silently shift a feature the model already relies on (`block_sim`
is the second most important stage-1 feature). Records found only by B/C/D
get `block_sim = 0`, which is honest: pass A genuinely did not find them.

Each pass also writes its own `sim_a`/`sim_b`/`sim_c`/`sim_d` column, so a
model can be trained to use *how* a candidate was found. Runners that do not
know about them ignore the extra columns.

GATING
------
A retriever earns its place only if it lifts the measured recall ceiling by
>= 0.002. `--measure` reports each pass's marginal contribution against the
ground truth so that decision is made on numbers.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blocking as B
import config as C


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# name, text column, analyzer, ngram_range, top_k, max_df
PASSES = {
    "a": ("_blob", "word", (1, 1), 30, C.BLOCK_MAX_DF),
    "b": ("_name_only", "word", (1, 1), 10, C.BLOCK_MAX_DF),
    "c": ("_tl_name", "char_wb", (3, 5), 10, 0.02),
    "d": ("_addr_only", "word", (1, 1), 5, C.BLOCK_MAX_DF),
}


_CHUNK = 250_000


def _cols_chunk(args):
    """Worker: every pass text column for one slice of records.

    Module-level and picklable, because Windows starts pool workers with spawn.
    All four columns are built in one pass over the slice so the records are
    touched once rather than once per retriever.
    """
    import normalize as N

    names, addrs, need = args
    out = {}
    if "_blob" in need:
        out["_blob"] = [f"{N.core_name(n)} {N.core_addr(a)}".strip()
                        for n, a in zip(names, addrs)]
    if "_name_only" in need:
        out["_name_only"] = [N.norm_name(n) for n in names]
    if "_addr_only" in need:
        out["_addr_only"] = [N.norm_addr(a) for a in addrs]
    if "_tl_name" in need:
        # transliterate, then strip spaces: char n-grams should not be able to
        # anchor on word boundaries that transliteration moves around.
        out["_tl_name"] = [N.translit_core(n).replace(" ", "") for n in names]
    return out


def load_multi(paths, which, sample=None, seed=C.SEED, workers=None):
    """[entity_id, country, <pass columns>] for one or more source files.

    ingest.load_split_lean cannot be reused here: it drops the name and
    address columns (that is what makes it lean), and every pass except A
    needs them. So this reads the sources itself and builds all the columns
    in one parallel pass.

    Strings are Arrow-backed, as in ingest.blocking_frame -- about 40% of the
    RAM of Python str objects at 10M rows, which matters when we are holding
    up to four text columns instead of one.
    """
    from multiprocessing import Pool

    from ingest import read_polars

    workers = workers or C.WORKERS
    need = {PASSES[p][0] for p in which}
    frames = []
    for path in paths:
        df = read_polars(path)
        if sample and sample < len(df):
            df = df.sample(n=sample, seed=seed)
        names, addrs = df[C.NAME].to_list(), df[C.ADDR].to_list()
        jobs = [(names[i:i + _CHUNK], addrs[i:i + _CHUNK], need)
                for i in range(0, len(names), _CHUNK)]
        del names, addrs
        if len(jobs) > 1 and workers > 1:
            with Pool(workers) as pool:
                parts = pool.map(_cols_chunk, jobs)
        else:
            parts = [_cols_chunk(j) for j in jobs]
        del jobs
        cols = {C.ID: pd.array(df[C.ID].to_list(), dtype="string[pyarrow]"),
                C.COUNTRY: pd.Categorical(df[C.COUNTRY].to_list())}
        for c in need:
            cols[c] = pd.array([v for part in parts for v in part[c]],
                               dtype="string[pyarrow]")
        del parts, df
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _vectorizer(corpus, analyzer, ngram_range, max_df):
    fit_on = corpus
    if C.BLOCK_FIT_SAMPLE and len(corpus) > C.BLOCK_FIT_SAMPLE:
        fit_on = pd.Series(corpus).sample(n=C.BLOCK_FIT_SAMPLE, random_state=C.SEED)
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        token_pattern=r"[a-z0-9]+" if analyzer == "word" else None,
        max_df=max_df,
        min_df=C.BLOCK_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit(fit_on)
    log(f"    vocab {len(vec.vocabulary_):,} ({analyzer} {ngram_range}, "
        f"max_df={max_df}) in {time.time()-t0:.0f}s")
    return vec


def run_pass(p, s1, others, within_country=True):
    """One retriever. Returns {s1_id: [(cand_id, sim), ...]}."""
    col, analyzer, ngrams, top_k, max_df = PASSES[p]
    log(f"  pass {p}: {col} {analyzer}{ngrams} k={top_k}")
    vec = _vectorizer(others[col], analyzer, ngrams, max_df)

    if within_country:
        groups = sorted(set(s1[C.COUNTRY].astype(str)) | set(others[C.COUNTRY].astype(str)))
        parts = [(s1[s1[C.COUNTRY].astype(str) == g], others[others[C.COUNTRY].astype(str) == g], g)
                 for g in groups]
    else:
        parts = [(s1, others, "all")]

    out = {}
    for q_df, i_df, label in parts:
        if len(q_df) == 0:
            continue
        # _block_pair reads a column literally named _blob, so hand it a
        # minimal projection with this pass's text under that name. Renaming
        # in place would leave TWO columns called _blob whenever pass A is
        # also in play, and df["_blob"] then returns a DataFrame.
        q = pd.DataFrame({C.ID: q_df[C.ID].to_numpy(), "_blob": q_df[col].to_numpy()})
        i = pd.DataFrame({C.ID: i_df[C.ID].to_numpy(), "_blob": i_df[col].to_numpy()})
        out.update(B._block_pair(q, i, vec, top_k, C.BLOCK_CHUNK, f"{p}/{label}"))
    return out


def union_frame(per_pass):
    """Merge the passes into one frame, one row per (s1_id, cand_id).

    block_sim stays pass A's similarity. Cosines from a char-n-gram space are
    not on the same scale as word-space cosines, and block_sim is the second
    most important stage-1 feature -- taking a max across passes would shift
    its distribution without telling the model. A record only B/C/D found gets
    block_sim = 0, which is the truth: pass A did not find it.
    """
    frames = []
    for p, cands in per_pass.items():
        s1_ids, cand_ids, sims = [], [], []
        for sid, hits in cands.items():
            for cid, s in hits:
                s1_ids.append(sid)
                cand_ids.append(cid)
                sims.append(s)
        frames.append(pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids,
                                    f"sim_{p}": np.asarray(sims, dtype=np.float32)}))
        log(f"  pass {p}: {len(s1_ids):,} pairs over {len(cands):,} entities")

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["s1_id", "cand_id"], how="outer")
    for p in per_pass:
        out[f"sim_{p}"] = out[f"sim_{p}"].fillna(0.0).astype(np.float32)
    out["block_sim"] = out["sim_a"] if "sim_a" in out else 0.0
    out["n_passes"] = sum((out[f"sim_{p}"] > 0).astype(np.int8) for p in per_pass)
    return out.sort_values(["s1_id", "cand_id"], kind="mergesort").reset_index(drop=True)


def measure(frame, which, truth):
    """Marginal recall of each pass, so a retriever earns its place on numbers.

    A pass that adds < 0.002 of ceiling is not worth the pairs it costs: the
    matcher sits 3.7 points below the ceiling we already have, so widening
    the ceiling only helps where it is actually binding.
    """
    def ceiling(r):
        return 1.25 * r / (0.25 + r)

    tot = sum(len(v) for v in truth.values())
    got = frame.groupby("s1_id")["cand_id"].apply(set)

    def recall_of(cols):
        m = np.zeros(len(frame), dtype=bool)
        for c in cols:
            m |= frame[f"sim_{c}"].to_numpy() > 0
        sub = frame[m].groupby("s1_id")["cand_id"].apply(set)
        hit = sum(len(truth.get(s, set()) & c) for s, c in sub.items())
        return hit / max(tot, 1)

    base = recall_of(["a"])
    log(f"  pass a alone      recall {base:.4f}  ceiling {ceiling(base):.4f}")
    cum = ["a"]
    for p in [x for x in which if x != "a"]:
        r_without = recall_of(cum)
        cum.append(p)
        r_with = recall_of(cum)
        d = ceiling(r_with) - ceiling(r_without)
        verdict = "KEEP" if d >= 0.002 else "drop (<0.002)"
        log(f"  + pass {p}        recall {r_with:.4f}  ceiling {ceiling(r_with):.4f}  "
            f"delta {d:+.4f}  {verdict}")
    log(f"  union             recall {recall_of(list(which)):.4f}  "
        f"ceiling {ceiling(recall_of(list(which))):.4f}")


def generate(which, split, sample=None, measure_recall=False):
    import data as D

    p1, p2, p3 = D.source_paths(split)
    s1 = load_multi([p1], which, sample=sample)
    others = load_multi([p2, p3], which)
    log(f"{split}: {len(s1):,} queries, {len(others):,} index records")

    per_pass = {}
    for p in which:
        t0 = time.time()
        per_pass[p] = run_pass(p, s1, others)
        log(f"  pass {p} done in {(time.time()-t0)/60:.1f} min")

    frame = union_frame(per_pass)
    log(f"union: {len(frame):,} pairs over {frame['s1_id'].nunique():,} entities "
        f"({len(frame)/max(frame['s1_id'].nunique(),1):.1f} per entity)")

    if measure_recall and split == "train":
        truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID].astype(str).tolist())
        measure(frame, which, truth)

    key = f"{split}_multi{''.join(sorted(which))}_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}_n{sample or 'all'}"
    path = C.INTERIM / f"cands_{key}.parquet"
    frame.to_parquet(path, index=False)
    log(f"cached -> {path.name}")
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default="abcd", help="subset of abcd")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--measure", action="store_true", help="report each pass's marginal recall")
    a = ap.parse_args()
    bad = set(a.passes) - set(PASSES)
    if bad:
        raise SystemExit(f"unknown passes {sorted(bad)}; known: {sorted(PASSES)}")
    generate(a.passes, a.split, a.sample, a.measure)
