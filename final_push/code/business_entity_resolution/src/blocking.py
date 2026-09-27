"""Stage B: candidate generation.

B.1 sparse char-3gram TF-IDF, partitioned by (country, state), plus a pass against pool records whose
    state is unknown. Two texts: name only, and name + street + city. Top-K per S1.
B.1r reverse pass: every pool record -> its top S1s on the name+address text.
B.2 exact keys: (locality, street), (state, name_key), (locality, house number, first street token).
B.4 union with per-blocker scores / ranks / flags  ->  cands_{split}.parquet

Every S1 (train incl. hidden, and test) is queried; the hidden set is only filtered out downstream.
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import (EXACT_BLOCK_CAP, N_JOBS, SPARSE_THRESHOLD, TOPK_ADDR, TOPK_NA, TOPK_NAME, TOPK_REVERSE,
                    TOPK_REVERSE_NOADDR, wpath)
from normalize import STOP, STREET_WORDS, load_norm

COLS = ["idx", "src", "country", "n_core", "n_alt", "n_key", "a_state", "a_loc", "a_city", "a_street", "a_hnd",
        "a_hn", "a_tok", "a_nums"]
KEYS = (("name", TOPK_NAME), ("na", TOPK_NA), ("ad", TOPK_ADDR))


def _texts(df: pl.DataFrame):
    return df.with_columns(
        (pl.col("n_core") + " " + pl.col("n_alt").str.replace_all(r"\|", " ")).str.strip_chars().alias("t_name"),
        (pl.col("n_core") + " " + pl.col("a_street") + " " + pl.col("a_city")).str.strip_chars().alias("t_na"),
        # address only (word level): catches records whose name is unrelated junk / a trade name
        (pl.col("a_tok") + " " + pl.col("a_nums") + " #" + pl.col("a_hn")).str.strip_chars(" #").alias("t_ad"),
    )


def _topn(Q, P, qidx, pidx, k):
    """Row-wise top-k cosine between rows of Q and rows of P -> (q_idx, p_idx, score) arrays."""
    if Q.shape[0] == 0 or P.shape[0] == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    R = sp_matmul_topn(Q, P.T.tocsr(), top_n=k, threshold=SPARSE_THRESHOLD, sort=True, n_threads=N_JOBS)
    R = R.tocoo()
    return qidx[R.row].astype(np.int32), pidx[R.col].astype(np.int32), R.data.astype(np.float32)


def sparse_block(split: str, df: pl.DataFrame):
    out = {"name": [], "na": [], "ad": [], "rev": []}
    for country in df["country"].unique().to_list():
        t0 = time.time()
        d = df.filter(pl.col("country") == country)
        is_s1 = (d["src"] == 1).to_numpy()
        idx = d["idx"].to_numpy()
        state = d["a_state"].to_numpy()
        mats = {}
        for key, _ in KEYS:
            if key == "ad":
                vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=0.02,
                                      sublinear_tf=True, dtype=np.float32)
            else:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02,
                                      sublinear_tf=True, dtype=np.float32)
            mats[key] = vec.fit_transform(d[f"t_{key}"].to_list()).tocsr()
        s1_states = set(state[is_s1]) - {""}
        passes = [(st, is_s1 & (state == st), ~is_s1 & (state == st)) for st in s1_states]
        passes.append(("<pool-missing>", is_s1, ~is_s1 & (state == "")))
        passes.append(("<s1-missing>", is_s1 & (state == ""), ~is_s1))
        print(f"  sparse {split} {country}: tfidf fitted {time.time() - t0:.0f}s", flush=True)
        for n_done, (st, qm, pm) in enumerate(passes):
            if n_done % 10 == 0:
                print(f"    {n_done}/{len(passes)} partitions, {time.time() - t0:.0f}s", flush=True)
            if not qm.any() or not pm.any():
                continue
            qi, pi = np.flatnonzero(qm), np.flatnonzero(pm)
            for key, k in KEYS:
                M = mats[key]
                out[key].append(_topn(M[qi], M[pi], idx[qi], idx[pi], k))
            if st == "<pool-missing>":   # no usable address: reverse on the name, wider
                M = mats["name"]
                c, s, sc = _topn(M[pi], M[qi], idx[pi], idx[qi], TOPK_REVERSE_NOADDR)
                out["rev"].append((s, c, sc))
            elif st != "<s1-missing>":
                M = mats["na"]
                c, s, sc = _topn(M[pi], M[qi], idx[pi], idx[qi], TOPK_REVERSE)
                out["rev"].append((s, c, sc))
        print(f"  sparse {split} {country}: {d.height} rows, {time.time() - t0:.0f}s", flush=True)
    res = {}
    for key, parts in out.items():
        s1 = np.concatenate([p[0] for p in parts])
        cand = np.concatenate([p[1] for p in parts])
        sc = np.concatenate([p[2] for p in parts])
        f = pl.DataFrame({"s1": s1, "cand": cand, f"s_{key}": sc})
        f = f.group_by("s1", "cand").agg(pl.col(f"s_{key}").max())
        if key != "rev":
            k = dict(KEYS)[key]
            f = (f.with_columns(pl.col(f"s_{key}").rank("ordinal", descending=True).over("s1")
                                .cast(pl.Int16).alias(f"r_{key}"))
                  .filter(pl.col(f"r_{key}") <= k))
        res[key] = f
    return res


def exact_block(df: pl.DataFrame) -> pl.DataFrame:
    s1 = df.filter(pl.col("src") == 1)
    pool = df.filter(pl.col("src") != 1)

    def explode_loc(x):
        return x.with_columns(pl.col("a_loc").str.split("|").alias("loc")).explode("loc", empty_as_null=True).filter(pl.col("loc") != "")

    def join(keys, a, b, bit):
        b = b.filter(pl.len().over(keys) <= EXACT_BLOCK_CAP)
        a = a.filter(pl.len().over(keys) <= EXACT_BLOCK_CAP)
        return (a.select(keys + [pl.col("idx").alias("s1")])
                 .join(b.select(keys + [pl.col("idx").alias("cand")]), on=keys)
                 .select("s1", "cand", pl.lit(bit, pl.Int8).alias("x")))

    parts = []
    s1l, pl_ = explode_loc(s1), explode_loc(pool)
    parts.append(join(["country", "loc", "a_street"], s1l.filter(pl.col("a_street") != ""),
                      pl_.filter(pl.col("a_street") != ""), 1))
    parts.append(join(["country", "a_state", "n_key"], s1.filter(pl.col("n_key") != ""),
                      pool.filter(pl.col("n_key") != ""), 2))
    parts.append(join(["country", "n_key"], s1.filter(pl.col("n_key") != ""),
                      pool.filter((pl.col("n_key") != "") & (pl.col("a_state") == "")), 8))
    # first DISTINCTIVE street word: skipping street types / articles, else in France nearly every key is
    # (city, number, "rue") and the oversized blocks get dropped by the cap (53% of French pool records)
    generic = sorted(STREET_WORDS | STOP)
    fs = lambda x: x.with_columns(  # noqa: E731
        pl.col("a_street").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(generic)))
        .list.first().fill_null(pl.col("a_street").str.split(" ").list.first()).alias("st1"))
    parts.append(join(["country", "loc", "a_hnd", "st1"], fs(s1l).filter(pl.col("a_hnd") != ""),
                      fs(pl_).filter(pl.col("a_hnd") != ""), 4))
    ex = pl.concat(parts).group_by("s1", "cand").agg(pl.col("x").sum().cast(pl.Int8))
    return ex


def union(sp, ex):
    c = ex
    for key in ("name", "na", "ad", "rev"):
        c = c.join(sp[key], on=["s1", "cand"], how="full", coalesce=True)
    return c.with_columns(pl.col("s_name", "s_na", "s_ad", "s_rev").fill_null(0.0),
                          pl.col("r_name", "r_na", "r_ad").fill_null(99), pl.col("x").fill_null(0))


def block_split(split: str):
    t0 = time.time()
    df = _texts(load_norm(split, COLS))
    sp = sparse_block(split, df)
    ex = exact_block(df)
    c = union(sp, ex)
    c = c.sort("s1", "cand")
    c.write_parquet(wpath(f"cands_{split}.parquet"))
    print(f"{split}: {c.height:,} candidate pairs, {c['s1'].n_unique():,} S1, {time.time() - t0:.0f}s")
    return c


def recall_report(c: pl.DataFrame):
    from splits import load_splits
    gt = pl.read_parquet(wpath("gt_pairs.parquet"))
    q = load_splits().filter(~pl.col("hidden")).select(pl.col("idx").alias("s1"))
    gtq = gt.join(q, on="s1")
    cq = c.join(q, on="s1")
    hit = gtq.join(cq, on=["s1", "cand"], how="left").with_columns(pl.col("x").is_not_null().alias("hit"))
    print(f"recall (Q): {hit['hit'].mean():.5f}  cands/S1: {cq.height / q.height:.1f}")
    h = hit.filter(pl.col("hit"))
    for name, expr in [("name", pl.col("r_name") <= TOPK_NAME), ("na", pl.col("r_na") <= TOPK_NA),
                       ("ad", pl.col("r_ad") <= TOPK_ADDR),
                       ("rev", pl.col("s_rev") > 0), ("exact", pl.col("x") > 0)]:
        print(f"  {name:6s} alone: {h.filter(expr).height / gtq.height:.5f}")
    return hit


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    for s in splits:
        cands = block_split(s)
        if s == "train":
            recall_report(cands)
