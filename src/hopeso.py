"""hopeso: recall jugaad on top of the word TF-IDF blocker.

Measured on the 30k train sample (blocking recall 0.9496, 5,225 true pairs missed):
  22% of misses have the SAME core_name as the S1 entity, 36% the same
  transliterated skeleton, 25% an empty address. They are not hard pairs --
  word TF-IDF top-30 just ranks them out (generic names, address tokens
  outvoting the name). And exact-name siblings of the true matches we DO find
  recover 23% of misses at only ~5 extra candidates per entity.

So cheap passes, unioned with the base frame:
  sibs   siblings (same country + key) of each entity's top-m base candidates.
         Sources hold 5-6 copies of one business; find one, pull the rest.
  dost   S1 -> records with the same (country, key), only when that key's
         pool is small (a unique-ish name, not "medical store").
  both keys: core_name (exact) and skeleton (translit, catches Indic script)

Every new pair gets block_sim recomputed with the SAME vectorizer the base
blocker used, because block_sim is a model feature -- a pass that filled it
with 0 would hand the matcher a value it never saw in training.

    python src/hopeso.py measure --n 30000                # recall vs cost table
    python src/hopeso.py build --split train --n 150000   # union frame for run_v4
    python src/hopeso.py build --split test                # union frame for test
    python src/run_v4.py ... --cands-tag hopeso           # train/score on it
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import data as D  # noqa: E402
import normalize as N  # noqa: E402

# the passes kept after `measure` (see context.md); change here, not in callers
VIBE = dict(sib_top=3, sib_cap=50, dost_cap=10)


def log(m):
    print(f"[hopeso {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _keyz_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        cn = N.core_name(n) if n else ""
        tn = N.translit_core(n) if n else ""
        out.append((cn, N.skeleton(tn) if tn else "", not (a or "").strip()))
    return out


def keyz(df: pl.DataFrame, workers=None) -> pl.DataFrame:
    """[entity_id, country, cn, sk, addr_empty] for any source frame."""
    names, addrs = df[C.NAME].to_list(), df[C.ADDR].to_list()
    step = 200_000
    jobs = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    with Pool(workers or C.WORKERS) as pool:
        res = [r for part in pool.map(_keyz_chunk, jobs) for r in part]
    return pl.DataFrame({
        C.ID: df[C.ID], C.COUNTRY: df[C.COUNTRY],
        "cn": [r[0] for r in res], "sk": [r[1] for r in res], "addr_empty": [r[2] for r in res]})


def src_keyz(split: str) -> pl.DataFrame:
    """Keys for all S2+S3 records of a split, cached (10M rows is ~4 min on 4 cores)."""
    path = C.INTERIM / f"hopeso_keys_{split}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    from ingest import read_polars
    _, p2, p3 = D.source_paths(split)
    k = keyz(pl.concat([read_polars(p2), read_polars(p3)]))
    k.write_parquet(path)
    log(f"keys -> {path.name} ({k.height:,} records)")
    return k


def s1_keyz(split: str, n=None) -> pl.DataFrame:
    """Keys for S1, sampled exactly like ingest.load_split_lean."""
    s1 = D.read_source(D.source_paths(split)[0])
    if n and n < len(s1):
        s1 = s1.sample(n=n, random_state=C.SEED).reset_index(drop=True)
    # read_source gives a Categorical country; plain strings before fillna
    return keyz(pl.from_pandas(s1[[C.ID, C.NAME, C.ADDR, C.COUNTRY]].astype("string").fillna("")))


def _pools(k: pl.DataFrame, key: str) -> pl.DataFrame:
    """(country, key) -> pool size, for capping generic names."""
    return k.filter(pl.col(key) != "").group_by([C.COUNTRY, key]).agg(pl.len().alias("pool"))


def sibs(base: pl.DataFrame, k: pl.DataFrame, key: str, top: int, cap: int) -> pl.DataFrame:
    """Siblings of each entity's top-`top` base candidates by block_sim."""
    tops = (base.sort(["s1_id", "block_sim"], descending=[False, True])
            .group_by("s1_id", maintain_order=True).head(top)
            .join(k.select(C.ID, C.COUNTRY, key), left_on="cand_id", right_on=C.ID)
            .filter(pl.col(key) != ""))
    ok = _pools(k, key).filter(pl.col("pool") <= cap)
    bros = k.select(C.ID, C.COUNTRY, key).join(ok.select(C.COUNTRY, key), on=[C.COUNTRY, key])
    return (tops.select("s1_id", C.COUNTRY, key)
            .join(bros, on=[C.COUNTRY, key])
            .select("s1_id", pl.col(C.ID).alias("cand_id")).unique())


def dost(s1k: pl.DataFrame, k: pl.DataFrame, key: str, cap: int) -> pl.DataFrame:
    """S1 -> every record with the same (country, key) when the pool is small."""
    ok = _pools(k, key).filter(pl.col("pool") <= cap)
    bros = k.select(C.ID, C.COUNTRY, key).join(ok.select(C.COUNTRY, key), on=[C.COUNTRY, key])
    return (s1k.filter(pl.col(key) != "").select(pl.col(C.ID).alias("s1_id"), C.COUNTRY, key)
            .join(bros, on=[C.COUNTRY, key])
            .select("s1_id", pl.col(C.ID).alias("cand_id")).unique())


def extras(base: pl.DataFrame, s1k: pl.DataFrame, k: pl.DataFrame, vibe=VIBE) -> pl.DataFrame:
    """All jugaad passes, minus what the base frame already has."""
    parts = [sibs(base, k, key, vibe["sib_top"], vibe["sib_cap"]) for key in ("cn", "sk")]
    if vibe.get("dost_cap"):
        parts += [dost(s1k, k, key, vibe["dost_cap"]) for key in ("cn", "sk")]
    return (pl.concat(parts).unique()
            .join(base.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti"))


def fill_sim(new: pd.DataFrame, split: str, n=None) -> np.ndarray:
    """block_sim for new pairs with the base blocker's own vectorizer (same fit
    sample, same seed, same corpus order -> same vocabulary and IDF)."""
    import blocking
    import ingest
    s1, s2, s3 = ingest.load_split_lean(split, sample=n)
    others = pd.concat([s2, s3], ignore_index=True)
    vec = blocking.build_vectorizer(others["_blob"])
    blob_of = pd.Series(others["_blob"].to_numpy(), index=others[C.ID].to_numpy())
    s1_blob = pd.Series(s1["_blob"].to_numpy(), index=s1[C.ID].astype(str).to_numpy())
    del s2, s3, others
    # Tokenize each distinct blob ONCE (transform is single-threaded Python;
    # new pairs repeat the same S1 ~8x and the same record many times), then
    # gather rows by position. Unknown ids would silently score 0, so refuse them.
    s_u, s_ix = np.unique(new["s1_id"].to_numpy(), return_inverse=True)
    c_u, c_ix = np.unique(new["cand_id"].to_numpy(), return_inverse=True)
    missing = int(pd.Index(s_u).isin(s1_blob.index).__invert__().sum()
                  + pd.Index(c_u).isin(blob_of.index).__invert__().sum())
    if missing:
        raise SystemExit(f"fill_sim: {missing} ids have no blob -- frame and split disagree")
    A = vec.transform(s1_blob.reindex(s_u).to_numpy())
    B = vec.transform(blob_of.reindex(c_u).to_numpy())
    out = np.empty(len(new), dtype=np.float32)
    step = 2_000_000
    for a in range(0, len(new), step):
        b = min(a + step, len(new))
        out[a:b] = np.asarray(A[s_ix[a:b]].multiply(B[c_ix[a:b]]).sum(1)).ravel()
    return out


def base_frame(split: str, n=None) -> pl.DataFrame:
    path = C.INTERIM / (f"cands_{split}_k{C.TOP_K}_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}"
                        f"_ctry{int(C.BLOCK_WITHIN_COUNTRY)}_n{n or 'all'}.parquet")
    if not path.exists():
        raise SystemExit(f"{path.name} missing -- run the base blocker first (run_v4 / run_v2)")
    return pl.read_parquet(path)


def tag_path(split: str, n=None, tag="hopeso", vibe=VIBE) -> Path:
    """The pass settings are in the name: change VIBE and a stale frame can't be reused."""
    v = f"t{vibe['sib_top']}c{vibe['sib_cap']}d{vibe.get('dost_cap', 0)}"
    return C.INTERIM / (f"cands_{split}_k{C.TOP_K}_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}"
                        f"_ctry{int(C.BLOCK_WITHIN_COUNTRY)}_n{n or 'all'}_{tag}_{v}.parquet")


def load_frame(s1, s2, s3, split, n=None, tag=None):
    """The candidate frame every consumer must share: base blocker, or base ∪
    hopeso passes when `tag` is set. run_v4 (train + test) and score_test (test)
    both call this, so a model trained on the tagged frame is never scored on
    the untagged one."""
    if not tag:
        from run_pipeline import cached_candidates
        return cached_candidates(s1, s2, s3, split, n, frame_only=True)[1]
    path = tag_path(split, n, tag)
    if not path.exists():
        raise SystemExit(f"{path.name} missing -- run: python src/hopeso.py build "
                         f"--split {split}" + (f" --n {n}" if n else ""))
    log(f"loading tagged candidates: {path.name}")
    return pd.read_parquet(path)


def build(split: str, n=None):
    """Write base ∪ extras (with block_sim) where run_v4 --cands-tag hopeso reads it."""
    t0 = time.time()
    base = base_frame(split, n)
    new = extras(base, s1_keyz(split, n), src_keyz(split)).to_pandas()
    log(f"{split}: base {base.height:,} pairs + {len(new):,} new "
        f"({len(new) / base['s1_id'].n_unique():.2f}/entity)")
    new["block_sim"] = fill_sim(new, split, n)
    out = (pl.concat([base.select("s1_id", "cand_id", "block_sim"),
                      pl.from_pandas(new).select("s1_id", "cand_id",
                                                 pl.col("block_sim").cast(base["block_sim"].dtype))])
           .sort("s1_id", maintain_order=True))
    out.write_parquet(tag_path(split, n))
    log(f"-> {tag_path(split, n).name}: {out.height:,} pairs in {(time.time() - t0) / 60:.1f} min")


def measure(n: int):
    """Recall and cost of each pass (and the union) on a train sample with labels."""
    base = base_frame("train", n)
    s1k, k = s1_keyz("train", n), src_keyz("train")
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1k[C.ID].to_list())
    tp = pl.DataFrame([(s, c) for s, cs in truth.items() for c in cs],
                      schema=["s1_id", "cand_id"], orient="row")
    n_true, n_ent = tp.height, s1k.height

    def hit(fr):
        return fr.join(tp, on=["s1_id", "cand_id"]).height

    b_hit = hit(base)
    log(f"base: recall {b_hit / n_true:.4f}  ({base.height / n_ent:.1f} cands/entity)")
    rows = []
    variants = {f"sibs {key} top{t} cap{c}": sibs(base, k, key, t, c)
                for key in ("cn", "sk") for t in (1, 3, 5) for c in (20, 50)}
    variants |= {f"dost {key} cap{c}": dost(s1k, k, key, c)
                 for key in ("cn", "sk") for c in (5, 10, 20)}
    for name, fr in variants.items():
        fr = fr.join(base.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
        rows.append((name, hit(fr), fr.height))
    for vibe in (VIBE, dict(VIBE, dost_cap=0), dict(VIBE, sib_top=5, dost_cap=20)):
        fr = extras(base, s1k, k, vibe)
        rows.append((f"UNION {vibe}", hit(fr), fr.height))
    miss = n_true - b_hit
    print(f"\n{'pass':<48} {'recovers':>16} {'new/entity':>11} {'recall':>8} {'F0.5 ceil':>9}")
    for name, h, sz in rows:
        r = (b_hit + h) / n_true
        print(f"{name:<48} {h:>7,} ({h / miss:5.1%}) {sz / n_ent:>11.2f} {r:>8.4f} "
              f"{1.25 * r / (0.25 + r):>9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--n", type=int, default=30000)
    b = sub.add_parser("build")
    b.add_argument("--split", choices=["train", "test"], required=True)
    b.add_argument("--n", type=int, default=None, help="train sample/frame size (None = all)")
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.n)
    else:
        build(a.split, a.n)
