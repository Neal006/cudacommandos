"""Dense-embedding candidate search (GPU job 1) — a MEASUREMENT tool, not a pipeline stage.

Run 002 sized the whole blocking backlog at +0.005 F0.5 ceiling, so this only
earns a place in the pipeline if it clears the kill criterion below on train.

Memory rules (from the RTX 3050 review):
  * GPU: index shards stay on the CPU and are moved to the GPU ONE AT A TIME.
    Peak = shard + (query chunk x shard) scores, sized from free VRAM
    (6 GB card: ~1.5 GB shard + ~2 GB scores; the 4 GB laptop card gets smaller shards).
  * RAM: embeddings are streamed to fp16 .npy files while encoding and read back
    with np.load(mmap_mode="r") — 10.3M x 384 x 2 B = 7.9 GB never sits in RAM.
  * Do not run this concurrently with CPU blocking on a 16-25 GB machine.

    python src/gpu/embed_knn.py --measure 20000       # train recall gain vs pass-A cache
Kill criterion: test-weighted F0.5 ceiling gain < +0.002 -> DROP (0.5pp recall ~ 0.001 F0.5).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"   # MIT, 118M params, 384-d
DIM = 384
KILL_CEILING_GAIN = 0.002


def _torch():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def encode_to_npy(texts, path, batch=512, model_name=MODEL):
    """Encode texts ('query: ' prefix, e5 convention for symmetric matching) straight to an
    fp16 .npy on disk. Returns a read-only memmap. Constant RAM regardless of len(texts)."""
    from sentence_transformers import SentenceTransformer
    torch, dev = _torch()
    m = SentenceTransformer(model_name, device=dev)
    if dev == "cuda":
        m = m.half()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), DIM))
    step = batch * 64
    for a in range(0, len(texts), step):
        chunk = ["query: " + t for t in texts[a:a + step]]
        out[a:a + len(chunk)] = m.encode(chunk, batch_size=batch, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    out.flush()
    del out
    return np.load(path, mmap_mode="r")


def _plan(n_index, q_chunk, vram_budget):
    """Rows per shard so that shard (fp16) + scores (q_chunk x shard, fp16) fit the budget."""
    per_row = DIM * 2 + q_chunk * 2
    return max(1024, min(n_index, int(vram_budget // per_row)))


def knn(queries, index, k=10, q_chunk=512, shard_rows=None, device=None):
    """Exact top-k cosine neighbours. queries/index: (n, DIM) fp16 arrays or memmaps,
    L2-normalized. Only one index shard is resident on the device at a time.
    Returns (scores [nq, k] float32, indices [nq, k] int64) into `index`."""
    torch, dev = _torch()
    dev = device or dev
    if shard_rows is None:
        free = torch.cuda.mem_get_info()[0] if dev == "cuda" else 2 * 1024 ** 3
        shard_rows = _plan(len(index), q_chunk, 0.7 * free)
    nq = len(queries)
    k = min(k, len(index))
    best_s = torch.full((nq, k), -2.0, dtype=torch.float32)
    best_i = torch.full((nq, k), -1, dtype=torch.int64)
    dt = torch.float16 if dev == "cuda" else torch.float32
    for s0 in range(0, len(index), shard_rows):
        shard = torch.from_numpy(np.array(index[s0:s0 + shard_rows], copy=True)).to(dev, dt)
        for q0 in range(0, nq, q_chunk):
            q = torch.from_numpy(np.array(queries[q0:q0 + q_chunk], copy=True)).to(dev, dt)
            s, i = (q @ shard.T).float().topk(min(k, shard.shape[0]), dim=1)
            s, i = s.cpu(), i.cpu() + s0
            cs = torch.cat([best_s[q0:q0 + len(q)], s], 1)
            ci = torch.cat([best_i[q0:q0 + len(q)], i], 1)
            top, j = cs.topk(k, dim=1)
            best_s[q0:q0 + len(q)], best_i[q0:q0 + len(q)] = top, ci.gather(1, j)
            del q, s, i
        del shard
        if dev == "cuda":
            torch.cuda.empty_cache()
    return best_s.numpy(), best_i.numpy()


def f05_ceiling(r):
    return 1.25 * r / (0.25 + r)


def measure(n, k=10, work=None):
    """Recall of pass-A candidates vs pass-A ∪ embedding top-k on a train sample, per country,
    weighted to the test mix. Needs the pass-A cache for the same sample (run_v2/run_pipeline)."""
    import pandas as pd
    import data as D
    import ingest
    from normalize import add_blocking_columns
    work = Path(work or C.INTERIM / "emb")
    work.mkdir(parents=True, exist_ok=True)
    cache = C.INTERIM / f"cands_train_k{C.TOP_K}_df{C.BLOCK_MAX_DF}_mdf{C.BLOCK_MIN_DF}_ctry1_n{n}.parquet"
    if not cache.exists():
        raise SystemExit(f"{cache.name} missing — run `python src/run_v2.py --sample {n} --train-only` first")
    pairs = pd.read_parquet(cache)
    s1 = D.read_source(C.TRAIN_S1).sample(n=n, random_state=C.SEED).reset_index(drop=True)
    s1 = add_blocking_columns(s1)
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID])
    have = pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
    res = {}
    full = ingest.blocking_frame([C.TRAIN_S2, C.TRAIN_S3])
    for country in sorted(s1[C.COUNTRY].astype(str).unique()):
        q = s1[s1[C.COUNTRY].astype(str) == country]
        idx = full[full[C.COUNTRY].astype(str) == country].reset_index(drop=True)
        t = time.time()
        qe = encode_to_npy(q["_blob"].tolist(), work / f"q_{country}_{n}.npy")
        ie_path = work / f"idx_train_{country}.npy"
        ie = np.load(ie_path, mmap_mode="r") if ie_path.exists() and len(np.load(ie_path, mmap_mode="r")) == len(idx) \
            else encode_to_npy(idx["_blob"].tolist(), ie_path)
        t_enc = time.time() - t
        t = time.time()
        _, nn = knn(qe, ie, k=k)
        t_knn = time.time() - t
        ids = idx[C.ID].astype(str).to_numpy()
        tot = base = union = 0
        for sid, row in zip(q[C.ID], nn):
            tr = truth.get(sid, set())
            a = have.get(sid, set())
            tot += len(tr)
            base += len(tr & a)
            union += len(tr & (a | set(ids[row[row >= 0]])))
        res[country] = dict(recall_A=base / tot, recall_union=union / tot, encode_s=t_enc, knn_s=t_knn)
        print(country, res[country], flush=True)
        del idx
    mix = {"US": 0.383, "India": 0.468}
    w = {c: mix.get(c, 0) for c in res}
    ra = sum(w[c] * res[c]["recall_A"] for c in res) / sum(w.values())
    ru = sum(w[c] * res[c]["recall_union"] for c in res) / sum(w.values())
    gain = f05_ceiling(ru) - f05_ceiling(ra)
    verdict = "KEEP" if gain >= KILL_CEILING_GAIN else "DROP"
    print(f"test-weighted recall {ra:.4f} -> {ru:.4f}; F0.5 ceiling gain {gain:+.4f} -> {verdict} "
          f"(kill < +{KILL_CEILING_GAIN})")
    return res, gain, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", type=int, required=True, help="train S1 sample size (needs its pass-A cache)")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    measure(a.measure, a.k)
