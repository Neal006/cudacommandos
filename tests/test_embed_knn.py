"""python tests/test_embed_knn.py — streamed-shard kNN equals brute force (GPU if present, and CPU)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu.embed_knn import DIM, knn  # noqa: E402
import torch  # noqa: E402

rng = np.random.default_rng(0)
idx = rng.standard_normal((20_000, DIM)).astype(np.float32)
idx /= np.linalg.norm(idx, axis=1, keepdims=True)
q = idx[rng.choice(len(idx), 300, replace=False)] + 0.05 * rng.standard_normal((300, DIM)).astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)
ref = np.argsort(-(q @ idx.T), axis=1)[:, :10]

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:          # the index comes back as a read-only memmap
    np.save(Path(d) / "i.npy", idx.astype(np.float16))
    mm = np.load(Path(d) / "i.npy", mmap_mode="r")
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        s, i = knn(q.astype(np.float16), mm, k=10, q_chunk=64, shard_rows=3_000, device=dev)  # 7 shards
        top1 = (i[:, 0] == ref[:, 0]).mean()
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i, ref)])
        assert top1 == 1.0 and overlap > 0.97, (dev, top1, overlap)     # fp16 may swap near-ties
        assert (np.diff(s, axis=1) <= 1e-6).all()                        # sorted descending
        print(f"{dev}: top1 {top1:.3f} top10-overlap {overlap:.3f}")
    del mm                                          # Windows cannot delete a mapped file
print("embed_knn ok")
