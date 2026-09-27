"""CPU smoke test for the reranker's --base / bf16 path (needs torch + a cached HF model).

    HF_HUB_OFFLINE=1 AMLC_CPU_BF16=1 python tests/test_reranker_cpu.py
    AMLC_RR_SMOKE_BASE=BAAI/bge-reranker-v2-m3 python tests/test_reranker_cpu.py   # on the box

Trains 1 epoch on 96 synthetic pairs, checks meta.json records the backbone,
and that the saved model scores through the same entry point run_v2 uses.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu import reranker as RR  # noqa: E402
from gpu import reranker_module  # noqa: E402

base = os.environ.get("AMLC_RR_SMOKE_BASE", "BAAI/bge-small-en-v1.5")
torch, dev, amp = RR._torch()
print(f"device {dev}, autocast {amp}, threads {torch.get_num_threads()}, base {base}")

names = ["ram medicals", "sai traders", "le petit cafe", "global exports", "city bakery", "blue lotus"]
a, b, y, g = [], [], [], []
for i in range(96):
    n = names[i % len(names)]
    same = i % 2 == 0
    other = n if same else names[(i + 1) % len(names)]
    a.append(f"name: {n} pvt ltd addr: {i} mg road")
    b.append(f"name: {other} addr: {i} m.g. rd")
    y.append(int(same))
    g.append(f"S1-{i // 4}")

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    out = Path(d) / "rr"
    meta = RR.train(a, b, y, out, epochs=1, bs=16, lr=5e-5, valid_frac=0.25, groups=g, base=base)
    saved = json.loads((out / "meta.json").read_text())
    assert saved["base"] == base and saved["kind"] == "e5", saved
    assert (out / "entities.txt").exists(), "leak guard file missing"
    mod = reranker_module(out)
    assert mod is RR, mod
    p = mod.score(out, np.array(a[:8], dtype=object), np.array(b[:8], dtype=object))
    assert p.shape == (8,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), p
print(f"reranker cpu ok: valid logloss {meta['valid_logloss']:.3f}, {meta['seconds']:.0f}s")
