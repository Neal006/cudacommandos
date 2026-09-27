"""python tests/smoke_laya_rr.py [BASE_DIR] — laya reranker end to end on synthetic records (GPU, ~2 min).

fine-tune on synthetic duplicates -> save -> run_v2.rerank_feature dispatches to laya via meta.json
-> held-out AUC -> the saved dir is also a plain laya checkpoint (laya.load + system_one).
Needs the laya-multilingual checkpoint (fetched to models/laya_ml_base if BASE_DIR is not given).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import roc_auc_score  # noqa: E402

import run_v2  # noqa: E402
from gpu import laya_rr as LR  # noqa: E402

base = Path(sys.argv[1]) if len(sys.argv) > 1 else LR.BASE_DIR
rng = np.random.default_rng(1)
FIRST = ["sharma", "tata", "acme", "global", "sunrise", "lakshmi", "orient", "royal", "metro", "green",
         "ganesh", "patel", "blue", "star", "nova", "delta", "prime", "apex", "zenith", "crown"]
KIND = ["traders", "sweets", "motors", "textiles", "pharma", "foods", "steel", "electricals", "consultancy", "bakery"]
CITY = ["mumbai", "delhi", "chennai", "pune", "paris", "lyon", "austin", "denver", "kolkata", "surat"]
ROAD = ["mg road", "rue de rivoli", "main street", "station road", "park avenue", "gandhi nagar"]


def entity():
    return dict(name=f"{rng.choice(FIRST)} {rng.choice(FIRST)} {rng.choice(KIND)}",
                addr=f"{rng.integers(1, 999)} {rng.choice(ROAD)} {rng.choice(CITY)}")


def noisy(e):
    """A duplicate: legal suffix, dropped character, abbreviated address."""
    name = e["name"] + rng.choice(["", " pvt ltd", " llc", " sarl", " limited"])
    if rng.random() < 0.3:
        i = rng.integers(len(name))
        name = name[:i] + name[i + 1:]
    addr = e["addr"].replace("road", "rd").replace("street", "st") if rng.random() < 0.5 else e["addr"]
    return dict(name=name, addr=addr)


def pairs(n):
    """Half duplicates, half hard negatives (same name, other address or vice versa)."""
    rows = []
    for i in range(n):
        e = entity()
        if i % 2:
            rows.append((e, noisy(e), 1))
        else:
            f = entity()
            other = dict(name=e["name"], addr=f["addr"]) if rng.random() < 0.5 else dict(name=f["name"], addr=e["addr"])
            rows.append((e, noisy(other), 0))
    return rows


def table(recs, prefix):
    ids = [f"{prefix}-{i}" for i in range(len(recs))]
    df = pd.DataFrame({"_name": [r["name"] for r in recs], "_addr": [r["addr"] for r in recs]}, index=ids)
    df["_core_name"] = df["_name"]
    df["_tl_core"] = df["_name"]
    return df


tr, va = pairs(3000), pairs(600)
L_tr, R_tr = table([p[0] for p in tr], "S1"), table([p[1] for p in tr], "S2")
a, b = LR.serialize(L_tr, L_tr.index), LR.serialize(R_tr, R_tr.index)
y = np.array([p[2] for p in tr])

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "rr_laya"
    meta = LR.train(a, b, y, out, base=base, bs=32, lr=5e-5, valid_frac=0.1,
                    groups=[f"g{i // 2}" for i in range(len(y))])
    assert meta["kind"] == "laya" and (out / "entities.txt").exists()
    for f in ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder", "meta.json"):
        assert (out / f).exists(), f

    # run_v2's own feature path: backend picked from meta.json, -1 outside the band
    L_va, R_va = table([p[0] for p in va], "S1"), table([p[1] for p in va], "S2")
    frame = pd.DataFrame({"s1_id": L_va.index, "cand_id": R_va.index})
    p1 = np.where(np.arange(len(va)) % 10 == 0, 0.95, 0.5)          # 10% outside the band
    rr = run_v2.rerank_feature(out, frame, p1, L_va, R_va, (0.2, 0.8))
    inb = p1 <= 0.8
    assert (rr[~inb] == -1).all() and ((rr[inb] >= 0) & (rr[inb] <= 1)).all()
    y_va = np.array([p[2] for p in va])
    auc = roc_auc_score(y_va[inb], rr[inb])
    print(f"held-out AUC {auc:.4f} (valid AUC at train time {meta['valid_auc']:.4f})")
    assert auc > 0.85, auc

    # the dir is a normal laya checkpoint: laya.load + system_one answer with the fitted temperature
    import laya
    ag = laya.load(str(out))
    served_t = float(ag.temperature[LR.NOUL])
    print(f"fitted T {meta['temperature']:.3f}, laya.load serves {served_t:.3f}")
    assert abs(served_t - meta["temperature"]) < 1e-6, "validated and served temperature differ"
    assert meta["base_revision"] == LR.BASE_REVISION
    q = {"same": {"type": "noul", "instructions": LR.QUESTION["ins"], "criteria": LR.QUESTION["crit"]}}
    s = "record A: " + a[1] + "\nrecord B: " + b[1]
    print("laya.load(out).system_one:", ag.system_one(s, q)["answers"]["same"])

print("smoke_laya_rr: passed")
