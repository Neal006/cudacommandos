"""python tests/smoke_finetune.py [LAYA_BASE_DIR] — fine-tuning options end to end on synthetic records (GPU, ~3 min).

1. laya with augmentation (2) + LoRA over all layers (3) + listwise loss by cand_id (4)
2. distil that model into e5-small (5): soft targets, e5 augmentation, entities.txt union
Both are scored through run_v2.rerank_feature, which picks the backend from meta.json.
Option 1 (band pairs) needs the dataset and a finished GBDT run; its logic is unit-tested in test_finetune.py.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.metrics import roc_auc_score  # noqa: E402

import run_v2  # noqa: E402
from gpu import distill as DS  # noqa: E402
from gpu import laya_rr as LR  # noqa: E402

base = Path(sys.argv[1]) if len(sys.argv) > 1 else LR.BASE_DIR
rng = np.random.default_rng(2)
NATIVE = {"sharma": "शर्मा", "ganesh": "गणेश", "lakshmi": "लक्ष्मी", "patel": "पटेल", "shri": "श्री", "royal": "रॉयल"}
FIRST = list(NATIVE) + ["acme", "global", "sunrise", "orient", "metro", "green", "blue", "star", "nova", "prime"]
KIND = ["traders", "sweets", "motors", "textiles", "pharma", "foods", "steel", "bakery"]
CITY = ["mumbai", "delhi", "chennai", "pune", "paris", "lyon", "austin", "kolkata"]


def entity(first=None):
    w = [first or rng.choice(FIRST), rng.choice(FIRST), rng.choice(KIND)]
    return dict(words=w, addr=f"{rng.integers(1, 999)} {rng.choice(['mg road', 'rue de rivoli', 'main street'])} "
                              f"{rng.choice(CITY)}")


def record(e, noisy):
    words = list(e["words"])
    if noisy and rng.random() < 0.3:
        words.pop(int(rng.integers(len(words))))
    latin = " ".join(words) + (str(rng.choice(["", " pvt ltd", " llc"])) if noisy else "")
    native = noisy and rng.random() < 0.5 and any(w in NATIVE for w in words)
    name = " ".join(NATIVE.get(w, w) for w in latin.split()) if native else latin
    return {"_name": name, "_core_name": name, "_tl_core": latin, "_addr": e["addr"]}


def world(n_cands, tag):
    """Each S2 record competes for 3 S1 entities: its true one (70% of records) and 2 look-alikes."""
    L, R, rows = [], [], []
    for c in range(n_cands):
        true = entity()
        cid = f"S2-{tag}{c}"
        R.append((cid, record(true, noisy=True)))
        s1s = [(true, int(rng.random() < 0.7))] + [(entity(first=true["words"][0]), 0) for _ in range(2)]
        for j, (e, y) in enumerate(s1s):
            sid = f"S1-{tag}{c}-{j}"
            L.append((sid, record(e, noisy=False)))
            rows.append((sid, cid, y))
    Lt = pd.DataFrame([r for _, r in L], index=[i for i, _ in L])
    Rt = pd.DataFrame([r for _, r in R], index=[i for i, _ in R])
    return Lt, Rt, pd.DataFrame(rows, columns=["s1_id", "cand_id", "y"])


def rr_auc(model_dir, Lv, Rv, pv):
    p1 = np.full(len(pv), 0.5)
    rr = run_v2.rerank_feature(model_dir, pv[["s1_id", "cand_id"]], p1, Lv, Rv, (0.2, 0.8))
    assert ((rr >= 0) & (rr <= 1)).all()
    return roc_auc_score(pv["y"], rr)


Lt, Rt, pt = world(1200, "t")
Lv, Rv, pv = world(250, "v")
a, b = LR.serialize(Lt, pt["s1_id"]), LR.serialize(Rt, pt["cand_id"])

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    teacher = d / "rr_laya_ft"
    meta = LR.train(a, b, pt["y"].to_numpy(), teacher, base=base, bs=32, valid_frac=0.1, groups=pt["s1_id"],
                    augment=0.5, lora=16, listwise=1.0, group_keys=pt["cand_id"].to_numpy())
    assert meta["lora"] == 16 and meta["augment"] == 0.5 and meta["listwise"] == 1.0
    import laya
    assert abs(float(laya.load(str(teacher)).temperature[LR.NOUL]) - meta["temperature"]) < 1e-6
    auc_t = rr_auc(teacher, Lv, Rv, pv)
    print(f"[laya aug+lora+listwise] held-out AUC {auc_t:.4f}")
    assert auc_t > 0.8, auc_t

    # distil on a DIFFERENT set: distill() refuses pairs from entities the teacher trained on
    def to_parquet(L, R, p, path):
        pd.DataFrame({"s1_id": p["s1_id"], "cand_id": p["cand_id"], "a": LR.serialize(L, p["s1_id"]),
                      "b": LR.serialize(R, p["cand_id"]), "y": p["y"], "p1": 0.5}).to_parquet(path)
        return path
    try:
        DS.distill(teacher, [to_parquet(Lt, Rt, pt, d / "own.parquet")], d / "nope", bs=64)
        raise AssertionError("distilling on the teacher's own entities must be refused")
    except SystemExit:
        pass
    Ld, Rd, pd_ = world(1200, "d")
    student = d / "rr_e5_kd"
    sm = DS.distill(teacher, [to_parquet(Ld, Rd, pd_, d / "kd.parquet")], student, alpha=0.5, bs=64, augment=0.3)
    sm_disk = json.loads((student / "meta.json").read_text())
    assert sm_disk["kind"] == "e5" and sm_disk["teacher"] == str(teacher) and sm_disk["alpha"] == 0.5
    ents = set((student / "entities.txt").read_text(encoding="utf-8").split())
    assert set((teacher / "entities.txt").read_text(encoding="utf-8").split()) <= ents
    auc_s = rr_auc(student, Lv, Rv, pv)
    print(f"[e5 distilled from laya] held-out AUC {auc_s:.4f} (teacher {auc_t:.4f})")
    assert auc_s > 0.7, auc_s

    # the shared held-out comparison: both models on one frozen file, leak-checked
    from gpu import ft_data as FD
    table = FD.evaluate([teacher, student], to_parquet(Lv, Rv, pv, d / "heldout.parquet"))
    assert list(table["model"]) == ["rr_laya_ft", "rr_e5_kd"] and (table["auc"] > 0.7).all()
    try:
        FD.evaluate([teacher], d / "kd.parquet")                    # the student trained on these
        FD.evaluate([student], d / "kd.parquet")
        raise AssertionError("evaluating on a model's own training entities must be refused")
    except SystemExit:
        pass

print("smoke_finetune: passed")
