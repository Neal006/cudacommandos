"""python tests/test_laya_rr.py — laya reranker input format, calibration and backend dispatch.

Data-free and network-free: a character-level fake tokenizer stands in for mmBERT's.
Needs `laya` installed (for its build_sequence), not its checkpoint.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu  # noqa: E402
from gpu import laya_rr as LR  # noqa: E402


class FakeTok:
    """Just enough of a HF tokenizer for laya.common.build_sequence and laya_rr."""
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 4
    mask_token = "<mask>"

    def _one(self, text, truncation=False, max_length=None, **_):
        ids = [10 + (ord(c) % 1000) for c in text]
        return ids[:max_length] if truncation and max_length else ids

    def __call__(self, text, **kw):
        if isinstance(text, list):
            return {"input_ids": [self._one(t, **kw) for t in text]}
        return {"input_ids": self._one(text, **kw)}


tok = FakeTok()
enc = LR.PairEncoder(tok, head_max_len=256)

# 1. prefix is laya's own format: [CLS] question [SEP] <mask> false <mask> true [SEP]
assert enc.prefix[0] == tok.cls_token_id and enc.prefix[-1] == tok.sep_token_id
assert len(enc.markers) == 2, enc.markers
assert all(enc.prefix[m] == tok.mask_token_id for m in enc.markers)
assert enc.prefix.count(tok.mask_token_id) == 2

# 2. every row = prefix + record A + record B + [SEP]; markers identical across rows
a = ["name: acme ltd addr: 1 main st", "x" * 500, "name: a<mask>b"]
b = ["name: acme limited addr: main street", "name: short", "name: c"]
rows = enc.encode(a, b)
assert len(rows) == 3
for r in rows:
    assert r[:len(enc.prefix)] == enc.prefix
    assert r[-1] == tok.sep_token_id
    assert len(r) <= enc.max_len == len(enc.prefix) + 2 * LR.REC_MAX + 1, len(r)

# 3. a long record A is capped on its own, so record B is never truncated away
tail = rows[1][len(enc.prefix):-1]
b_ids = tok("\nrecord B: " + b[1])["input_ids"]
assert tail[-len(b_ids):] == b_ids, "record B lost behind a long record A"
assert len(tail) - len(b_ids) <= LR.REC_MAX

# 4. a literal mask token inside record text never becomes an extra marker
assert rows[2].count(tok.mask_token_id) == 2

# 5. collate pads to the longest row; attention covers exactly the real tokens
batch = LR.collate(rows, enc.markers, tok.pad_token_id)
n, L = batch["input_ids"].shape
assert n == 3 and L == max(map(len, rows))
assert batch["attention_mask"].sum(1).tolist() == [len(r) for r in rows]
assert batch["marker_pos"].tolist() == [enc.markers] * 3
assert batch["marker_mask"].all() and (batch["qtype"] == 2).all()

# 6. temperature fit recovers a known over-confidence factor, and stays inside the clamp
rng = np.random.default_rng(0)
z = rng.normal(0, 4, 20000)                        # logit(true) - logit(false)
y = (rng.random(len(z)) < 1 / (1 + np.exp(-z / 2.0))).astype(np.float32)
t = LR.fit_temperature(z, y)
assert abs(t - 2.0) < 0.15, t
t_sep = LR.fit_temperature(np.array([3.0, -3.0]), np.array([1.0, 0.0]))   # separable -> sharpen
assert LR.T_MIN <= t_sep <= 1.0, t_sep
# laya.load clamps temperatures to its own range; a fit outside it would be validated at one T
# and served at another, so the search range must BE laya's range
from laya.common import TEMP_MAX, TEMP_MIN, clamp_temperature  # noqa: E402
assert (LR.T_MIN, LR.T_MAX) == (TEMP_MIN, TEMP_MAX)
assert clamp_temperature(t_sep) == t_sep and clamp_temperature(t) == t

# 7. run_v2 picks the backend from meta.json: laya dirs -> laya_rr, anything else -> e5 reranker
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    assert gpu.reranker_module(d).__name__ == "gpu.reranker"            # no meta.json (old dirs)
    (d / "meta.json").write_text(json.dumps({"base": "intfloat/multilingual-e5-small"}))
    assert gpu.reranker_module(d).__name__ == "gpu.reranker"
    (d / "meta.json").write_text(json.dumps({"kind": "laya"}))
    assert gpu.reranker_module(d).__name__ == "gpu.laya_rr"
    (d / "meta.json").write_text(json.dumps({"kind": "bogus"}))
    try:
        gpu.reranker_module(d)
        raise AssertionError("unknown kind must fail loudly")
    except ValueError:
        pass

# 8. both backends expose the same surface run_v2 calls
from gpu import reranker as RR  # noqa: E402
for m in (RR, LR):
    assert callable(m.serialize) and callable(m.score)
assert LR.serialize is RR.serialize, "one record serializer, so both rerankers see identical text"

print("test_laya_rr: all checks passed")
