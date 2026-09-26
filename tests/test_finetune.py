"""python tests/test_finetune.py — fine-tuning options 1-5 for the band reranker, data-free.

  1 band-matched pairs   ft_data.select_band
  2 in-data augmentation ft_data.parse/render, AUG_OPS, augment_train
  3 LoRA                 ft_train.apply_lora / merge_lora on a tiny random ModernBERT (no download)
  4 listwise loss        ft_train.group_batches / listwise_loss
  5 distillation         distill.soft_targets / union_entities
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from gpu import distill as DS  # noqa: E402
from gpu import ft_data as FD  # noqa: E402
from gpu import ft_train as FT  # noqa: E402

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 2. augmentation
native = "name: श्री गणेश ट्रेडर्स प्राइवेट लिमिटेड | shri ganesh traders addr: 12 station road pune"
latin = "name: acme global traders pvt ltd addr: 4 main street austin"
for s in (native, latin, "name: solo addr: ", "name:  addr: x"):
    assert FD.render(*FD.parse(s)) == s, s
assert FD.parse(native) == ("श्री गणेश ट्रेडर्स प्राइवेट लिमिटेड", "shri ganesh traders", "12 station road pune")
assert FD.parse(latin) == ("acme global traders pvt ltd", "", "4 main street austin")

r = np.random.default_rng(1)
assert FD.AUG_OPS["native_only"](native, r) == "name: श्री गणेश ट्रेडर्स प्राइवेट लिमिटेड addr: 12 station road pune"
assert FD.AUG_OPS["latin_only"](native, r) == "name: shri ganesh traders addr: 12 station road pune"
assert FD.AUG_OPS["native_only"](latin, r) == latin            # no transliteration -> unchanged
assert FD.AUG_OPS["strip_legal"](latin, r) == "name: acme global traders addr: 4 main street austin"
dropped = FD.parse(FD.AUG_OPS["drop_name_token"](latin, r))[0].split()
assert len(dropped) == 4 and set(dropped) < set("acme global traders pvt ltd".split())
assert FD.AUG_OPS["drop_name_token"]("name: solo addr: x", r) == "name: solo addr: x"   # never empties a name
sh = FD.parse(FD.AUG_OPS["shuffle_addr"](latin, r))[2].split()
assert sorted(sh) == sorted("4 main street austin".split())
for op in FD.AUG_OPS.values():                                   # every op keeps the serialize() shape
    for s in (native, latin):
        FD.parse(op(s, np.random.default_rng(3)))

a = np.array([latin, native, latin, native, latin], dtype=object)
b = a.copy()
y = np.array([1, 0, 1, 1, 0], dtype=np.float32)
g = np.array(["e1", "e1", "e2", "e3", "e3"], dtype=object)
tr, va = np.array([0, 1, 2]), np.array([3, 4])
a0, b0 = a.copy(), b.copy()
a2, b2, y2, g2, tr2 = FD.augment_train(a, b, y, g, tr, rate=1.0, rng=np.random.default_rng(0))
assert (a == a0).all() and (b == b0).all(), "inputs must not be mutated"
assert len(a2) == len(a) + 3 and len(tr2) == len(tr) + 3
assert set(tr2[-3:]) == {5, 6, 7} and not (set(tr2) & set(va)), "augmented rows are train-only"
src = [0, 1, 2]                                                   # one copy per train row at rate 1.0
assert sorted(y2[5:].tolist()) == sorted(y[src].tolist()) and sorted(g2[5:].tolist()) == sorted(g[src].tolist())
assert (a2[:5] == a).all() and (y2[:5] == y).all()
same = FD.augment_train(a, b, y, g, tr, rate=0.0, rng=np.random.default_rng(0))
assert len(same[0]) == len(a) and (same[4] == tr).all()

# ---------------------------------------------------------------- 1. band-matched pairs
p1 = np.concatenate([np.full(100, 0.5), np.full(900, 0.01), np.full(1000, 0.99)])
keep = FD.select_band(p1, (0.2, 0.8), outside_frac=0.1, rng=np.random.default_rng(0))
assert set(range(100)) <= set(keep.tolist()), "every band pair is kept"
assert 150 <= (keep >= 100).sum() <= 250, (keep >= 100).sum()    # ~10% of the 1900 outside
assert (np.diff(keep) > 0).all(), "sorted, unique"
assert (FD.select_band(p1, (0.2, 0.8), 0.0, np.random.default_rng(0)) == np.arange(100)).all()

# ---------------------------------------------------------------- 4. listwise
keys = np.array(["c1", "c1", "c1", "c2", "c3", "c3", "c4"] * 5, dtype=object)
keys = np.array([f"{k}-{i // 7}" for i, k in enumerate(keys)], dtype=object)
order = np.random.default_rng(0).permutation(len(keys))
batches = list(FT.group_batches(keys, order, bs=8))
seen = np.concatenate(batches)
assert sorted(seen.tolist()) == list(range(len(keys))), "every row exactly once"
for bt in batches:
    assert len(bt) <= 8
    for k in set(keys[bt]):
        assert set(np.where(keys == k)[0]) <= set(bt.tolist()), "a group never straddles batches"

# augmented rows get singleton keys; regression: NUL-prefixed keys all hashed to one 1,620-row "group"
ext = FT.singleton_keys(np.array(["c1", "c1"], dtype=object), 100)
assert len(ext) == 102 and len(set(ext)) == 101
big = list(FT.group_batches(ext, np.arange(102), bs=8))
assert max(map(len, big)) <= 8 and len(big) >= 13, [len(x) for x in big]

# regression: a missing key (pairs mixed from files with/without cand_id) must fail, not merge into
# the last group through torch's negative-index wraparound
for fn in (lambda k: FT.listwise_loss(torch.zeros(3), k, torch.zeros(3)),
           lambda k: list(FT.group_batches(k, np.arange(3), 8))):
    try:
        fn(np.array(["g1", np.nan, "g2"], dtype=object))
        raise AssertionError("missing group key must raise")
    except ValueError:
        pass

z = torch.tensor([2.0, -1.0, 0.5, 3.0])
yy = torch.tensor([1.0, 0.0, 0.0, 1.0])
single = FT.listwise_loss(z, np.array(["a", "b", "c", "d"], dtype=object), yy)
bce = torch.nn.functional.binary_cross_entropy_with_logits(z, yy)
assert torch.allclose(single, bce, atol=1e-6), (single, bce)     # singletons + none slot == BCE
gk = np.array(["a", "a", "a", "b"], dtype=object)
lw = FT.listwise_loss(z, gk, yy)
ref_a = -torch.log_softmax(torch.tensor([2.0, -1.0, 0.5, 0.0]), 0)[0]
ref_b = -torch.log_softmax(torch.tensor([3.0, 0.0]), 0)[0]
assert torch.allclose(lw, (ref_a + ref_b) / 2, atol=1e-6), lw
two = FT.listwise_loss(torch.tensor([1.0, 1.0]), np.array(["a", "a"], dtype=object), torch.tensor([1.0, 1.0]))
assert torch.allclose(two, -torch.log_softmax(torch.tensor([1.0, 1.0, 0.0]), 0)[:2].mean(), atol=1e-6)
none = FT.listwise_loss(torch.tensor([1.0, -2.0]), np.array(["a", "a"], dtype=object), torch.tensor([0.0, 0.0]))
assert torch.allclose(none, -torch.log_softmax(torch.tensor([1.0, -2.0, 0.0]), 0)[2], atol=1e-6)
z.requires_grad_(True)
FT.listwise_loss(z, gk, yy).backward()
assert z.grad is not None and torch.isfinite(z.grad).all()

# ---------------------------------------------------------------- 3. LoRA on a tiny random ModernBERT
from laya.common import DecisionModel  # noqa: E402
from transformers import ModernBertConfig, ModernBertModel  # noqa: E402

torch.manual_seed(0)
cfg = ModernBertConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                       num_attention_heads=4, max_position_embeddings=64, pad_token_id=0,
                       bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
model = DecisionModel(ModernBertModel(cfg), head_layers=1, n_act=2).eval()
keys0 = set(model.state_dict())
batch = dict(input_ids=torch.randint(3, 64, (3, 12)), attention_mask=torch.ones(3, 12, dtype=torch.long),
             marker_pos=torch.tensor([[2, 5]] * 3), marker_mask=torch.ones(3, 2, dtype=torch.bool),
             qtype=torch.full((3,), 2))
FT.apply_lora(model, r=4)
assert not any(m.training for m in model.modules()), "apply_lora keeps an eval model in eval mode (no dropout)"
names ={n for n, p in model.named_parameters() if p.requires_grad}
assert names and all("lora_" in n or not n.startswith("encoder.") for n in names), names
assert not any("tok_embeddings" in n for n in names), "the embedding table stays frozen"
assert any(n.startswith("head.") for n in names) and any(n.startswith("scorer.") for n in names)
for n, p in model.named_parameters():                            # make the adapter non-trivial
    if "lora_B" in n:
        torch.nn.init.normal_(p, std=0.05)
with torch.no_grad():
    before = model(**batch)[0]
FT.merge_lora(model)
with torch.no_grad():
    after = model(**batch)[0]
assert set(model.state_dict()) == keys0, "merged model has laya's original state-dict keys"
assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()

# ---------------------------------------------------------------- 5. distillation
t = DS.soft_targets(np.array([1, 0, 1, 0.0]), np.array([0.9, 0.2, 0.1, 0.7]), alpha=0.5)
assert np.allclose(t, [0.95, 0.1, 0.55, 0.35]) and t.dtype == np.float32
assert np.allclose(DS.soft_targets(np.array([1.0]), np.array([0.3]), alpha=1.0), [1.0])
try:
    DS.soft_targets(np.array([1.0]), np.array([0.3]), alpha=1.5)
    raise AssertionError("alpha outside [0, 1] must fail")
except ValueError:
    pass
assert DS.union_entities(["b", "a"], ["c", "a"]) == ["a", "b", "c"]
assert DS.unseen_by_teacher(np.array(["e1", "e2", "e3"]), ["e2"]).tolist() == [True, False, True]

# ---------------------------------------------------------------- shared held-out evaluation
assert FD.eval_leak(["e1", "e2"], ["e3"]) == 0
assert FD.eval_leak(["e1", "e2"], ["e2", "e9"]) == 1

print("test_finetune: all checks passed")
