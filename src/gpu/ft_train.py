"""Training variants for the band reranker: LoRA (option 3) and the listwise loss (option 4).

Option 3 — LoRA over ALL encoder layers. laya_rr's default freezes all but the top 8 of mmBERT's 22
layers to fit 4 GB, which leaves the lower layers (where script/tokenisation handling lives) untouched.
LoRA adapters on Wqkv/Wo/Wi reach every layer with ~5M trainable params; laya's decision head trains
fully. `merge_lora` folds the adapters back, so the saved checkpoint is a plain laya checkpoint.

Option 4 — listwise loss. The ground truth is a partition: an S2/S3 record matches AT MOST ONE S1
entity. Grouping by cand_id, the S1 candidates of one record compete in a softmax with a "none" slot
(logit 0): one positive -> its slot, no positive -> "none". A group of one reduces exactly to BCE, so
the loss is BCE where records have no competition and a ranking loss where they do.
"""
import numpy as np
import pandas as pd

LORA_TARGETS = ("Wqkv", "Wo", "Wi")           # ModernBERT attention qkv/out and MLP in/out projections


# ------------------------------------------------------------------ option 3: LoRA
def apply_lora(model, r, alpha=None, dropout=0.05, targets=LORA_TARGETS, checkpointing=False):
    """Wrap model.encoder with LoRA adapters. The encoder's base weights (embedding table included) are
    frozen by peft; everything outside the encoder (laya's decision head) stays trainable.
    `checkpointing` recomputes encoder activations in backward: LoRA backpropagates through all 22
    layers, and bs 32 otherwise peaks at 3.8 GiB, which spills to shared memory on a 4 GB card."""
    from peft import LoraConfig, get_peft_model
    was_training = model.training
    model.requires_grad_(True)
    if checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules=list(targets),
                     bias="none")
    model.encoder = get_peft_model(model.encoder, cfg)
    model.train(was_training)       # peft builds its dropout modules in train mode even on an eval model
    return model


def merge_lora(model):
    """Fold the adapters into the base weights; the state dict gets laya's original keys back."""
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
    return model


# ------------------------------------------------------------------ option 4: listwise
def singleton_keys(keys, n_new):
    """`keys` extended by n_new keys that are each their own group (augmented rows -> plain BCE).
    No NUL bytes: pandas hashes object strings as C strings, so '\\x00aug1' and '\\x00aug2' collide."""
    return np.concatenate([np.asarray(keys, dtype=object),
                           np.array([f"__aug__{i}" for i in range(n_new)], dtype=object)])


def _codes(keys):
    """pd.factorize codes; a missing key (NaN/None) would get -1, which torch indexing reads as the LAST
    group and silently merges into it (e.g. pairs mixed from files with and without cand_id)."""
    codes, uniq = pd.factorize(np.asarray(keys, dtype=object), sort=False)
    if (codes < 0).any():
        raise ValueError(f"{int((codes < 0).sum()):,} rows have no group key; every pair needs one for "
                         f"the listwise loss (do all --pairs files have the group column?)")
    return codes, uniq


def group_batches(keys, order, bs):
    """Row-index batches of <= bs rows that never split a group (a group larger than bs is its own
    batch). Groups are visited in the order their first row appears in `order`."""
    keys = np.asarray(keys, dtype=object)
    codes, _ = _codes(keys[order])
    by_group = pd.Series(order).groupby(codes, sort=True).apply(np.asarray)
    cur, n = [], 0
    for rows in by_group:
        if cur and n + len(rows) > bs:
            yield np.concatenate(cur)
            cur, n = [], 0
        cur.append(rows)
        n += len(rows)
    if cur:
        yield np.concatenate(cur)


def listwise_loss(z, keys, y):
    """Mean over groups of the cross-entropy of softmax([z_group..., 0]) against the group's target:
    uniform over its positives, or the "none" slot when it has none. z: (n,) logits, keys: (n,) group
    keys, y: (n,) 0/1 labels (torch)."""
    import torch
    codes, uniq = _codes(keys)
    pos = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    n_groups, k = len(uniq), int(pos.max()) + 1 if len(pos) else 0
    ci = torch.as_tensor(codes, device=z.device)
    pi = torch.as_tensor(pos, device=z.device)
    logits = torch.full((n_groups, k + 1), float("-inf"), device=z.device, dtype=z.dtype)
    logits[:, k] = 0.0                                               # the "none" slot
    logits = logits.index_put((ci, pi), z)
    target = torch.zeros((n_groups, k + 1), device=z.device, dtype=z.dtype)
    target = target.index_put((ci, pi), y.to(z.dtype))
    n_pos = target[:, :k].sum(1, keepdim=True)
    target = torch.where(n_pos > 0, target / n_pos.clamp(min=1), target)
    target[:, k] = (n_pos.squeeze(1) == 0).to(z.dtype)
    logp = torch.log_softmax(logits, dim=1).masked_fill(target == 0, 0.0)  # 0 * -inf would be NaN
    return -(target * logp).sum(1).mean()
