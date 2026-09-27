"""GPU job 3: cross-encoder reranker for the uncertain band (Ditto-style pair classifier).

Matcher precision is the bottleneck (run 002 + OOF ~0.90 vs a ~0.99 blocking ceiling),
so this is where the GPU earns its keep: re-score only pairs stage 1 is unsure
about, and hand that score to stage 2 as a feature (src/run_v2.py --rerank).

  backbone  intfloat/multilingual-e5-small — MIT, 118M params, multilingual (Indic + French)
  memory    the 96M-param word-embedding table is FROZEN -> AdamW state for ~22M params;
            bf16 autocast on Ampere (RTX 3050 4-6 GB), fp32 fallback on CPU
  leakage   trained ONLY on train entities outside the GBDT sample (--exclude folds.tsv),
            so its score is a clean feature for that run's stage 2
  mlguard   training curve goes to RunLog (metrics.jsonl) when --run-dir is given

    python src/gpu/reranker.py train --exclude runs/<id>/folds.tsv --entities 40000 --out models/rr_e5s
    python src/gpu/reranker.py bench --model models/rr_e5s
"""
import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

BASE = "intfloat/multilingual-e5-small"
MAX_LEN = 96


def _torch():
    """(torch, device, autocast dtype).

    The dtype is the thing that differs between our boxes and it matters more
    than it looks. bf16 carries fp32's exponent range, so activations simply
    cannot overflow it; fp16 tops out at 65504. An RTX 3050 is sm_86 and gets
    bf16. A T4 (ml.g4dn.xlarge) is sm_75 -- Turing, no bf16 -- so it silently
    falls back to fp16, and a model whose activations are fine on one box
    produces inf on the other.

    AMLC_AMP overrides the choice: bf16 | fp16 | fp32 | auto (default).
    fp32 is the safe harbour -- slower, more memory, but no overflow.
    """
    import os
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    want = os.environ.get("AMLC_AMP", "auto").lower()
    if dev != "cuda" or want == "fp32":
        return torch, dev, None
    if want == "bf16":
        return torch, dev, torch.bfloat16
    if want == "fp16":
        return torch, dev, torch.float16
    if want not in ("auto", ""):
        raise ValueError(f"AMLC_AMP={want!r}; expected bf16|fp16|fp32|auto")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch, dev, amp


def serialize(rec: pd.DataFrame, ids) -> np.ndarray:
    """Record text for the cross-encoder from a features_v2.record_table. The
    transliteration is appended only when it differs (Indic names, domains)."""
    r = rec.loc[list(ids)]
    tl = np.where(r["_tl_core"].to_numpy() != r["_core_name"].to_numpy(), " | " + r["_tl_core"].to_numpy(), "")
    return ("name: " + r["_name"].to_numpy() + tl + " addr: " + r["_addr"].to_numpy()).astype(object)


# ------------------------------------------------------------------ data
def make_training_pairs(exclude_ids, n_entities, seed=7, neg_per_pos=3):
    """Blocked candidate pairs for train entities NOT in exclude_ids: all positives plus
    the hardest negatives (highest blocking similarity). Cached in INTERIM."""
    import blocking
    import data as D
    import features_v2 as F2
    import ingest
    from normalize import add_blocking_columns

    ex = set(exclude_ids)
    path = C.INTERIM / f"rr_pairs_n{n_entities}_s{seed}_ex{len(ex)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s1 = D.read_source(C.TRAIN_S1)
    s1 = s1[~s1[C.ID].isin(ex)].sample(n=n_entities, random_state=seed).reset_index(drop=True)
    s1 = add_blocking_columns(s1)
    s2, s3 = ingest.blocking_frame([C.TRAIN_S2]), ingest.blocking_frame([C.TRAIN_S3])
    pairs = blocking.candidates_to_frame(blocking.generate_candidates(s1, s2, s3))
    del s2, s3
    truth = D.read_ground_truth(C.TRAIN_GT, keep_ids=s1[C.ID])
    pairs["y"] = [int(c in truth[s]) for s, c in zip(pairs["s1_id"], pairs["cand_id"])]
    pos = pairs[pairs["y"] == 1]
    neg = pairs[pairs["y"] == 0].sort_values("block_sim", ascending=False).head(len(pos) * neg_per_pos)
    pairs = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    L = F2.record_table(F2.load_records([C.TRAIN_S1], set(pairs["s1_id"])))
    R = F2.record_table(F2.load_records([C.TRAIN_S2, C.TRAIN_S3], set(pairs["cand_id"])))
    out = pd.DataFrame({"s1_id": pairs["s1_id"], "a": serialize(L, pairs["s1_id"]),
                        "b": serialize(R, pairs["cand_id"]), "y": pairs["y"]})
    out.to_parquet(path, index=False)
    return out


# ------------------------------------------------------------------ model
def _load(model_dir_or_base, train=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch, dev, _ = _torch()
    tok = AutoTokenizer.from_pretrained(model_dir_or_base)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir_or_base, num_labels=1).to(dev)
    if train:
        model.base_model.embeddings.word_embeddings.requires_grad_(False)
    return tok, model


def _batches(a, b, bs, order=None):
    order = np.arange(len(a)) if order is None else order
    for i in range(0, len(order), bs):
        ix = order[i:i + bs]
        yield ix, [a[j] for j in ix], [b[j] for j in ix]


def valid_split(n, valid_frac, rng, groups=None):
    """(train_idx, valid_idx). By entity when `groups` is given, so no entity straddles both."""
    if groups is not None:
        g = np.asarray(groups)
        vg = set(rng.choice(np.unique(g), max(1, int(valid_frac * len(np.unique(g)))), replace=False))
        va = np.fromiter((x in vg for x in g), bool, len(g))
    else:
        va = rng.random(n) < valid_frac
    return np.where(~va)[0], np.where(va)[0]


def warmup_linear(torch, opt, steps):
    """5% linear warm-up, then linear decay to 0 at `steps`."""
    warm = max(1, steps // 20)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))


def train(a, b, y, out, epochs=1, bs=64, lr=5e-5, valid_frac=0.02, run_dir=None, seed=7, groups=None,
          augment=0.0, y_eval=None, extra_meta=None):
    """Fine-tune on (a, b, y). The validation split is by entity (`groups`) when given.
    `y` may be soft targets (distillation); metrics then use the hard labels `y_eval`.
    `augment` > 0 appends that fraction of augmented TRAIN rows (gpu/ft_data.py), never valid rows."""
    torch, dev, amp = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object), np.asarray(y, dtype=np.float32)
    y_ev = y if y_eval is None else np.asarray(y_eval, dtype=np.float32)
    tr_idx, va_idx = valid_split(len(y), valid_frac, rng, groups)
    if augment > 0:                               # own rng: the default path's RNG stream is unchanged
        from gpu.ft_data import augment_train
        a, b, y, _, tr_idx = augment_train(a, b, y, None, tr_idx, augment, np.random.default_rng(seed + 1))
    tok, model = _load(BASE, train=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr_idx) / bs)
    sched = warmup_linear(torch, opt, steps)
    # torch.amp.GradScaler(device) landed in 2.4; torch.cuda.amp.GradScaler is
    # the older spelling and is deprecated in newer versions. requirements-gpu
    # allows torch>=2.2, and SageMaker images pin their own, so support both.
    if amp == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    lossf = torch.nn.BCEWithLogitsLoss()
    log = None
    if run_dir:
        from runlog import RunLog
        log = RunLog(run_dir)
    print(f"reranker: {len(tr_idx):,} train / {len(va_idx):,} valid pairs, {steps} steps, device {dev}, "
          f"amp {amp}, trainable {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
    step, t0, run_loss, bad_batches = 0, time.time(), [], 0
    for ep in range(epochs):
        model.train()
        for ix, xa, xb in _batches(a, b, bs, rng.permutation(tr_idx)):
            enc = tok(xa, xb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            tgt = torch.from_numpy(y[ix]).to(dev)
            with torch.autocast(device_type=dev, dtype=amp, enabled=amp is not None):
                z = model(**enc).logits.squeeze(-1).float()
            # fp16 caps at 65504 and a T4 has no bf16, so an activation that is
            # fine on Ampere can come back inf here. .float() preserves the inf,
            # so clamp before the loss: sigmoid(30) is 1 - 1e-13, which makes
            # this free in every case that is not already broken.
            if _DIAG and not torch.isfinite(z).all():
                nb = (~torch.isfinite(z)).nonzero(as_tuple=True)[0].tolist()
                print(f"  NON-FINITE LOGITS step {step}: {len(nb)} of {len(z)}", flush=True)
                for bi in nb[:3]:
                    j = ix[bi]
                    print(f"    idx={j} len_a={len(str(a[j]))} len_b={len(str(b[j]))}", flush=True)
                    print(f"      a={str(a[j])[:200]!r}", flush=True)
                    print(f"      b={str(b[j])[:200]!r}", flush=True)
            z = torch.nan_to_num(z, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP)
            z = z.clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)
            loss = lossf(z, tgt)
            if not torch.isfinite(loss):
                # Never step on a non-finite loss: one such backward writes nan
                # into every trainable weight and the run cannot recover.
                bad_batches += 1
                if bad_batches <= 5 or bad_batches % 100 == 0:
                    print(f"  skipped non-finite loss at step {step} ({bad_batches} so far)", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            # clip_grad_norm_ returns ONE norm combined across every parameter.
            # If a single row produced a nan gradient, that norm is nan, the
            # clip scale is nan, and multiplying every other gradient by it
            # corrupts the whole model in one step -- which is how "2 of 32 bad
            # at step 1" becomes "32 of 32 bad forever" at step 2. Clipping
            # does not protect against this; it is what spreads it.
            #
            # Checking the LOSS is not enough: the logits are clamped above, so
            # the loss is finite while the backward pass can still be nan from
            # the poisoned saved activations. The norm is the first place the
            # corruption is visible, so the skip has to happen here.
            total_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(total_norm):
                bad_batches += 1
                if bad_batches <= 5 or bad_batches % 100 == 0:
                    print(f"  skipped step {step}: grad norm {total_norm} "
                          f"({bad_batches} so far)", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            run_loss.append(loss.item())
            step += 1
            if step % 200 == 0 or step == steps:
                vl, _ = _eval(tok, model, a, b, y_ev, va_idx[:4000], bs=min(bs, 64))
                tl = float(np.mean(run_loss[-200:]))
                print(f"  step {step}/{steps}  train {tl:.4f}  valid {vl:.4f}  {step*bs/(time.time()-t0):,.0f} pairs/s", flush=True)
                if log:
                    log.tick(fold=0, iter=step, train_loss=tl, valid_loss=vl)
                model.train()
    vl, auc = _eval(tok, model, a, b, y_ev, va_idx, bs=min(bs, 64))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = dict(kind="e5", base=BASE, max_len=MAX_LEN, pairs=int(len(tr_idx)), valid_logloss=vl, valid_auc=auc,
                seconds=time.time() - t0, device=dev, augment=augment, **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if groups is not None:  # run_v2 --rerank refuses a model that saw any of its entities
        (out / "entities.txt").write_text("\n".join(sorted(set(map(str, groups)))), encoding="utf-8")
    if log:
        log.end()
    if bad_batches:
        print(f"reranker: WARNING {bad_batches} of {steps} batches skipped for a non-finite "
              f"loss. On a GPU without bf16 (T4/Turing) try AMLC_AMP=fp32, or lower --bs.",
              flush=True)
    print(f"reranker saved -> {out}  valid logloss {vl:.4f}  AUC {auc:.4f}", flush=True)
    return meta


def _eval(tok, model, a, b, y, idx, bs=256):
    from sklearn.metrics import log_loss, roc_auc_score
    if len(idx) == 0:
        return float("nan"), float("nan")
    p = _score(tok, model, a[idx], b[idx], bs=bs)
    auc = roc_auc_score(y[idx], p) if 0 < y[idx].mean() < 1 else float("nan")
    return float(log_loss(y[idx], np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])), float(auc)


# Log-odds beyond this are meaningless (sigmoid(30) = 1 - 1e-13) and are the
# signature of fp16 overflow rather than confidence.
LOGIT_CLAMP = 30.0

# AMLC_RR_DIAG=1 prints the text of any row whose logits go non-finite.
_DIAG = os.environ.get("AMLC_RR_DIAG") == "1"


def _score(tok, model, a, b, bs=256):
    torch, dev, amp = _torch()
    model.eval()
    # Length bucketing cuts padding, but it also puts the LONGEST sequences in
    # the same batches -- the largest activations and the largest VRAM spike.
    # That is why a run can train happily for 200 random batches and then die
    # on the first validation pass.
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    with torch.inference_mode():
        for ix, xa, xb in _batches(list(a), list(b), bs, order):
            enc = tok(xa, xb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev, dtype=amp, enabled=amp is not None):
                z = model(**enc).logits.squeeze(-1).float()
            z = torch.nan_to_num(z, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP)
            out[ix] = torch.sigmoid(z).cpu().numpy()
    return out


def score(model_dir, a, b, bs=256):
    """Match probability for aligned text arrays a, b (use serialize())."""
    tok, model = _load(model_dir)
    return _score(tok, model, np.asarray(a, dtype=object), np.asarray(b, dtype=object), bs)


def bench(model_dir, n=5000):
    rng = np.random.default_rng(0)
    words = ["global", "business", "private", "limited", "road", "street", "chennai", "mumbai", "rue", "avenue"]
    mk = lambda: "name: " + " ".join(rng.choice(words, 4)) + " addr: " + " ".join(rng.choice(words, 8)) + " 12"
    a, b = [mk() for _ in range(n)], [mk() for _ in range(n)]
    tok, model = _load(model_dir)
    _score(tok, model, np.array(a[:256], dtype=object), np.array(b[:256], dtype=object))  # warm-up
    t = time.time()
    _score(tok, model, np.array(a, dtype=object), np.array(b, dtype=object))
    rate = n / (time.time() - t)
    print(f"inference: {rate:,.0f} pairs/s -> 5M band pairs in {5e6 / rate / 60:.0f} min")
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--exclude", help="folds.tsv of the GBDT run (its entities are excluded)")
    src.add_argument("--pairs", nargs="+", help="prebuilt pair parquet(s), e.g. ft_data.py band output")
    t.add_argument("--entities", type=int, default=40000)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--augment", type=float, default=0.0, help="fraction of augmented train rows to add")
    t.add_argument("--run-dir", default=None)
    bn = sub.add_parser("bench")
    bn.add_argument("--model", default=BASE)
    a = ap.parse_args()
    if a.cmd == "train":
        if a.pairs:
            from gpu.ft_data import load_pairs
            d = load_pairs(a.pairs)
        else:
            d = make_training_pairs(pd.read_csv(a.exclude, sep="\t", dtype=str)["s1_id"], a.entities)
        train(d["a"], d["b"], d["y"], a.out, epochs=a.epochs, bs=a.bs, run_dir=a.run_dir, groups=d["s1_id"],
              augment=a.augment)
    else:
        bench(a.model)
