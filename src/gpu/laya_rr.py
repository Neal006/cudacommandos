"""GPU job 3b: laya (mmBERT-base typed-decision model) as the band reranker.

Drop-in challenger for src/gpu/reranker.py (e5-small): same training pairs, same record
text (reranker.serialize), same entities.txt leak guard, same `score(model_dir, a, b)`.
run_v2 --rerank picks the backend from the model dir's meta.json `kind`.

  why       laya-multilingual's backbone is mmBERT-base (322M; 1800+ languages incl. the
            9 Indic scripts in India S2), and the gap to the blocking ceiling is mostly India.
            Zero-shot it is useless here (measured: an unrelated pair scored 0.90 "same"),
            so it is ALWAYS fine-tuned on our pairs.
  speed     laya's predict() builds one sequence per call in Python. Every pair here asks the
            SAME yes/no question, so the question prefix is tokenized once (marker positions are
            constant), records are batch-tokenized, and DecisionModel.forward runs directly on
            length-bucketed batches under autocast. Only the two marker logits are read.
  memory    197M of the 322M params are the word-embedding table: frozen. Only the top
            --train-layers encoder layers + laya's decision head train (default 8 -> fits the
            4 GB RTX 3050; --train-layers -1 trains all but the table, for a 16 GB T4).
  output    a laya checkpoint (rl_agent_config.json with the fitted noul temperature,
            model.safetensors, tokenizer/, encoder/) so `laya.load(dir)` works on it as-is,
            plus meta.json (kind=laya) and entities.txt.

    python src/gpu/laya_rr.py fetch                  # -> models/laya_ml_base (no symlinks: Windows-safe)
    python src/gpu/laya_rr.py train --exclude runs/<id>/folds.tsv --entities 40000 --out models/rr_laya
    python src/gpu/laya_rr.py bench --model models/rr_laya [--train]
    python src/run_v2.py --sample 30000 --train-only --rerank models/rr_laya
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from gpu import ft_data as FD  # noqa: E402
from gpu import ft_train as FT  # noqa: E402
from gpu import reranker as RR  # noqa: E402

serialize = RR.serialize          # identical record text for both rerankers

BASE_REPO = "convaiinnovations/laya-multilingual"
BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"   # pinned: a re-fetch must give the same base weights
BASE_DIR = C.ROOT / "models" / "laya_ml_base"
CKPT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
QUESTION = {"t": "noul", "ins": "Do record A and record B describe the same real-world business?",
            "crit": {"true": "the same business", "false": "different businesses"}}
NOUL = 2                          # laya.common.QTYPES["noul"]
REC_MAX = 64                      # tokens per record, "record A: " label included
T_MIN, T_MAX = 0.5, 5.0           # = laya.common.TEMP_MIN/MAX: laya.load clamps to this, so fit inside it
TRAIN_LAYERS = 8                  # top encoder layers fine-tuned by default (4 GB GPU)
FULL_LR, LORA_LR = 3e-5, 2e-4     # default lr: partial unfreeze vs LoRA adapters
BAND_PAIRS_TEST = 620_000         # ~1.2% of the 52M test candidates (EXPERIMENTS: 0.2-0.8 band)


# ------------------------------------------------------------------ input format
class PairEncoder:
    """laya's sequence for one fixed yes/no question over many record pairs:
    [CLS] noul question [SEP] <mask> false-option <mask> true-option [SEP] record A record B [SEP]."""

    def __init__(self, tok, head_max_len=256):
        from laya.common import build_sequence
        self.tok = tok
        ids, markers = build_sequence(tok, "", QUESTION, max_len=head_max_len + 16, head_max_len=head_max_len)
        self.prefix = ids[:-1]    # drop the empty state's closing [SEP]; the options' [SEP] stays
        self.markers = markers
        self.max_len = len(self.prefix) + 2 * REC_MAX + 1

    def _records(self, label, texts):
        m = self.tok.mask_token   # a literal mask token in data would become a third "option"
        return self.tok([label + str(x).replace(m, " ") for x in texts], add_special_tokens=False,
                        truncation=True, max_length=REC_MAX)["input_ids"]

    def encode(self, a, b):
        """Token ids per pair. Each record is capped on its own, so a long A never hides B."""
        sep = [self.tok.sep_token_id]
        return [self.prefix + x + y + sep
                for x, y in zip(self._records("record A: ", a), self._records("\nrecord B: ", b))]


def collate(rows, markers, pad_id):
    """Right-padded batch in DecisionModel.forward's argument layout."""
    import torch
    n, width = len(rows), max(map(len, rows))
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    att = torch.zeros((n, width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        att[i, :len(r)] = 1
    return {"input_ids": ids, "attention_mask": att,
            "marker_pos": torch.tensor([markers] * n, dtype=torch.long),
            "marker_mask": torch.ones((n, len(markers)), dtype=torch.bool),
            "qtype": torch.full((n,), NOUL, dtype=torch.long)}


def fit_temperature(z, y):
    """Scalar T minimising log loss of sigmoid(z / T): laya's noul calibration, on our valid split."""
    from scipy.optimize import minimize_scalar
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def nll(log_t):
        s = z / math.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, s) - y * s))
    r = minimize_scalar(nll, bounds=(math.log(T_MIN), math.log(T_MAX)), method="bounded")
    return float(math.exp(r.x))


# ------------------------------------------------------------------ model
def fetch(repo=BASE_REPO, out=BASE_DIR, revision=BASE_REVISION):
    """Download a laya checkpoint into a plain folder. The HF cache needs symlinks, which
    Windows refuses without Developer Mode (WinError 1314); local_dir does not."""
    from huggingface_hub import snapshot_download
    out = Path(out)
    snapshot_download(repo, revision=revision, local_dir=str(out), allow_patterns=CKPT_FILES)
    print(f"laya checkpoint {repo}@{(revision or 'main')[:8]} -> {out}", flush=True)
    return out


def _load(model_dir, train_layers=None):
    """(tokenizer, DecisionModel, laya cfg, noul temperature) on the best device."""
    import laya
    torch, dev, _ = RR._torch()
    model_dir = Path(model_dir)
    if not (model_dir / "rl_agent_config.json").exists():
        if model_dir.resolve() != BASE_DIR.resolve():
            raise FileNotFoundError(f"{model_dir} is not a laya checkpoint (no rl_agent_config.json)")
        fetch(out=model_dir)
    ag = laya.load(str(model_dir), device=dev)
    if ag.device.type != dev:     # laya silently falls back to CPU on OOM: 10-50x slower, say so
        raise RuntimeError(f"laya loaded on {ag.device.type}, not {dev} (GPU out of memory?); "
                           f"free VRAM or lower --train-layers")
    model = ag.model
    if train_layers is not None:
        _freeze(model, train_layers)
    return ag.tok, model, ag.cfg, float(ag.temperature[NOUL])


def _freeze(model, train_layers):
    """Freeze the word-embedding table always; with train_layers >= 0 also all but the top
    `train_layers` encoder layers (and the embedding block under them). The head always trains."""
    model.requires_grad_(True)
    enc = model.encoder
    enc.get_input_embeddings().requires_grad_(False)
    if train_layers < 0:
        return
    layers = enc.layers
    enc.embeddings.requires_grad_(False)
    for layer in layers[:max(0, len(layers) - train_layers)]:
        layer.requires_grad_(False)


def _logit_diff(model, batch, dev, amp):
    """logit(true) - logit(false) per row: the pre-temperature noul log-odds."""
    import torch
    with torch.autocast(device_type=dev, dtype=amp, enabled=amp is not None):
        logits, _ = model(**{k: v.to(dev, non_blocking=True) for k, v in batch.items()})
    return (logits[:, 1] - logits[:, 0]).float()


def _diffs(tok, model, enc, a, b, bs=128):
    """Log-odds for aligned record arrays, length-bucketed to cut padding."""
    torch, dev, amp = RR._torch()
    model.eval()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)], kind="stable")
    out = np.empty(len(a), dtype=np.float32)
    with torch.inference_mode():
        for ix, xa, xb in RR._batches(list(a), list(b), bs, order):
            out[ix] = _logit_diff(model, collate(enc.encode(xa, xb), enc.markers, tok.pad_token_id),
                                  dev, amp).cpu().numpy()
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def score(model_dir, a, b, bs=128):
    """Calibrated match probability for aligned text arrays a, b (use serialize())."""
    tok, model, cfg, t = _load(model_dir)
    enc = PairEncoder(tok, cfg.get("head_max_len", 256))
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    return _sigmoid(_diffs(tok, model, enc, a, b, bs) / t).astype(np.float32)


# ------------------------------------------------------------------ training
def _metrics(z, y, t):
    from sklearn.metrics import log_loss, roc_auc_score
    if len(y) == 0:
        return float("nan"), float("nan")
    p = np.clip(_sigmoid(z / t), 1e-7, 1 - 1e-7)
    auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
    return float(log_loss(y, p, labels=[0, 1])), float(auc)


def _save(model, base, out, cfg, t):
    """Write a laya checkpoint: base tokenizer/encoder config, our weights (bf16, as laya ships), cfg with T."""
    import torch
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        if (base / sub).is_dir():
            shutil.copytree(base / sub, out / sub, dirs_exist_ok=True)
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[NOUL] = t
    (out / "rl_agent_config.json").write_text(json.dumps({**cfg, "temperature": temps}, indent=2))
    sd = {k: (v.detach().to("cpu", torch.bfloat16) if v.is_floating_point() else v.detach().cpu()).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))


def _epoch_batches(tr_idx, bs, rng, keys=None):
    """Row-index batches for one epoch: shuffled, or group-contiguous when the listwise loss needs keys."""
    order = rng.permutation(tr_idx)
    if keys is None:
        return [order[i:i + bs] for i in range(0, len(order), bs)]
    return list(FT.group_batches(keys, order, bs))


def train(a, b, y, out, base=BASE_DIR, epochs=1, bs=32, lr=None, valid_frac=0.02, run_dir=None, seed=7,
          groups=None, train_layers=TRAIN_LAYERS, augment=0.0, lora=0, listwise=0.0, group_keys=None,
          extra_meta=None):
    """Fine-tune laya's noul decision on (a, b, y); valid split by entity (`groups`) when given.
    Options (docs/FINETUNE.md): augment = fraction of augmented train rows (ft_data), lora = LoRA rank over
    all layers instead of the top-`train_layers` unfreeze (ft_train), listwise = weight of the listwise
    loss added to BCE, over `group_keys` (cand_id: the S1 candidates of one S2/S3 record compete)."""
    torch, dev, amp = RR._torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object), np.asarray(y, dtype=np.float32)
    if listwise > 0 and group_keys is None:
        raise ValueError("the listwise loss needs group_keys (a cand_id column in the pairs)")
    keys = None if group_keys is None else np.asarray(group_keys, dtype=object)
    tr_idx, va_idx = RR.valid_split(len(y), valid_frac, rng, groups)
    if augment > 0:                               # own rng: the default path's RNG stream is unchanged
        n0 = len(a)
        a, b, y, _, tr_idx = FD.augment_train(a, b, y, None, tr_idx, augment, np.random.default_rng(seed + 1))
        if keys is not None:                      # augmented copies are singleton groups (plain BCE)
            keys = FT.singleton_keys(keys, len(a) - n0)
    base = Path(base)
    tok, model, cfg, _ = _load(base, train_layers=None if lora else train_layers)
    if lora:                                      # checkpointing: 1.9 GiB instead of 3.8 at bs 32, ~10% slower
        FT.apply_lora(model, lora, checkpointing=True)
    lr = lr or (LORA_LR if lora else FULL_LR)
    enc = PairEncoder(tok, cfg.get("head_max_len", 256))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    plan = [_epoch_batches(tr_idx, bs, rng, keys if listwise > 0 else None) for _ in range(epochs)]
    steps = sum(map(len, plan))
    sched = RR.warmup_linear(torch, opt, steps)
    oversized = sum(len(ix) > bs for batches in plan for ix in batches)
    if oversized:                                 # a listwise group bigger than bs is one batch: VRAM spike
        print(f"laya reranker: WARNING {oversized} batch(es) exceed bs {bs} (one group each); "
              f"raise --bs if VRAM allows", flush=True)
    if lora:
        tuned = f"LoRA r={lora} on all layers"
    else:
        tuned = f"top {train_layers} layers" if train_layers >= 0 else "all layers"
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None
    lossf = torch.nn.BCEWithLogitsLoss()
    log = None
    if run_dir:
        from runlog import RunLog
        log = RunLog(run_dir)
    print(f"laya reranker: {len(tr_idx):,} train / {len(va_idx):,} valid pairs, {steps} steps, device {dev}, "
          f"amp {amp}, trainable {sum(p.numel() for p in params)/1e6:.1f}M params "
          f"({tuned} + head), lr {lr:g}, augment {augment:g}, listwise {listwise:g}", flush=True)
    step, t0, run_loss = 0, time.time(), []
    for batches in plan:
        model.train()
        for ix in batches:
            batch = collate(enc.encode(a[ix], b[ix]), enc.markers, tok.pad_token_id)
            z_b, tgt = _logit_diff(model, batch, dev, amp), torch.from_numpy(y[ix]).to(dev)
            loss = lossf(z_b, tgt)
            if listwise > 0:
                loss = loss + listwise * FT.listwise_loss(z_b, keys[ix], tgt)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sched.step()
            run_loss.append(loss.item())
            step += 1
            if step % 200 == 0 or step == steps:
                sub = va_idx[:4000]
                vl, _ = _metrics(_diffs(tok, model, enc, a[sub], b[sub]), y[sub], 1.0)
                tl = float(np.mean(run_loss[-200:]))
                print(f"  step {step}/{steps}  train {tl:.4f}  valid {vl:.4f}  "
                      f"{step*bs/(time.time()-t0):,.0f} pairs/s", flush=True)
                if log:
                    log.tick(fold=0, iter=step, train_loss=tl, valid_loss=vl)
                model.train()
    if lora:
        FT.merge_lora(model)                      # validate and save exactly what will be served
    z = _diffs(tok, model, enc, a[va_idx], b[va_idx])
    t = fit_temperature(z, y[va_idx]) if len(va_idx) else 1.0
    vl, auc = _metrics(z, y[va_idx], t)
    out = Path(out)
    _save(model, base, out, cfg, t)
    import laya
    meta = dict(kind="laya", base=str(base), base_repo=BASE_REPO, base_revision=BASE_REVISION,
                laya_version=laya.__version__,
                question=QUESTION["ins"], rec_max=REC_MAX, train_layers=train_layers, pairs=int(len(tr_idx)),
                temperature=t, valid_logloss=vl, valid_auc=auc, seconds=time.time() - t0, device=dev,
                lr=lr, augment=augment, lora=lora, listwise=listwise, **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if groups is not None:  # run_v2 --rerank refuses a model that saw any of its entities
        (out / "entities.txt").write_text("\n".join(sorted(set(map(str, groups)))), encoding="utf-8")
    if log:
        log.end()
    print(f"laya reranker saved -> {out}  T {t:.3f}  valid logloss {vl:.4f}  AUC {auc:.4f}", flush=True)
    return meta


# ------------------------------------------------------------------ bench
def _synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    words = ["global", "business", "private", "limited", "road", "street", "chennai", "mumbai", "rue", "avenue"]
    mk = lambda: "name: " + " ".join(rng.choice(words, 4)) + " addr: " + " ".join(rng.choice(words, 8)) + " 12"
    return np.array([mk() for _ in range(n)], dtype=object), np.array([mk() for _ in range(n)], dtype=object)


def bench(model_dir=BASE_DIR, n=5000, train_steps=0, bs=32, train_layers=TRAIN_LAYERS, lora=0, grad_ckpt=False):
    """Inference pairs/s (and optionally train pairs/s) plus peak VRAM on synthetic records."""
    torch, dev, amp = RR._torch()
    a, b = _synthetic(n)
    tok, model, cfg, _ = _load(model_dir, train_layers=train_layers if train_steps and not lora else None)
    if train_steps and lora:
        FT.apply_lora(model, lora, checkpointing=grad_ckpt)
    enc = PairEncoder(tok, cfg.get("head_max_len", 256))
    print(f"prefix {len(enc.prefix)} tokens, max row {enc.max_len} tokens", flush=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _diffs(tok, model, enc, a[:256], b[:256])                      # warm-up
    t = time.time()
    _diffs(tok, model, enc, a, b)
    rate = n / (time.time() - t)
    print(f"inference: {rate:,.0f} pairs/s -> {BAND_PAIRS_TEST:,} test band pairs in "
          f"{BAND_PAIRS_TEST / rate / 60:.0f} min", flush=True)
    if train_steps:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-5)
        y = torch.zeros(bs, device=dev)
        model.train()
        t = time.time()
        for i in range(train_steps):
            j = (i * bs) % (n - bs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                _logit_diff(model, collate(enc.encode(a[j:j + bs], b[j:j + bs]), enc.markers, tok.pad_token_id),
                            dev, amp), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"train: {train_steps * bs / (time.time() - t):,.0f} pairs/s at bs {bs}, "
              f"{sum(p.numel() for p in params)/1e6:.1f}M trainable", flush=True)
    if dev == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--repo", default=BASE_REPO)
    f.add_argument("--out", default=str(BASE_DIR))
    f.add_argument("--revision", default=BASE_REVISION, help="commit SHA; 'main' for the latest")
    t = sub.add_parser("train")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--exclude", help="folds.tsv of the GBDT run (its entities are excluded)")
    src.add_argument("--pairs", nargs="+", help="prebuilt pair parquet(s), e.g. ft_data.py band output")
    t.add_argument("--entities", type=int, default=40000)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default=str(BASE_DIR), help="laya checkpoint dir to start from (fetched if missing)")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=None, help=f"default {FULL_LR:g}, or {LORA_LR:g} with --lora")
    t.add_argument("--train-layers", type=int, default=TRAIN_LAYERS, help="-1 = all but the embedding table")
    t.add_argument("--augment", type=float, default=0.0, help="option 2: fraction of augmented train rows to add")
    t.add_argument("--lora", type=int, default=0, help="option 3: LoRA rank over all layers (e.g. 16); 0 = off")
    t.add_argument("--listwise", type=float, default=0.0, help="option 4: weight of the listwise loss; 0 = off")
    t.add_argument("--group-by", default="cand_id", help="listwise groups (cand_id: records compete for one S1)")
    t.add_argument("--run-dir", default=None)
    bn = sub.add_parser("bench")
    bn.add_argument("--model", default=str(BASE_DIR))
    bn.add_argument("--train", action="store_true", help="also time 30 training steps")
    bn.add_argument("--bs", type=int, default=32)
    bn.add_argument("--train-layers", type=int, default=TRAIN_LAYERS)
    bn.add_argument("--lora", type=int, default=0, help="time LoRA training at this rank instead")
    bn.add_argument("--grad-ckpt", action="store_true", help="with --lora: gradient checkpointing")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.repo, a.out, a.revision)
    elif a.cmd == "train":
        if a.pairs:
            d = FD.load_pairs(a.pairs)
        else:
            d = RR.make_training_pairs(pd.read_csv(a.exclude, sep="\t", dtype=str)["s1_id"], a.entities)
        if a.listwise > 0 and (a.group_by not in d.columns or d[a.group_by].isna().any()):
            raise SystemExit(f"--listwise groups by {a.group_by!r}, which some or all of these pairs lack; "
                             f"build them with `python src/gpu/ft_data.py band` (make_training_pairs has no cand_id)")
        train(d["a"], d["b"], d["y"], a.out, base=a.base, epochs=a.epochs, bs=a.bs, lr=a.lr,
              run_dir=a.run_dir, groups=d["s1_id"], train_layers=a.train_layers, augment=a.augment,
              lora=a.lora, listwise=a.listwise,
              group_keys=d[a.group_by].to_numpy() if a.group_by in d.columns else None,
              extra_meta=dict(pair_files=a.pairs or [], group_by=a.group_by))
    else:
        bench(a.model, train_steps=30 if a.train else 0, bs=a.bs, train_layers=a.train_layers, lora=a.lora,
              grad_ckpt=a.grad_ckpt)
