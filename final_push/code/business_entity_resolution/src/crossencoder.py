"""Cross-encoder (plan §E.2).

CPU box (needs the work/ parquet files):
  python crossencoder.py export            -> work/ce/{train_pairs,band_train,band_test}.parquet
  python crossencoder.py pseudo            -> work/ce/pseudo_fr.parquet   (France pseudo-labels from s2_test)
GPU side (A10G; copy work/ce/ over, results back):
  python crossencoder.py train --model intfloat/multilingual-e5-small --out ce_e5s
  python crossencoder.py train --model ce_e5s --extra pseudo_fr.parquet --epochs 1 --out ce_e5s_fr
  python crossencoder.py infer --model ce_e5s --band band_test.parquet --out ce_test.parquet
Then copy ce_train.parquet / ce_test.parquet into work/ and rerun context.py.

Training data is the hidden set H only, so every score on the query set Q is out-of-sample.
pair_id = (s1 << 32) | cand.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
CE_DIR = Path(os.environ.get("ER_CE_DIR") or
              (_here.parents[3] / "work" / "ce" if len(_here.parents) > 3 else _here.parent / "ce"))


# ----------------------------------------------------------------------------------------------
# CPU box: exports
# ----------------------------------------------------------------------------------------------
def _texts(split):
    import polars as pl
    from normalize import load_norm
    n = load_norm(split, ["idx", "n_ce", "a_ce"])
    return n.select("idx", (pl.col("n_ce") + " | " + pl.col("a_ce")).str.slice(0, 300).alias("t"))


def _attach(pairs, texts):
    import polars as pl
    return (pairs.join(texts.rename({"idx": "s1", "t": "text_a"}), on="s1")
                 .join(texts.rename({"idx": "cand", "t": "text_b"}), on="cand")
                 .with_columns((pl.col("s1").cast(pl.Int64) * (1 << 32) + pl.col("cand").cast(pl.Int64)).alias("pair_id")))


def export(seed=2026, n_val=20000):
    import polars as pl
    from config import CE_MAX_P1, CE_MAX_PER_S1, CE_MIN_P1, wpath
    from prune import label
    from splits import load_splits
    CE_DIR.mkdir(parents=True, exist_ok=True)
    sp = load_splits()
    h = sp.filter(pl.col("hidden")).select(pl.col("idx").alias("s1"))
    p = label(pl.read_parquet(wpath("pruned_train.parquet"), columns=["s1", "cand", "p_prune"]).join(h, on="s1"))
    p = p.filter(pl.col("p_prune") >= 0.01)
    rng = np.random.default_rng(seed)
    p = p.with_columns(pl.Series("u", rng.random(p.height)))
    neg = p.filter(pl.col("y") == 0).with_columns(
        pl.col("p_prune").rank("ordinal", descending=True).over("s1").alias("hr"),
        pl.col("u").rank("ordinal").over("s1").alias("ur"))
    neg = neg.filter((pl.col("hr") <= 4) | (pl.col("ur") == 1))
    tr = pl.concat([p.filter(pl.col("y") == 1).select("s1", "cand", "y"), neg.select("s1", "cand", "y")])
    val_s1 = h.sample(n_val, seed=seed).with_columns(pl.lit(True).alias("val"))
    tr = tr.join(val_s1, on="s1", how="left").with_columns(pl.col("val").fill_null(False))
    tt = _texts("train")
    out = _attach(tr, tt).select("pair_id", "text_a", "text_b", pl.col("y").cast(pl.Float32).alias("label"), "val")
    out.sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(CE_DIR / "train_pairs.parquet")
    print(f"train pairs: {out.height:,} ({out['label'].mean():.3f} positive), val {out['val'].sum():,}")
    for split in ("train", "test"):
        s = pl.read_parquet(wpath(f"s1_{split}.parquet"), columns=["s1", "cand", "p1"])
        band = (s.filter((pl.col("p1") >= CE_MIN_P1) & (pl.col("p1") <= CE_MAX_P1))
                 .filter(pl.col("p1").rank("ordinal", descending=True).over("s1") <= CE_MAX_PER_S1))
        b = _attach(band.select("s1", "cand"), tt if split == "train" else _texts("test"))
        b.select("pair_id", "text_a", "text_b").write_parquet(CE_DIR / f"band_{split}.parquet")
        print(f"band {split}: {b.height:,} pairs ({b.height / s['s1'].n_unique():.2f} per S1)")


def pseudo(p_pos=0.97, p_neg=0.03, margin=0.3):
    """France pseudo-labels from the current test predictions (plan §H.1)."""
    import polars as pl
    from config import wpath
    from normalize import load_norm
    n = load_norm("test", ["idx", "country", "a_hnd", "a_street"])
    s2 = pl.read_parquet(wpath(os.environ.get("ER_PSEUDO_SRC", "s2_test.parquet")))
    fr = n.filter(pl.col("country") == "France").select(pl.col("idx").alias("s1"))
    d = s2.join(fr, on="s1")
    d = d.with_columns(pl.col("p2").top_k(2).over("cand", mapping_strategy="join").alias("_t"),
                       pl.col("p2").rank("ordinal", descending=True).over("cand").alias("_r"))
    d = d.with_columns(pl.when(pl.col("_r") == 1).then(pl.col("p2") - pl.col("_t").list.get(1, null_on_oob=True)
                                                      .fill_null(0.0)).otherwise(-1.0).alias("margin"))
    hn = n.select("idx", "a_hnd")
    d = (d.join(hn.rename({"idx": "s1", "a_hnd": "h1"}), on="s1").join(hn.rename({"idx": "cand", "a_hnd": "h2"}), on="cand"))
    pos = d.filter((pl.col("p2") > p_pos) & (pl.col("margin") > margin) &
                   ((pl.col("h1") == pl.col("h2")) & (pl.col("h1") != "")))
    neg = d.filter(pl.col("p2") < p_neg)
    # rejected twins of accepted positives: same S1, lower score, strong name match is implied by the band
    twin = d.join(pos.select("s1"), on="s1", how="semi").filter((pl.col("p2") < 0.3) & (pl.col("p2") >= p_neg))
    # cap sizes (the link to the GPU box is slow); prefer hard negatives that stage-1 found plausible
    cap = lambda x, n: x.sample(min(n, x.height), seed=2026)  # noqa: E731
    pos = cap(pos, 120_000)
    neg = pl.concat([cap(neg.filter(pl.col("p1") >= 0.01), 100_000), cap(neg.filter(pl.col("p1") < 0.01), 20_000)])
    twin = cap(twin, 50_000)
    lab = pl.concat([pos.select("s1", "cand", pl.lit(1.0).alias("label")),
                     neg.select("s1", "cand", pl.lit(0.0).alias("label")),
                     twin.select("s1", "cand", pl.lit(0.0).alias("label"))])
    out = _attach(lab, _texts("test")).select("pair_id", "text_a", "text_b", pl.col("label").cast(pl.Float32),
                                              pl.lit(False).alias("val"))
    out.write_parquet(CE_DIR / "pseudo_fr.parquet")
    print(f"pseudo FR: {pos.height:,} pos, {neg.height:,} neg, {twin.height:,} twin-neg")


# ----------------------------------------------------------------------------------------------
# GPU side
# ----------------------------------------------------------------------------------------------
def _device():
    import torch
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        if torch.cuda.get_device_capability(0)[0] >= 7:
            return torch.device("cuda"), torch.float16
        return torch.device("cuda"), None          # Pascal (e.g. GTX 1080 Ti): fp32
    if torch.backends.mps.is_available():
        return torch.device("mps"), None
    return torch.device("cpu"), None


def pack(names=("train_pairs", "band_train", "band_test")):
    """CPU box: shrink the export for slow links (unique texts once + integer pairs, zstd-22)."""
    import polars as pl
    for name in names:
        d = pl.read_parquet(CE_DIR / f"{name}.parquet")
        d = d.with_columns((pl.col("pair_id") // (1 << 32)).alias("a"), (pl.col("pair_id") % (1 << 32)).alias("b"))
        texts = pl.concat([d.select(pl.col("a").alias("id"), pl.col("text_a").alias("t")),
                           d.select(pl.col("b").alias("id"), pl.col("text_b").alias("t"))]).unique("id")
        texts.write_parquet(CE_DIR / f"{name}_texts.parquet", compression="zstd", compression_level=22)
        d.drop("text_a", "text_b", "a", "b").write_parquet(CE_DIR / f"{name}_pairs.parquet", compression="zstd",
                                                           compression_level=22)
        print(f"packed {name}: {d.height:,} pairs, {texts.height:,} texts")


def _load(name, columns):
    """Read an export either as <name>.parquet or as the compact <name>_texts / <name>_pairs pair."""
    import pyarrow.parquet as pq
    full = CE_DIR / name if name.endswith(".parquet") else CE_DIR / f"{name}.parquet"
    stem = full.name[:-len(".parquet")]
    if full.exists():
        return pq.read_table(full, columns=columns).to_pydict()
    import polars as pl
    t = pl.read_parquet(CE_DIR / f"{stem}_texts.parquet")
    p = pl.read_parquet(CE_DIR / f"{stem}_pairs.parquet").with_row_index("_o").with_columns(
        (pl.col("pair_id") // (1 << 32)).alias("a"), (pl.col("pair_id") % (1 << 32)).alias("b"))
    p = (p.join(t.rename({"id": "a", "t": "text_a"}), on="a", how="left")
          .join(t.rename({"id": "b", "t": "text_b"}), on="b", how="left").sort("_o"))
    return p.select(columns).to_dict(as_series=False)


def train(args):
    import torch
    from sklearn.metrics import average_precision_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(2026)
    dev, amp = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    t = _load(args.pairs, ["text_a", "text_b", "label", "val"])
    A, B, Y, V = t["text_a"], t["text_b"], np.asarray(t["label"], np.float32), np.asarray(t["val"])
    if args.extra:
        e = _load(args.extra, ["text_a", "text_b", "label"])
        n_base = int(len(e["label"]) / max(args.extra_ratio, 1e-6))
        keep = np.flatnonzero(~V)
        keep = np.random.default_rng(0).choice(keep, min(n_base, keep.size), replace=False)
        A = [A[i] for i in keep] + e["text_a"]
        B = [B[i] for i in keep] + e["text_b"]
        Y = np.r_[Y[keep], np.asarray(e["label"], np.float32)]
        V = np.zeros(len(Y), bool)
    if args.limit:
        A, B, Y, V = A[:args.limit], B[:args.limit], Y[:args.limit], V[:args.limit]
    tr_idx, va_idx = np.flatnonzero(~V), np.flatnonzero(V)[:50000]
    print(f"train {tr_idx.size:,} pairs, val {va_idx.size:,}, device {dev}", flush=True)

    def batches(idx, bs, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc = tok([A[k] for k in j], [B[k] for k in j], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt")
            yield {k: v.to(dev) for k, v in enc.items()}, torch.from_numpy(Y[j]).to(dev)

    def evaluate():
        if va_idx.size == 0:
            return float("nan")
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            for x, _ in batches(va_idx, 512, False):
                out.append(model(**x).logits.float().squeeze(-1).cpu().numpy())
        model.train()
        return average_precision_score(Y[va_idx], np.concatenate(out))

    steps = args.epochs * int(np.ceil(tr_idx.size / args.bs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for x, y in batches(tr_idx, args.bs, True):
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                logit = model(**x).logits.float().squeeze(-1)
            loss = lossf(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} "
                      f"{step * args.bs / (time.time() - t0):.0f} pairs/s", flush=True)
            if step % max(1, steps // 5) == 0:
                print(f"  val PR-AUC {evaluate():.5f}", flush=True)
    print(f"final val PR-AUC {evaluate():.5f}")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


def infer(args):
    """Score a band (or one shard of it). Output parquet: pair_id (int64), ce (float32)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev, amp = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    if amp is not None:
        model = model.to(amp)
    d = _load(args.band, ["pair_id", "text_a", "text_b"])
    n_all = len(d["pair_id"])
    lo, hi = n_all * args.shard // args.nshards, n_all * (args.shard + 1) // args.nshards
    if args.limit:
        hi = min(hi, lo + args.limit)
    A, B, ids = d["text_a"][lo:hi], d["text_b"][lo:hi], d["pair_id"][lo:hi]
    n = len(ids)
    order = np.argsort([len(a) + len(b) for a, b in zip(A, B)], kind="stable")
    scores = np.empty(n, np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, args.bs):
            j = order[i:i + args.bs]
            enc = tok([A[k] for k in j], [B[k] for k in j], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            scores[j] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
            if (i // args.bs) % 200 == 0:
                print(f"  {i + len(j):,}/{n:,} pairs, {(i + len(j)) / (time.time() - t0):.0f} pairs/s", flush=True)
    pq.write_table(pa.table({"pair_id": pa.array(ids, pa.int64()), "ce": pa.array(scores)}), args.out,
                   compression="zstd")
    print(f"wrote {args.out}: {n:,} pairs in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    sub.add_parser("pseudo")
    t = sub.add_parser("train")
    t.add_argument("--model", default="intfloat/multilingual-e5-small")
    t.add_argument("--pairs", default="train_pairs.parquet")
    t.add_argument("--extra", default=None)
    t.add_argument("--extra-ratio", type=float, default=0.25)
    t.add_argument("--out", default="ce_model")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--bs", type=int, default=128)
    t.add_argument("--max-len", type=int, default=96)
    t.add_argument("--limit", type=int, default=0)
    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--band", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--bs", type=int, default=512)
    i.add_argument("--max-len", type=int, default=96)
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--shard", type=int, default=0)
    i.add_argument("--nshards", type=int, default=1)
    sub.add_parser("pack")
    a = ap.parse_args()
    {"export": lambda: export(), "pseudo": lambda: pseudo(), "pack": lambda: pack(), "train": lambda: train(a),
     "infer": lambda: infer(a)}[a.cmd]()


if __name__ == "__main__":
    main()
