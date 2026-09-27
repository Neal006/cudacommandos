# Krisha — GPU stage runbook (SageMaker, e5 reranker)

**Your CPU stage is done and you do not need to redo it.** Everything changed
here is GPU-side.

---

## What changed and why

Two things:

**1. We're fine-tuning e5 now, not laya.** Priyanshu's call, based on a
head-to-head we ran overnight — same band, same 514,335 training pairs, same
valid split, same 30k GBDT sample, only the model swapped:

| | baseline | laya | **e5** |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| train time | — | 132 min | **23 min** |

laya lost on India — the metric its whole hypothesis was built to win — and
costs 5.8× the training time. The premise was that e5 can't read native-script
names, but the incumbent is `intfloat/multilingual-e5-small`, already
multilingual, so the comparison was multilingual against multilingual.

Good news: the reranker itself is **proven**. Submission 002 (a *weaker* 30k
GBDT plus the e5 reranker) scored **0.951** against submission 001's **0.943**
without it. So this is worth your GPU hours — just pointed at the model that
won.

**2. Your NaN crash is fixed.** It was not your learning rate and not AdamW.

---

## Why it was crashing

**The T4 has no bfloat16.** That's the whole thing.

```python
amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

| Box | GPU | Arch | bf16? | gets |
|---|---|---|---|---|
| Priyanshu's laptop | RTX 3050 | sm_86 Ampere | yes | **bf16** |
| your `ml.g4dn.xlarge` | T4 | sm_75 **Turing** | **no** | **fp16** |

Both are 16-bit, but they split the bits differently. bf16 keeps fp32's
exponent range and maxes out around 3.4e38. **fp16 maxes out at 65504.**
Every run we did was bf16, so this could never happen to us — the code wasn't
wrong, it was only ever run on hardware that hides the problem.

**And here's why it died at the first validation, not during training.** The
scoring path sorts by length before batching, to cut padding:

```python
order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
```

That puts all the **longest sequences into the same batches**. Attention sums
scale with sequence length, so those batches produce the largest activations
the model ever sees. Training batches are random and never concentrate them.

So fp16 survives 200 random training batches and overflows on the first
length-bucketed validation batch. One `inf` row makes the loss `nan`, which
makes the gradients `nan`, which makes **every trainable weight** `nan` from
that step onward. One bad row kills the whole run.

That's also why your two experiments came back negative, and they were good
experiments:

- **3× lower lr didn't move it** — correct, because it was never divergence.
  The weights were fine; a forward pass overflowed a number format.
- **`--train-layers -1` made it worse** — correct, because unfreezing all 322M
  parameters lets activations drift further from their pretrained scale,
  closer to 65504.

Your instinct in the last message was right, incidentally: *"bf16 itself is
the problem, not the optimizer... some pairs produce inf/nan logits directly
during the forward pass."* That's exactly it, with one inversion — it's fp16
that overflows, and you're on fp16 because the T4 can't do bf16.

---

## What I fixed

| Fix | File | What it does |
|---|---|---|
| Clamp logits to ±30 | `gpu/reranker.py`, `gpu/laya_rr.py` | `nan_to_num` then clamp, in fp32. sigmoid(30) = 1 − 1e-13, so it changes nothing real but stops one row poisoning a batch |
| **Skip** non-finite batches | both | if the loss is still non-finite, don't step. One bad backward writes nan into every weight and nothing recovers. Counted and reported |
| Gradient clipping | `gpu/reranker.py` | `clip_grad_norm_(1.0)`, which laya already had and e5 didn't |
| Validation batch ≤ training | both | was hardcoded 128/256 regardless of `--bs` — also the biggest VRAM spike |
| `AMLC_AMP` override | `gpu/reranker.py` | `bf16 / fp16 / fp32 / auto`. fp32 cannot overflow |
| GradScaler fallback | both | `torch.amp.GradScaler("cuda")` needs torch ≥ 2.4; requirements allow 2.2 |
| `transformers>=4.48` | `requirements-gpu.txt` | laya 0.3.20 needs 4.48+; the old `>=4.40,<5.0` pin could install a version it refuses to import |
| GPU auto-detect | `train_entry_gpu.py` | prints arch/bf16/VRAM, pins fp16 on Turing, passes env to the child |

---

## How to run it

**Branch:** `fix/t4-fp16` (cut from `finetune`, so it has all of Neal's work).

```bash
git fetch origin
git checkout fix/t4-fp16
```

Then launch the GPU job exactly as before. **No argument changes.**
`train_entry_gpu.py` now targets e5 and configures itself for the T4.

### What you keep

| Artifact | Still valid? |
|---|---|
| `folds.tsv` in your S3 bucket | ✅ **yes — required** |
| your other CPU-stage output | ✅ yes |
| the CPU stage itself | ✅ no rerun needed |

`folds.tsv` must be on the **`runinfo`** channel. It's the leak guard: it stops
the reranker training on entities the GBDT gets scored on. The entry point now
**refuses to start** if it's missing, rather than silently training a model
whose gain we couldn't trust.

### What you should see

```
[train_entry_gpu] GPU Tesla T4  sm_75  15.8 GB  bf16=False
[train_entry_gpu] no bf16 on this GPU (Turing) -> fp16 autocast, with logit
clamping and non-finite-batch skipping.
[train_entry_gpu] leak guard: /opt/ml/input/data/runinfo/folds.tsv
reranker: ... train / ... valid pairs, N steps, device cuda, amp torch.float16
  step 200/N  train 0.0xxx  valid 0.0xxx  ... pairs/s
```

The `valid` number appearing at step 200 is the thing that used to crash.

### If it still misbehaves

```bash
AMLC_AMP=fp32 ...      # cannot overflow; ~2x slower, more memory
AMLC_BS=16 ...         # if it's VRAM rather than overflow
```

A handful of `skipped non-finite loss` lines out of thousands of steps is
harmless — the run continues and the total is reported at the end. **Hundreds**
means fp16 isn't viable for that configuration and fp32 is the answer.

### Knobs (env vars or SageMaker hyperparameters)

| Var | Default | Note |
|---|---|---|
| `AMLC_ENTITIES` | 40000 | entities to build training pairs from |
| `AMLC_BS` | 32 | e5-small is 22M trainable, so a T4 has room |
| `AMLC_LR` | 3e-5 | |
| `AMLC_EPOCHS` | 2 | our current model used 1 |
| `AMLC_AUGMENT` | 0.3 | token drops/swaps/translit noise on **train rows only** |
| `AMLC_AMP` | auto | `fp32` is the escape hatch |

---

## What to report back

From `meta.json` in the output dir:

```
valid_auc        compare against 0.99907
valid_logloss    compare against 0.03339
```

Those are our current `models/rr_e5s`, trained without augmentation for 1
epoch. **If your fine-tune doesn't beat both, it isn't an improvement** — say
so and we keep the existing one. A reranker that looks better on its own valid
split but doesn't improve the downstream F0.5 is worth nothing, so the real
test is a 30k A/B afterwards:

```bash
python src/run_v2.py --sample 30000 --train-only --rerank <your new dir>
python src/run_v2.py --sample 30000 --train-only --rerank models/rr_e5s
```

The number that matters is **India** in `per_country` from `summary.json`.

Also flag immediately if you see:
- `skipped non-finite loss` more than a few times
- `reranker: WARNING N of M batches skipped` at the end
- anything about loading on CPU instead of CUDA (means VRAM ran out)

---

## Context

`docs/T4_FIX.md` — the fp16 diagnosis in full.
`docs/CHASING99.md` — where the submissions stand, what's been measured, and
why 99% isn't reachable (our blocking caps a *perfect* matcher near 0.980).
