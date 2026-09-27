# Why the laya fine-tune NaNs on SageMaker but not on the laptop

**Short version: the T4 has no bfloat16.** Everything else follows from that.

## The cause

`_torch()` picks the autocast dtype from the GPU:

```python
amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

| Box | GPU | Arch | bf16? | gets |
|---|---|---|---|---|
| Priyanshu's laptop | RTX 3050 | sm_86 Ampere | yes | **bf16** |
| `ml.g4dn.xlarge` | T4 | sm_75 **Turing** | **no** | **fp16** |

bf16 and fp16 are both 16 bits, but they spend them differently:

| | exponent | max value |
|---|---|---|
| bf16 | 8 bits, same as fp32 | ~3.4e38 |
| fp16 | 5 bits | **65504** |

bf16 trades precision for range, so an activation that would overflow fp16
cannot overflow bf16. Every run we did was on the laptop, in bf16, so this
failure mode was invisible to us. The code was never wrong here; it was only
ever exercised on hardware that hides the problem.

## Why it dies at the first validation, not during training

This is the detail that makes it look mysterious, and it is the strongest
evidence for the diagnosis.

`_diffs` **sorts by length** before batching, to cut padding:

```python
order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
```

So validation concentrates all the **longest** sequences into the same few
batches. Attention sums scale with sequence length, so those are the largest
activations the model will ever see. Training batches are drawn at random and
never concentrate long sequences that way.

Result: fp16 holds up through 200 random training batches and overflows on the
first length-bucketed validation batch. Once one row is `inf`, the loss is
`nan`, the gradients are `nan`, and every trainable weight is `nan` from that
step onward. The run is dead even though only one row was bad.

It also explains the two things that ruled out the earlier hypotheses:

- **Lowering the learning rate did not move it.** It was never divergence. The
  weights were fine; a forward pass overflowed a number format.
- **`--train-layers -1` made it worse.** Unfreezing all 322M parameters lets
  activations drift further from their pretrained scale, closer to 65504.

## The fixes

| Fix | File | What it does |
|---|---|---|
| Clamp the log-odds | `gpu/laya_rr.py` | `nan_to_num` then `clamp(-30, 30)` on the fp32 output. sigmoid(30) is 1 - 1e-13, so this costs nothing real and stops one row poisoning a batch |
| Skip non-finite batches | `gpu/laya_rr.py` | if the loss is still non-finite, **do not step** - one bad backward writes nan into every weight and nothing recovers. Skips, counts, warns |
| Validation batch <= training | `gpu/laya_rr.py` | `bs=min(bs, 64)`; it had been running at 128 regardless of `--bs`, and the longest-sequence bucket was also the biggest VRAM spike |
| `AMLC_AMP` override | `gpu/reranker.py` | `bf16 / fp16 / fp32 / auto`. fp32 cannot overflow |
| GradScaler compatibility | both | `torch.amp.GradScaler("cuda")` needs torch >= 2.4; falls back to `torch.cuda.amp.GradScaler()` |
| transformers pin | `requirements-gpu.txt` | was `>=4.40,<5.0`, but laya 0.3.20 needs `>=4.48` - the old pin could resolve to a version laya refuses to import |
| GPU report + batch | `train_entry_gpu.py` | prints arch/bf16/VRAM, sets fp16 explicitly on Turing, defaults `--bs 16` |

## How to run it

No change needed. `train_entry_gpu.py` detects the T4 and configures itself.
It will print:

```
[train_entry_gpu] GPU Tesla T4 sm_75 15.8 GB  bf16=False
[train_entry_gpu] no bf16 on this GPU -> fp16 autocast with logit clamping.
```

If you still see `skipped non-finite loss` warnings, escalate:

```bash
AMLC_AMP=fp32 python train_entry_gpu.py      # cannot overflow, ~2x slower
AMLC_BS=8     python train_entry_gpu.py      # if VRAM is the problem instead
```

A handful of skipped batches out of thousands is harmless - the run continues
and the count is reported at the end. Hundreds means fp16 is not viable for
this configuration and fp32 is the answer.

## Worth knowing before spending more GPU hours

The reranker is **proven worth +0.008 on the leaderboard**: submission 002
scored 0.951 against 001's 0.943 using a *weaker* 30k model. So improving it
is a reasonable thing to chase.

But in our head-to-head, **laya lost to e5** on India - the metric it was built
to win - at 5.8x the training cost:

| | baseline | laya | e5 |
|---|---|---|---|
| decision | 0.9512 | 0.9608 | 0.9607 |
| **India** | 0.9369 | 0.9460 | **0.9469** |
| train time | - | 132 min | **23 min** |

The premise was that e5 handles native-script names poorly, but the incumbent
is `intfloat/multilingual-e5-small` - already multilingual. So the A/B was
multilingual against multilingual, which would explain a null result more
simply than anything about mmBERT.

If the goal is a better reranker, fine-tuning **e5** is the cheaper and
currently better-performing starting point. Full detail in `docs/CHASING99.md`.
