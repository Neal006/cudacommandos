# Submission 005 — hopeso, no reranker

**Status: HOLD (validated fallback).** Box `e5` scores 0.9704 on the same
holdout against this run's 0.9614, so this should not take an upload slot
unless `e5` and `params` both fail.

---

## What this is

| | |
|---|---|
| run | `runs/013_hopeso_a` (cloud, `ml.m7i.48xlarge`, 192 vCPU) |
| candidates | **hopeso** — base TF-IDF top-30 ∪ sibling expansion |
| **blocking recall** | **0.9630** (base was 0.9498) |
| model | 150k sample, two-stage LightGBM, **no reranker** |
| stage 1 → stage 2 | 0.9526 → 0.9602 |
| **holdout (50k unseen entities)** | **0.9614** |
| holdout per country | India 0.9539 · US 0.9664 |
| decision | `assign=none`, `select=expected_f`, `miss=0.05` |
| singletons | 0.0591 predicted (true train rate 0.0563) |

This is the first run on Neal's hopeso frames. Sibling expansion lifted
blocking recall **0.9498 → 0.9630** for 8.15 extra candidates per entity, and
stage 1 went 0.9494 → 0.9526 as a result.

## Why it is held

It has no band reranker. Box `e5` is the same frames, same sample, plus the
e5-small reranker on the 0.2–0.8 band, and scores **0.9704** on the identical
holdout — **+0.0090**. There is no argument for spending a slot here.

Its value is as a fallback and as the control that isolates hopeso's
contribution from the reranker's.

## Validation

Fetched from S3 and checked with `tools/pick_submission.py`, which refuses to
rank anything the official validator rejects:

```
amlc-a-p98b   holdout 0.9614   oof 0.9602   India 0.9520   US 0.9656   PASS
```

## Decision-layer sweep run on this run's holdout

`src/decide_source.py` (branch `decide-by-source`) cross-fits every recipe
across two entity halves and reports the held-out score:

```
assign  recipe                  held-out   overfit
none    global threshold         0.9745    0.0001
hard    global threshold         0.9749    0.0001
hard    per source (S2 / S3)     0.9749    0.0002
hard    per country              0.9749    0.0004
  per source thresholds: S2 0.68, S3 0.71
```

Per-source thresholds are real — S2 and S3 do want different cutoffs, the
direction two public solutions report — but worth **+0.0004** here, under the
+0.0010 adoption gate. Kept the global threshold.

Note these absolute values are not comparable to `holdout_score`: the sweep
counts each entity's true links *from the candidate set*, ignoring links
blocking never retrieved. The comparison between recipes is valid; the level
is optimistic.
