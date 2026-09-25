# GPU Plan (RTX 3050, 6 GB) — after review

The first GPU plan (chat, 25 Sep) was reviewed against measurements and four points
were wrong. This file is the corrected plan and what `nealstuff` implements.

## Review findings → what changed

| # | Finding (measured) | Change |
|---|---|---|
| 1 | kNN held **all** index shards on the GPU: US = 4.75 GB index + 2.05 GB scores = 6.8 GB > 5.41 GB free → OOM | `src/gpu/embed_knn.py` keeps shards on the CPU and moves **one shard at a time**; shard size derived from free VRAM (`torch.cuda.mem_get_info`). Peak ≈ shard + q_chunk×shard scores (~3.6 GB on 6 GB, less on 4 GB). Tested vs brute force with 7 streamed shards (`tests/test_embed_knn.py`). |
| 2 | Full run: test blocking 255 min, test features 90 min, **LightGBM 8 min** — GPU tree training saves ~5 of 414 min | **Job 2 dropped.** The CPU stages got the work instead: `sparse_dot_topn` top-k blocking (identical candidates, 2.7× faster on a 2M index) and vectorized features (identical values, 115 s → 42 s on 900k pairs incl. 11 new features). |
| 3 | 7.1 GB free RAM; blocking peaks 6–8 GB; 10.3M embeddings = 7.93 GB → concurrent CPU+GPU thrashes | Embeddings are streamed to fp16 `.npy` while encoding and read back with `np.load(mmap_mode="r")`; the tool says **do not run it next to CPU blocking**. `src/ingest.py` also cut the S2 load peak to 3.1 GB (Arrow strings), which is what lets the rest of the pipeline share the box. |
| 4 | Run 002 sized the entire blocking backlog at **+0.005 F0.5 ceiling**; "0.5 pp recall" ≈ 0.001 F0.5, far too lenient a kill | Job 1 is a **measurement tool only** (`--measure`), kill criterion **test-weighted F0.5-ceiling gain < +0.002 → DROP**. Job 4 (fine-tuning the embedder) is dropped with it. |

## What the GPU does now

The bottleneck is matcher precision (OOF ~0.90 vs ~0.99 ceiling), so the GPU goes there.

| Job | Status | Command | Budget |
|---|---|---|---|
| **3. Cross-encoder reranker** on the stage-1 uncertain band | implemented | `python src/gpu/reranker.py train --exclude runs/<id>/folds.tsv --entities 40000 --out models/rr_e5s` then `python src/run_v2.py ... --rerank models/rr_e5s --band 0.2 0.8` | measured on a 4 GB 3050: train 257k pairs in 11 min (~410 pairs/s), score ~1,870 pairs/s; band = 1.2% of pairs (EXPERIMENTS.md 003–005) |
| 1. Embedding kNN | measurement only | `python src/gpu/embed_knn.py --measure 30000` (needs the pass-A cache of that sample) | encode ~10M records once (hours on a 3050) — run only if a teammate's box is idle |
| 2. GPU trees | dropped | — | — |
| 4. Embedder fine-tune | dropped | — | — |

### Reranker design (why it fits 6 GB)
- Backbone `intfloat/multilingual-e5-small` (MIT, 118M). 96M of those are the word-embedding
  table, which is **frozen** → AdamW state for ~22M params. bf16 autocast on Ampere.
- Pairs serialized as `name: <norm name> | <transliteration if different> addr: <norm address>`.
- Hard negatives = highest blocking-similarity non-matches (3 per positive).
- **Leakage guard:** it trains only on entities outside the GBDT sample, and writes
  `entities.txt`; `run_v2.py --rerank` aborts if any of its own entities are in it.
- Its score enters **stage 2 as a feature** (`rr`, −1 outside the band), so the GBDT decides
  how much to trust it; the reranker never overrides decisions directly.

### Setup (Priyanshu's box)
```powershell
python -m venv .venv-gpu; .venv-gpu\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/<cu12x-or-cu13x>
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python tests/test_embed_knn.py
python src/gpu/reranker.py bench        # pairs/s on this card before committing to a band size
```
Never run GPU encoding and CPU test blocking at the same time on a 16–25 GB machine.
