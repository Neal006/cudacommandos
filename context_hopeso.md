# context (branch `nealultraprohopeso`)

Okay so here is the full scene, simple language only.

## Where we are

We are at 0.953 on the leaderboard. The problem is entity resolution: for every Source 1 business, find all its copies in Source 2 and Source 3. Metric is macro F0.5 per entity, so precision is king, and an entity with no matches gives full 1.0 only if we predict nothing.

The pipeline so far: word TF-IDF blocking picks top 30 candidates per entity, LightGBM stage 1 scores them, stage 2 adds context features, the e5 reranker cleans the confusing middle band, then a calibrator plus an expected F decoder decides how many links to keep.

## The real bottleneck, measured

Everyone was saying "matcher, matcher". But see the math. Blocking recall is 0.9496, so even a perfect matcher tops out around 0.990 offline, and we lose about 0.010 more going offline to leaderboard. So 98 is basically not possible unless we fix recall and the gap both. 99 is not happening with these candidates, let's be honest about that.

So I sat and profiled the 5,225 true pairs that blocking missed on the 30k sample. The finding was a bit shocking yaar:

- 22% of missed records have the exact same cleaned name as the S1 business
- 36% have the same transliterated skeleton (this catches Hindi, Tamil etc script names)
- 25% have an empty address

These are not tough pairs at all. TF-IDF top 30 just ranks them out, because generic names and address tokens outvote the name.

## What I built (`src/hopeso.py`)

Two cheap jugaad passes, unioned with the normal blocker:

1. **sibs**: every source keeps 5 to 6 copies of the same business. So take an entity's top 3 candidates, and pull all records in the same country with the same name key. Find one copy, get the whole gang. Pools bigger than 50 are skipped so "Medical Store" type names don't flood us.
2. **dost**: direct lookup from the S1 name key, but only when that name is rare (pool of 10 or less).

Both run on two keys: exact core name, and the translit skeleton.

On 30k train:

| | recall | extra cands per entity | F0.5 ceiling |
|---|---|---|---|
| base | 0.9496 | 0 | 0.9897 |
| hopeso union | 0.9623 | 8.2 | 0.9922 |

So we recover 25% of the misses for 27% more pairs. One important detail: `block_sim` is a model feature, so for the new pairs I recompute it with the same vectorizer, otherwise the model sees garbage.

## Gang features in stage 2

Research on Foursquare 2022 (the closest Kaggle match) says graph and sibling tricks gave the biggest jumps, not fancy transformers. So stage 2 now gets 4 new features: how many of this entity's candidates share this record's name key, and the best score among those siblings (excluding itself). Same for skeleton. If your sibling is strongly matched, you probably are also.

## Other fixes on this branch (from PR 7)

- Holdout in `run_v4`: calibrator and decision are fitted on entities scored exactly like test. OOF uses one model per pair, test uses the mean of five, which is a sneaky calibration mismatch.
- Exact expected F decoder, reranker leak guard, and a check that stops a score file from being applied to the wrong row order.

## How to run on the big SageMaker box

Take a second `ml.m7i.48xlarge` (192 cores, 768 GB) so it runs in parallel with 004, not after it. First copy the full train and test candidate caches from run 011 into `interim/`, otherwise step 1 blocks again for 2 hours.

```
export AMLC_WORKERS=176 AMLC_BLOCK_THREADS=176
python src/hopeso.py build --split train
python src/hopeso.py build --split test
python src/run_v4.py --sample 150000 --rerank models/rr_e5s --rounds 4000 --chunk 4000000 --cands-tag hopeso --train-only
python src/score_test.py --run runs/<id> --rerank models/rr_e5s --chunk 4000000 --cands-tag hopeso
```

Upload only if the validator says PASS and links per entity per country look sane against 003.

## Lifting the limits (all knobs, no code edits)

Every limit that caps recall or training is now a setting. Defaults are unchanged, so the run 011 caches still match. Any change goes into the cache file name, so a stale frame can never be reused by mistake.

| Knob | Default | What it limits | Cost of raising it |
|---|---|---|---|
| `AMLC_VIBE="top,cap,dost,keep"` | `3,50,10,0` | sibling passes: top N candidates, biggest name pool, rare-name lookup pool, best new cands kept per entity | only `hopeso.py build` reruns, no re-blocking |
| `AMLC_TOP_K` | 30 | base shortlist size | full re-blocking of train + test |
| `AMLC_MAX_DF` | 0.01 | common words ignored in blocking | full re-blocking, more RAM |
| `--sample` (or `AMLC_TRAIN_SAMPLE`) | 150000 | businesses the GBDT trains on | linear in time; keep it below about 600k so the holdout and the reranker's 40k stay disjoint |
| `--rounds` | 2000 | boosting rounds | early stopping picks the real number |

Cheapest big lever first: pick `AMLC_VIBE` from the numbers, not by guess.

```
python src/hopeso.py measure --n 30000          # recall + new cands/entity for caps 50 / 200 / 1000
export AMLC_VIBE="5,1000,200,40"               # example: big pools searched, best 40 new kept
python src/hopeso.py build --split train
python src/hopeso.py build --split test
python src/run_v4.py --sample 400000 --rounds 4000 --cands-tag hopeso ...   # same AMLC_VIBE set
python src/score_test.py ... --cands-tag hopeso                              # same AMLC_VIBE set
```

`AMLC_VIBE` must be the same in all four commands, otherwise `load_frame` stops with "missing".

What this can and cannot buy: a perfect matcher at recall R scores 1.25R/(0.25+R). Recall 0.98 gives 0.995 offline and about 0.985 on the leaderboard with the usual 0.010 gap. So lifting recall raises the ceiling, but the score itself only moves as far as the matcher follows.

## Honest expectation

Recall fix is worth maybe +0.002 to +0.004, contention plus calibration maybe +0.003 to +0.008. So realistic landing is 0.958 to 0.965. Target 98 would need a different retrieval game altogether. Tests: `python tests/test_review_fixes.py`, all 8 passing.
