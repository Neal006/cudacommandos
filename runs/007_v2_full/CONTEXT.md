# Run 007 — first full-sample run, first test phase

**Run id** `v2_20260926_0140` · started 2026-09-26 01:40 IST · machine:
Windows 11, 10 physical cores, 25.5 GB RAM, dataset on `D:\amlc_data`.

```bash
PIPELINE=src/run_v2.py tools/mlguard/train_guarded.sh 007_v2_full --sample 150000
```

**Outcome in one line:** training and test blocking both finished and are
banked on disk; the run then hung in `split_stats`, so there is **no
submission from it**. The expensive half survived — see §5.

---

## 1. What it was for

Runs 003–005 all trained on 30k Source-1 entities, which was chosen to make
iteration fast, not because 30k was enough. Two open questions:

1. Does the score keep climbing with more training entities, or had 30k
   already saturated the feature set?
2. What does the **test** phase actually cost? Nobody had run it. Every
   number in `docs/EXPLAINER.md` for it was an estimate.

Both got answered. The second one embarrassingly so.

## 2. Results

OOF macro F0.5 on 150k entities, GroupKFold by Source-1 entity, 5 folds:

| Layer | 30k (run 004) | **150k (this run)** | Δ |
|---|---|---|---|
| old features (run 001 baseline) | 0.9266 | — | — |
| stage 1 — pairwise LightGBM | 0.9490 | **0.9500** | +0.0010 |
| stage 2 — stacker on OOF stage-1 | 0.9508 | **0.9526** | +0.0018 |
| decision layer | 0.9512 | **0.9532** | +0.0020 |

**More data helps, and it helps more the further down the stack you go.**
Stage 2 gains twice what stage 1 does. That makes sense: stage 2's features
are competition and peer statistics (`claim_rank`, `n_strong_claims`,
`peer1_sim`), which are only meaningful when the entity's candidate list is
densely covered by the training sample. 5x the entities means those
statistics are computed over a far less sparse graph.

Per country:

| Country | OOF F0.5 |
|---|---|
| US | 0.9620 |
| India | 0.9399 |
| France | *(no labels — test only)* |

**India trails the US by 2.2 points.** Run 002's blocking diagnostic already
found India lagging by ~6pp at the candidate-generation stage, so part of
this is inherited: the matcher cannot recover a true pair that blocking never
proposed. The rest is the script problem (~23% of India Source-2 names are in
one of 9 Indic scripts) and the fact that Indian addresses have no postal
code to anchor on.

Blocking recall on the training sample: **0.9498** at K=30. Converting a
recall ceiling to the best F0.5 a perfect matcher could reach —
`1.25R/(0.25+R)` — that is a ceiling of **0.9903**. We are at 0.9532, so
**the gap to a perfect matcher is 3.7 points and the gap to a perfect blocker
is 1.0 point.** Blocking is not the bottleneck. Matcher precision is.

### The decision layer barely earns its keep

`decision_table.csv` has all 288 combinations. Best of each family:

| `select` | best F0.5 |
|---|---|
| `expected_f` | 0.953182 |
| `threshold` (single global cut) | 0.952573 |

**+0.0006.** And within `expected_f`, the `soft` assignment beats doing no
conflict resolution at all (`none`) by 0.00007 — seven ten-thousandths.

Neal's own bar for keeping added complexity is 0.003. Expected-F clears
neither that nor any reasonable noise floor at 150k entities. It is kept as
the default because it is already written, tested and costs nothing at
inference — but **nobody should spend another hour tuning it.** The
per-entity decision theory is not where the remaining 3.7 points are.

## 3. Timings (measured, not estimated)

| Stage | Wall clock |
|---|---|
| train blocking (150k entities, 10.3M index) | 8.4 min |
| feature build, 4.5M pairs x 46 | 2.4 min |
| stage 1, 5 folds | 33 min |
| stage 2, 5 folds | 28 min |
| decision-layer tuning | 7 min |
| **training total** | **2 h 11 m** |
| test blocking — France | 4.1 min |
| test blocking — India | 34.0 min |
| test blocking — US | 13.8 min |
| **test blocking total** | **52 min** |

### The EXPLAINER's test-phase estimate was wrong by 4.4x

`docs/EXPLAINER.md` §11 budgeted **255 minutes** for test blocking. It took
**52**. The estimate extrapolated from the training run's throughput
(75–150 queries/sec). Measured test rates, each with the index it was scored
against:

| Partition | Queries | Index | q/s | Wall clock |
|---|---:|---:|---:|---:|
| France | 259,452 | 1,434,993 | 1,066 | 4.1 min |
| US | 663,106 | 3,817,031 | 800 | 13.8 min |
| India | 809,986 | 4,717,565 | 397 | 34.0 min |

Throughput is not a property of the machine — it is a property of the index.
Two things drive it: index size, and how dense the posting lists are for the
query's tokens. US searches **fewer** records than India and is still twice
as fast, because Indian names and addresses share more tokens. India is 34 of
the 52 minutes on the strength of that density, not because it has the
biggest index.

Anything derived from the 255-minute figure is wrong, including the "~7 hours
end to end" conclusion. Corrected in `EXPLAINER.md` §11 and `DATA_BRIEF.md`.

## 4. How it died

```
[7887.8s] cached candidates -> cands_test_k30_df0.01_mdf3_ctry1_nall.parquet (51,974,499 pairs)
Exception in thread Thread-19 (_handle_workers):
  ...
  File "multiprocessing\popen_spawn_win32.py", line 101, in duplicate_for_child
    return reduction.duplicate(handle, self.sentinel)
PermissionError: [WinError 5] Access is denied
```

`WinError 5` reads like a permissions problem and is not one. Windows returns
it from `DuplicateHandle` when there is not enough memory to set up the new
process's handles. Free RAM at that moment was 1.4 GB of 25.5.

**It did not crash — it hung.** 0% CPU, no child processes, 10.2 GB resident,
silent. `split_stats` called `Pool(workers)` unconditionally, unlike
`record_table` beside it which guards on `workers > 1`. On the test split it
runs with the 52M-pair frame already resident, so the parent was at ~10 GB
when it tried to spawn. A worker died, the pool could not replace it (the
traceback above, on the pool's *maintenance* thread), and `pool.map` waited
forever for a result that would never arrive.

Stalling is the worse failure mode. A crash would have freed the memory and
printed a cause; instead 52 minutes of finished blocking sat on disk while a
dead process held 10 GB. Fixed in `c701d01` — `split_stats` now takes the
serial path when `workers <= 1` and frees the source lists before spawning.

`stats_test_v1.pkl` on disk was **not** produced by this run. It was
regenerated standalone after the fix (63 seconds), which is why its mtime
03:56:46 sits 21 seconds before that commit. Worth knowing before anyone
reads the timestamps as evidence the run got further than it did.

**A second wall is waiting behind this one.** The test phase builds one
feature frame for all 51,974,499 pairs: 46 float64 columns over 52M rows is
~19 GB, doubled by stage 2's `concat`, on a 23.7 GB box. Run 007 never
reached it, but a retry would. Fixed pre-emptively in `predict_test_chunked`
(commit `9aad360`) — see §7.

## 5. What to reuse

Everything expensive is on disk. A relaunch **skips blocking and stats
entirely** and goes straight to scoring.

| Artifact | Size | Cost to rebuild |
|---|---|---|
| `<DATA_DIR>/interim/cands_test_k30_df0.01_mdf3_ctry1_nall.parquet` | 758 MB | 52 min |
| `<DATA_DIR>/interim/stats_test_v1.pkl` | 153 MB | ~4 min |
| `<DATA_DIR>/interim/cands_train_k30_df0.01_mdf3_ctry1_n150000.parquet` | 66 MB | 8 min |
| `runs/007_v2_full/model.pkl` | 35.7 MB | 2 h 11 m |

The two test caches are the ones worth sharing — they reproduce blocking
exactly, so one person runs it and everyone else pulls the result:

```bash
./aws/s3.sh share-cache        # push
./aws/s3.sh get-cache          # on another machine
```

**The candidate cache key does not include a normalizer version.** If you
edit `normalize.py` in a way that changes `_blob`, these caches are stale and
nothing will tell you. Purge them by hand.

### The 50 missing entities

The cache holds 51,974,499 pairs over **1,732,494 distinct Source-1
entities**. Test has **1,732,544**. Fifty entities got zero candidates.

That is 0.003%, and it is expected rather than a bug: an entity whose name
and address produce no token surviving `min_df=3` / `max_df=0.01` pruning has
nothing to match against. They must still appear in `matching_results.tsv`
with an empty match list — the submission is rejected if any Source-1 id is
missing. `data.write_outputs` iterates the full test id list rather than the
prediction frame, so this is already handled; it is called out here so nobody
rediscovers it as a panic at upload time.

## 6. mlguard

`mlguard watch` stopped stage-2 fold 0 at iteration 210:

```
[FAIL] loss_ratio   fold 20 iter 210: valid/train loss 1.53 > 1.5
```

The rollback worked as designed — the fold kept `best_iter 149` rather than
the stopped iteration, and its valid score (0.9494) is in line with the other
four folds. The guard did its job.

**Fold numbering gotcha:** `summary.json`'s `folds` array shows folds 20–24.
Those are *stage-2* folds 0–4; the runlog encodes fold id as `stage*10 + fold`.
Stage-1 folds are 0–4 and are not in `summary.json` at all.

**Known mlguard bug:** `checks.rs` iterates `for fd in &s.folds` with no stage
filter, and the fold record is `{fold, train_score, valid_score, best_iter,
max_iter}` — there is no `stage` field to filter on. So `--ablate` runs, which
hardcode `stage=9` for the baseline, are guaranteed to trip a false
`overfit_gap` failure. Tracked in PR #3; do not treat that failure as real.

## 7. What to do next

1. **Rerun the test phase on the chunked code** to get the first submission.
   With both caches present this is scoring only — roughly 1.5 h, and peak
   memory stays around 1.5 GB per chunk instead of 19 GB in one lump.
2. Leave the decision layer alone (§2). The 3.7 points to the ceiling are in
   the matcher, and India is where they are concentrated.
3. **France has no label-free diagnostic yet.** Blocking is unsupervised, so
   France should track US rather than India, but that is a hypothesis and
   nothing has measured it. The test cache is enough to check it without
   labels: per-country orphan rate and top-candidate similarity distribution.
   If France's distribution looks like India's rather than the US's, 15% of
   the test set is in trouble and we would not otherwise find out until the
   leaderboard says so.
