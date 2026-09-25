# Data brief — Business Entity Resolution

Every number here was measured directly from the released files on 2026-09-25.
Nothing is estimated or carried over from the problem statement.

---

## 1. Inventory

| File | Rows | Size |
|---|---:|---:|
| `train/train_source1.tsv` | 2,206,821 | 210 MB |
| `train/train_source2.tsv` | 5,034,616 | 489 MB |
| `train/train_source3.tsv` | 5,285,603 | 504 MB |
| `train/train_ground_truth.tsv` | 2,206,821 | 127 MB |
| `test/test_source1.tsv` | 1,732,544 | 175 MB |
| `test/test_source2.tsv` | 4,887,273 | 509 MB |
| `test/test_source3.tsv` | 5,082,316 | 506 MB |

Columns in every source file: `entity_id`, `business_name`, `business_address`,
`country`. Source is encoded only in the `entity_id` prefix (`S1-`/`S2-`/`S3-`);
there is no source column.

**Scale is the dominant engineering constraint.** Test blocking is 1.73M
Source-1 entities against 9.97M S2+S3 records. Any approach that materialises a
dense similarity matrix is off by ~10 orders of magnitude and will not
complete. See §6.

## 2. Ground truth is perfectly aligned

```
train_source1 entity_ids   2,206,821
ground_truth  entity_ids   2,206,821
present in both            2,206,821
GT ids missing from S1              0
S1 ids with no GT row               0
```

Exactly one ground-truth row per Source-1 record, including singletons (empty
`matched_entity_ids`).

> **Note on an earlier analysis.** An initial EDA pass reported ~253k S1 rows,
> ~18k ground-truth rows, and only 2,116 IDs overlapping — which would imply a
> badly broken dataset. That came from a partially-extracted download, not from
> the data. On the complete files the overlap is exact. If your row counts do
> not match the table in §1, re-extract before drawing conclusions.

## 3. Match distribution

| Matches | Entities | Share |
|---:|---:|---:|
| 0 (singleton) | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10+ | 571 | 0.03% |

```
total S2 links     3,693,619
total S3 links     3,944,746
mean links/entity  3.46
```

**Singletons are a minor term at 5.6%.** The metric gives a full 1.0 for
correctly predicting an empty list, which makes them look strategically
important, but 94.4% of entities *do* have matches and the typical count is
2–5. Being too conservative loses far more than it protects. S2 and S3 carry
roughly equal link mass, so dropping either source costs ~half your recall.

## 4. Country distribution shifts between train and test

| Split | US | India | France |
|---|---:|---:|---:|
| train_source1 | 60.0% | 40.0% | — |
| train_source2 | 59.9% | 40.1% | — |
| train_source3 | 60.0% | 40.0% | — |
| **test_source1** | **38.3%** | **46.8%** | **15.0%** |
| test_source2 | 38.3% | 47.3% | 14.4% |
| test_source3 | 38.3% | 47.3% | 14.4% |

This is the most consequential property of the dataset and it is easy to miss.

- **France is 15% of test and 0% of train** — roughly **260,000 Source-1 test
  entities** with no training examples of their naming or address conventions.
  Anything learned as a country-specific rule will not transfer to them.
- **The US/India ratio inverts.** A model tuned on a 60/40 US-majority training
  set is evaluated on a 38/47 India-majority test set, so per-country
  validation matters more than the aggregate number.

Practical consequences:

1. Treat `country` as an opaque string. Never enumerate, one-hot, or filter on
   `{US, India}` — the problem statement calls this out explicitly and the
   distribution is why.
2. French vocabulary (`SARL`, `SAS`, `Bd`, `Rue`, `Résidence`) can only come
   from a lookup table; it cannot be learned from training data. See
   `src/normalize.py`.
3. Validate per country, not just overall. Use `scores_breakdown()` and split
   by country to see whether the India-heavy slice is dragging.
4. Accent folding is mandatory — `République` and `Republique` must collide.

## 5. Field quality

| File | Blank address | Blank name |
|---|---:|---:|
| train_source2 | 3.36% | 0.00% |
| train_source3 | 3.33% | 0.00% |
| test_source2 | 2.65% | 0.00% |
| test_source3 | 2.68% | 0.00% |
| both source1 files | 0.00% | 0.00% |

Source 1 is clean on both fields. ~3% of S2/S3 records have no address at all,
so any feature that assumes a non-empty address needs a defined fallback —
those records must still be matchable on name alone.

Noise patterns confirmed in the data (consistent with the problem statement):
abbreviated street types (`Rd`/`Road`, `M.G. Rd`/`Mahatma Gandhi Road`), legal
suffix variation (`Pvt. Ltd`/`Private Limited`, `SARL` present/absent),
landmark references (`Near SBI ATM`), punctuation and possessives
(`Orelee's`/`Orelees`), and accented vs unaccented French text.

## 6. What the scale forces

At 1.73M × 9.97M, the candidate-generation strategy is the whole ballgame.

- **Brute-force nearest neighbours is impossible.** ~10¹³ comparisons.
- **Character n-grams do not belong in blocking.** They explode the nonzero
  count across ten million records. They are excellent *features* — run them
  over the few dozen candidates per entity that survive blocking instead.
- **Word-level TF-IDF with document-frequency pruning does work.** Two records
  only produce a nonzero if they share a surviving rare token, so memory
  tracks real co-occurrences rather than n×m. Chunked sparse matmul keeps peak
  memory bounded. This is what `src/blocking.py` implements.
- **Training does not need all 2.2M entities.** A pairwise matcher converges on
  a small fraction; `C.TRAIN_SAMPLE` defaults to 150k. Blocking and inference
  still run over the full test set — only matcher training is subsampled.

Measured on this machine: 25.5 GB RAM total but ~9.7 GB free, 10 physical
cores. RAM, not CPU, is the binding constraint.

## 7. Measured blocking ceiling

Run on a 150k Source-1 sample against the full 10.3M S2+S3 index
(`max_df=0.01`, `min_df=3`, per-country partitions). **Zero orphans** — every
entity got candidates.

| K | recall ceiling | F_0.5 ceiling | test pairs at that K |
|---:|---:|---:|---:|
| 5 | 0.8340 | 0.962 | 8.7M |
| 10 | 0.9187 | 0.983 | 17.3M |
| 20 | 0.9411 | 0.988 | 34.7M |
| **30** | **0.9499** | **0.990** | **52.0M** |
| 50 | 0.9586 | 0.992 | 86.6M |

**Recall is not the score, and reading this table as "still climbing, raise K"
is the trap.** F_0.5 weights precision 2×, so a perfect matcher limited only by
blocking recall R scores `1.25R / (0.25 + R)` — losing 4% of recall costs under
1% of score. K=30 → 50 buys **+0.002** of ceiling for **35 million** extra pairs
to featurize.

`TOP_K = 30` is the setting. The conclusion that matters:

> **Blocking is not the bottleneck. Matcher precision is.**
> Spend time on features and the threshold, not on recall.

### Throughput (this machine, 10 physical cores)

| Partition | Index size | Queries/sec |
|---|---:|---:|
| India | 4.13M | 75 |
| US | 6.19M | 150 |

India runs 2× slower on a *smaller* index — Indian names and addresses share
more tokens, so posting lists are denser and each query touches more of the
matrix. Budget accordingly: full test blocking (1.73M queries: ~810k India,
~663k US, ~260k France) is **roughly 4–5 hours**.

Candidates are cached to `<data dir>/interim/cands_*.parquet`, keyed on every
parameter that changes the result, so the cost is paid once and matcher
experiments reuse it.

### If you re-tune blocking

| Observation | Action |
|---|---|
| Ceiling short at your K | Raise `BLOCK_MAX_DF` (0.01 → 0.05) before raising `TOP_K` — it recovers pairs lost to pruning rather than just adding weak candidates. |
| Many orphans | `BLOCK_MAX_DF`/`BLOCK_MIN_DF` pruned every token those records had. Loosen both. |
| Suspect country labels disagree across sources | Set `BLOCK_WITHIN_COUNTRY=False` and compare recall. Costs a lot of speed, so only as a diagnostic. |
