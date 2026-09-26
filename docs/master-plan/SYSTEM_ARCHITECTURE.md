# System Architecture & End-to-End Implementation Plan

## 1. System view

```
                        ┌──────────────── GitHub (Neal006/cudacommandos) ────────────────┐
                        │  branches: pipeline/entity-resolution (code) · Master-Plan (plan)│
                        │  Actions: mlguard.yml → unit tests · fixtures · gate runs/*/     │
                        └───────────────▲───────────────────────────────▲──────────────────┘
                                        │ push code + runs/<id>/summary │
     ┌──────────── Laptop (25 GB, 16 thr) ───────────┐      ┌───────── AWS (credits) ─────────┐
     │ dev, unit tests, 50k-entity smoke runs        │      │ r6i/r7i.8xlarge-class (32 vCPU, │
     │ EDA (polars)                                  │      │ 256 GB) for full blocking +      │
     │                                               │      │ featurize; g5.xlarge only if the │
     │                                               │      │ cross-encoder / ANN is used      │
     └───────────────────────┬───────────────────────┘      └──────────────┬──────────────────┘
                             │  s3://hackathon-<team> (private, teammate-only policy)        │
                             └────────── interim/*.parquet · runs/* · output/* ──────────────┘

 Pipeline process (one run)                                  Guard process (parallel)
 ─────────────────────────                                  ─────────────────────────
 L0 ingest → L1 contract → L2 normalize → L3 block ─┐        mlguard watch runs/<id>/metrics.jsonl
                                                    ├─ cache  (writes MLGUARD_STOP on violation;
 L5 stage-1 → stage-2 → L6 decide → L7 write ◄──────┘         the LightGBM callback stops the fold)
        │ RunLog: metrics.jsonl, folds.tsv, summary.json
        ▼
 mlguard split · mlguard run (--champion) · validate_submission.py · mlguard submission --summary
        ▼
 leaderboard upload (≤5/day) → docs/EXPERIMENTS.md entry → git tag lb-YYYYMMDD-n
```

Instance sizes are a guess based on the RAM Priyanshu measured. Check remaining
credits first:

```bash
aws freetier get-account-plan-state --region us-east-1 --profile amlc
```

(`aws/04_check_credits.sh` was replaced by `aws/s3.sh`; see
[`../TEAM_BUCKET.md`](../TEAM_BUCKET.md).)

## 2. Data flow and artefacts

| Stage | Input | Output (cached) | Cache key | Budget (AWS / laptop) |
|---|---|---|---|---|
| L0 ingest | `dataset/{train,test}/*.tsv` | `interim/{split}_s{n}.parquet` | file size + mtime | 2 / 5 min |
| L1 contract | parquet | `runs/<id>/contract.json` | — | < 1 min |
| L2 normalize | parquet | `interim/{split}_norm.parquet` | normalizer version hash | 10 / 40 min |
| L3 blocking | norm parquet | `interim/cands_{split}_{hash}.parquet` | passes + K + df params | 60–90 min / 5 h |
| L5 features | cands + norm | `interim/feats_{split}_{hash}.parquet` | feature-set version | 20 / 60 min |
| L5 train | train feats | `runs/<id>/models/*.txt`, `oof.parquet` | run id | 30 / 45 min |
| L6 decide | scores | `runs/<id>/decision.json` (p_min, mode, miss) | run id | 5 min |
| L7 write | decisions | `output/*.tsv` | run id | 2 min |

Everything downstream of a cache hit reruns in minutes, so matcher/decision experiments
never re-block.

## 3. Team split

| Track | Owner | Scope |
|---|---|---|
| **T1 Candidates** | Priyanshu | L0–L4: polars ingest, normalizer v2 (translit, skeleton, domain, FR tables), passes B/C/D, per-slice recall |
| **T2 Matcher & decision** | Neal | L5–L6: vectorized features, stage 1/2, calibration, assignment, expected-F, mlguard wiring |
| **T3 Trust & delivery** | shared | mlguard in every run, EXPERIMENTS.md log, submission zip, Documentation_template.md |

Interface between T1 and T2: the **candidates parquet schema** (LLD §2). T2 builds on the
existing pass-A cache while T1 adds passes, so the tracks don't block each other.

## 4. Implementation plan: challenge window 25 Sep 00:00 → 27 Sep 23:59 IST

Submissions: 5/day. **Never upload a run that `mlguard run` fails.**

### Phase 0: tonight (25 Sep, remaining hours)
| # | Task | Owner | Done when |
|---|---|---|---|
| 0.1 | Run the existing full pipeline (already built) under `tools/mlguard/train_guarded.sh 002_baseline` with the LLD §9 hooks | Neal | OOF F0.5 + summary.json; mlguard PASS |
| 0.2 | **LB #1**: baseline upload → calibrates the LB-vs-OOF gap | Neal | score logged in EXPERIMENTS.md |
| ~~0.3~~ ✅ | E03: recall ceiling split by country × script × domain × empty-address | Priyanshu | **Done** — run 002 in EXPERIMENTS.md. India 0.9132 vs US 0.9735 @K30; empty address is the strongest miss signal (7.8×). `src/analyze_blocking.py` |
| 0.4 | E07+E08 on baseline OOF: assignment + expected-F vs global threshold | Neal | Δ OOF measured |
| 0.5 | Start AWS instance + S3 sync of `dataset/` and `interim/` | Priyanshu | `aws s3 ls` shows parquet |

### Phase 1: 26 Sep morning
| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Normalizer v2: translit + skeleton (prototype exists) + domain split + legal form + FR/IN/US tables + unit tests on ANALYSIS §1.5 rows | Priyanshu | `pytest tests/test_normalize.py` green |
| 1.2 | Vectorized features (`cpdist`) + the new feature groups (LLD §5) | Neal | 1M pairs featurized in < 60 s |
| 1.3 | Decision layer in the pipeline (`decide.py`) if E07/E08 won | Neal | OOF ↑ and mlguard PASS |
| 1.4 | **LB #2**: baseline + decision layer | Neal | logged |

### Phase 2: 26 Sep afternoon/evening
| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | Passes B, C, D + union; E02/E04/E05 | Priyanshu | recall ≥ 0.985 at ≤ 60 cands/entity (train sample) |
| 2.2 | Geo-cluster training sample (E06) | Neal | OOF on the full-locality slice compared |
| 2.3 | Stage-2 competition + peer features (E09) | Neal | Δ OOF ≥ +0.003 or dropped |
| 2.4 | LOCO transfer check (E10) and France drift check on test | Neal | per-country report |
| 2.5 | Full test run on AWS with blocking v2 + features v2 | Priyanshu | outputs + `mlguard submission` PASS |
| 2.6 | **LB #3–#5** (day 2): v2 pipeline; v2 + stage-2; best + decision tweak | both | logged |

### Phase 3: 27 Sep (final day)
| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Optional: cross-encoder on the uncertain band (E14) only if Phase 2 finished by morning | Neal | Δ ≥ +0.003 in OOF on the band |
| 3.2 | Seed/fold ensemble of the champion (E15) | Neal | fold std ↓ |
| 3.3 | Freeze the champion by noon: `runs/champion.json`, tag `final-candidate` | both | mlguard PASS with `--champion` |
| 3.4 | Submission zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` filled | both | zip rebuilt from a clean clone reproduces the files (hash compare) |
| 3.5 | **LB #1–#4** (day 3): champion variants; keep **#5 in reserve** for a last-hour fix | both | logged |
| 3.6 | Final upload = champion (rule R1 default) before 22:00 IST | Neal | confirmation screenshot |

### Critical path and fallbacks
- The critical path is **test blocking v2 (2.5)**. If AWS isn't ready by 26 Sep 14:00, run
  passes B/C only for India (the measured gap) and keep pass A elsewhere.
- If the vectorized features slip, cap test candidates with the pass-A rank ≤ 20 ∪ other passes
  ≤ 10 (the candidate file must then be exactly that capped set).
- Every phase leaves a submittable champion, so nothing depends on the last phase landing.

## 5. Definition of done (per run)
1. `mlguard watch` quiet during training. 2. `mlguard split` PASS. 3. `mlguard run
--champion` PASS. 4. `validate_submission.py` PASS. 5. `mlguard submission --summary` PASS
(France cardinality within 0.6–1.4× OOF). 6. EXPERIMENTS.md entry + `runs/<id>/summary.json`
committed. 7. Tag `lb-YYYYMMDD-n` if uploaded.
