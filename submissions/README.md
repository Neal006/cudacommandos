# submissions/

Every leaderboard upload we make, one folder per attempt, newest number
highest. **Krisha uploads from here** — nothing else in the repo needs to be
touched to submit.

Each folder is self-contained: the file to upload, what produced it, and what
it scored. If a submission turns out to be our best, this is the record that
proves which code and which model made it.

---

## Index (updated 2026-09-27 16:25)

| # | model | blocking recall | OOF | **LB** | status |
|---|---|---|---|---|---|
| 001 | 150k, no reranker | 0.9498 | 0.9532 | **0.943** | uploaded, rank 934 |
| 002 | 30k + e5 reranker | 0.9496 | 0.9607 | **0.951** | uploaded |
| 003 | 150k + e5 reranker | 0.9498 | 0.9626 | **0.953** | uploaded, rank ~1760 |
| 004 | 150k + e5 + contention fix | 0.9498 | **0.9644** | not uploaded | HOLD |

Every file here is `matching_results.tsv.gz`. Unzip before uploading; the
portal takes the `.tsv`. `candidate_pairs.tsv` is never uploaded and is not in
git (694 MB, not scored).

The OOF-to-leaderboard gap has been stable at **-0.0102 / -0.0097 / -0.0096**,
which is what makes OOF a usable predictor at all. Tonight's runs report a
`holdout_score` instead, computed on 50k unseen entities scored by the fold
mean -- the way test is scored -- so it should need no such correction.

004 is the odd one out: it is our best offline model and has never been
uploaded, because the hopeso runs finishing this evening are built on better
candidates (recall 0.9630 against its 0.9498) and are expected to beat it.

---

## For Krisha — what to upload to Unstop

**Upload one file: `matching_results.tsv`.** That is the only file the
leaderboard scores.

1. Open the newest folder here (highest number — `001`, then `002`, …).
2. Read its `README.md`. It says what changed, what it scored offline, and
   whether it is recommended for upload. **Some folders are marked DO NOT
   UPLOAD** — a submission we built, validated and then chose not to spend a
   slot on. Check this before anything else.
3. Get `matching_results.tsv`:
   - It is stored **gzipped** (`matching_results.tsv.gz`, ~20 MB) because the
     raw file is too big for GitHub's 100 MB limit.
   - Unzip it before uploading. Windows: right-click → 7-Zip → Extract Here,
     or `gzip -d matching_results.tsv.gz` in Git Bash. On the Unstop portal
     upload the **`.tsv`**, not the `.gz`.
4. Upload to the Unstop submission portal.
5. **Tell the team what it scored**, and add the number back to that folder's
   `README.md` (or send it to Priyanshu to record). A leaderboard score we
   did not write down is a slot we cannot learn from.

### Before you click submit

- **5 submissions per day, and the window closes 27 Sep 2026 23:59 IST.**
  The button disables after the fifth. Check `docs/EXPERIMENTS.md` for what
  has already gone up today.
- If the portal rejects the file, do not re-upload a guess — post the exact
  error in the team chat. Every file here has already passed the organisers'
  own validator locally, so a rejection means something we have not seen
  before and re-uploading burns another slot.

### What you do *not* upload to the leaderboard

`candidate_pairs.tsv` is **not** scored and does not go to the leaderboard. It
is audited as part of the final package only. It is ~1 GB, so it is not in git
at all — see *Where the big files live* below.

---

## The other deliverable (not the leaderboard)

Separate from these uploads, the team submits one final package,
`cudacommandos_submission.zip`, containing both TSVs, all the code, and the
filled-in `Documentation_template.md`. That is built by:

```bash
tools/package_submission.sh
```

It refuses to build if the row count is wrong, the official validator fails,
or the documentation is still template text — so if it errors, read the
message rather than working around it.

---

## Where the big files live

| File | Size | Where |
|---|---|---|
| `matching_results.tsv` | ~70–100 MB | here, gzipped, in each folder |
| `candidate_pairs.tsv` | ~1 GB | S3 (see below) — too big for GitHub |
| test candidate cache (`.parquet`) | 758 MB | S3 |
| trained model (`model.pkl`) | 35.7 MB | `runs/<id>/`, gitignored — S3 |

GitHub hard-rejects any file over 100 MB, which is why the big artifacts are
in the team bucket instead:

```bash
./aws/s3.sh ls                  # see what is there
./aws/s3.sh pull <member> <path>
```

The bucket is `amazon-cuda-commandos-2026` (ap-south-1, cross-account) and
everyone writes under their own prefix. If a command says the session
expired:

```bash
aws login --region ap-south-1 --profile amlc
```

Details in [`../docs/TEAM_BUCKET.md`](../docs/TEAM_BUCKET.md).

---

## What each folder contains

```
submissions/001/
├── README.md                  # what this is, what it scored, upload or not
├── matching_results.tsv.gz    # the file to upload, gzipped
└── MANIFEST.md                # provenance: commit, model, config, checks
```

`MANIFEST.md` records the git commit, the run that produced the model, the
decision-layer settings and the validator output. It exists so that a score on
the leaderboard can be traced back to exactly the code that made it — which
matters because the organisers may ask top teams for their source, and because
"which run was our 0.95?" is a question we will otherwise be guessing at on
the last day.

---

## Reference

- [`../docs/SUBMISSION.md`](../docs/SUBMISSION.md) — the format rules, the
  metric, what causes outright rejection, and the pre-upload checklist
- [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) — the run log; every
  upload gets an entry
- [`../runs/`](../runs/) — training run artifacts and per-run context
