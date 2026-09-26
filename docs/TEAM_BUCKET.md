# Shared S3 bucket

```
bucket   amazon-cuda-commandos-2026
region   ap-south-1 (Mumbai)
access   cross-account — every member's AWS account is granted read+write
```

It will not appear in your own S3 console. Cross-account grants are addressed
by name over the CLI; that is expected, not a misconfiguration.

Everyone writes under their **own prefix** (`priyanshu/`, `krina/`, `krisha/`,
`neal/`) so concurrent uploads cannot overwrite each other.

---

## Authentication — use `aws login`, not access keys

```bash
aws login --region ap-south-1 --profile amlc
```

Opens a browser, issues credentials valid 12 hours and renewable for 90 days.
Re-run it whenever a command reports `Your session has expired`.

**Do not create a static access key for this.** `aws configure` with a key pair
writes a long-lived secret to `~/.aws/credentials` in plaintext that never
expires on its own, has to be rotated by hand, and is the single most common
way AWS credentials leak out of a hackathon repo. The SSO session above is
strictly better and is already configured.

If you genuinely need a static key for a machine that cannot open a browser,
create it scoped to a dedicated IAM user with S3 access to this bucket only —
never the account root — and delete it when the challenge ends.

## Usage

The helper wraps the right profile, region and prefix:

```bash
./aws/s3.sh ls                      # what's in the bucket
./aws/s3.sh ls krina                # one member's files
./aws/s3.sh push model.txt          # -> s3://.../priyanshu/model.txt
./aws/s3.sh push sub.tsv output     # -> s3://.../priyanshu/output/sub.tsv
./aws/s3.sh pull krina models/lgbm.txt
```

Set `AMLC_MEMBER` to your own name on your machine (defaults to `priyanshu`):

```bash
export AMLC_MEMBER=krina
```

Raw CLI equivalents, if you prefer:

```bash
aws s3 ls s3://amazon-cuda-commandos-2026/ --profile amlc --region ap-south-1
aws s3 cp ./file s3://amazon-cuda-commandos-2026/<you>/ --profile amlc --region ap-south-1
```

## What is actually worth sharing

**The candidate cache.** Test blocking is 1.73M Source-1 queries against a
~10M-record index — roughly **4–5 hours** of compute (measured: 75 q/s on the
India partition, 150 q/s on US). The resulting parquet reproduces it exactly.

One person runs it once, everyone else pulls the result and starts from the
matcher:

```bash
# whoever ran the pipeline
./aws/s3.sh share-cache

# everyone else
./aws/s3.sh get-cache priyanshu
```

That drops the files into `<data dir>/interim/`, where `run_pipeline.py` picks
them up automatically. The cache filename encodes every parameter that changes
the result (`cands_test_k30_df0.01_mdf3_ctry1_nall.parquet`), so a mismatched
config recomputes rather than silently using the wrong candidates.

Also worth uploading: trained models, submission TSVs you scored, and anything
that took more than a few minutes to produce.

**Not worth uploading:** the dataset itself. It is ~2.4 GB, everyone already
has it, and it is gitignored for the same reason.

## Cost

S3 storage is about $0.025/GB/month in ap-south-1 — a few cents for the whole
challenge. Cross-region transfer is not free, so **keep everything in
ap-south-1**; the helper pins the region for you.

## If access fails

| Symptom | Cause |
|---|---|
| `Your session has expired` | Re-run `aws login --region ap-south-1 --profile amlc` |
| `AccessDenied` on the bucket | Your account ID is not in the bucket policy — send it to whoever owns the bucket (`aws sts get-caller-identity`) |
| Bucket missing from your S3 console | Expected. Cross-account access is by name over CLI only. |
| `aws: command not found` | CLI not on PATH. On Windows it installs to `%LOCALAPPDATA%\Programs\Amazon\AWSCLIV2\aws.exe` |
