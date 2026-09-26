# AWS — current state

AWS is **not on the critical path**. This pipeline is single-threaded CPU work
(sparse matmul, rapidfuzz, LightGBM); no stage uses a GPU, and cloud cores are
not faster per-core than the laptop. See `EXPLAINER.md` §11 for the measured
comparison.

Kept here so nobody re-derives the account state. The step-by-step setup this
file used to contain is done and lives in git history.

## Account

```
account id   654479364872
region       ap-south-1 (Mumbai)
profile      amlc
auth         aws login  (browser, 12h sessions, renewable 90 days)
credits      $120        plan reports PAID
```

Re-authenticate when a command says the session expired:

```bash
aws login --region ap-south-1 --profile amlc
```

**Do not create static access keys.** A key pair in `~/.aws/credentials` never
expires on its own and is the usual way credentials leak out of a hackathon
repo. The SSO session above is already configured.

Two gotchas that cost real time, in case they recur:

- Sign-in loops endlessly if a Builder ID session is live on the same email.
  Fix: incognito → console.aws.amazon.com → **Root user**. Builder ID and AWS
  account are separate credentials sharing one address.
- The account returned `NotSignedUp` / `OptInRequired` on S3, EC2 and Lambda
  until the payment method was verified, even though sign-in worked. Service
  activation is separate from login.

## Spend guardrails (live)

```
zero-spend-budget    $0.01   alert at 100% actual
monthly-10-usd      $10.00   alert at 80% actual
```

Both notify `doshipriyanshu3@gmail.com`. The plan reports `PAID`, so spend past
credits bills the card — the alerts are the safety net.

## SageMaker quotas (granted)

```
ml.g4dn.xlarge  training job       2
ml.g4dn.xlarge  spot training      2
ml.g4dn.xlarge  notebook instance  4
ml.g4dn.xlarge  endpoint           2
Studio JupyterLab / CodeEditor     4
```

All defaulted to **0** and needed increase requests, which went to human review
rather than auto-approval. If anyone else on the team needs GPU, file early.

**Worth using for exactly one thing:** embedding ~12M records for the
Indic-script gap (Krisha's branch). Hours on CPU, 20–40 min on a T4. Nothing
else in the pipeline benefits.

## Shared bucket

`amazon-cuda-commandos-2026`, ap-south-1, cross-account. Setup, helper script
and the candidate-cache workflow are in [`TEAM_BUCKET.md`](TEAM_BUCKET.md).

## Free alternatives

- **Kaggle Notebooks** — 30 GPU-hours/week, 16 GB VRAM. Needs phone
  verification once. The practical choice for embedding experiments.
- **Builder Center sandbox** — real AWS accounts, 8 h/week, no card.
