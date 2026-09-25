# Setup — do this tonight

Roughly 12 hours to kickoff. Phases A–D are the critical path; E–H can run in parallel
or be split across the team. Times are wall-clock estimates, not effort.

Everyone on the team does A, B, C, D, G in **their own** account. Only the leader does F.

---

## A. AWS Free Tier account (~20 min) — blocks everything else

Each team member needs their **own** account: own email, own phone number, own card.
Four people sharing one account defeats the point — the credits are per-account.

1. Go to <https://aws.amazon.com> → **Create an AWS Account**.
2. Root email: use a personal address you will still have after graduation. Gmail `+` aliases
   (`you+aws@gmail.com`) work if you need a fresh one.
3. Account name: anything. Verify the emailed code.
4. Set a root password. Save it in a password manager, not in the team chat.
5. Contact info → **Personal** account type.
6. **Payment**: a card with international and recurring transactions enabled. AWS runs a
   ~₹2 refundable authorization. Indian debit cards frequently fail here — if yours does,
   try a credit card or a different bank's card. This is the single most common blocker.
7. Phone verification (SMS or voice).
8. Support plan: **Basic — free**.
9. On the plan screen choose the **Free plan**. This is what grants the $100 signup credit
   and up to $100 more as you use foundational services.
10. Sign in to the console. **Set the region to Asia Pacific (Mumbai) `ap-south-1`** in the
    top-right region picker. Every teammate uses the same region — cross-region S3 copies
    bill data transfer against your credits.

**Immediately after, before anything else:**

11. Enable MFA on the root user: account menu → **Security credentials** → Multi-factor
    authentication → assign a virtual MFA device (Google Authenticator / Authy).
12. Account menu → **Account** → scroll to *IAM user and role access to Billing Information*
    → **Edit** → tick **Activate** → Update. Without this, the IAM user you make in step C
    cannot see billing or credits.

---

## B. Student verification (~10 min, runs in the background)

Do this in parallel with A — SheerID may take up to 48h if it asks for documents.

1. <https://builder.aws.com> → sign in with your Builder ID.
2. Click your name → **Manage profile** → **Edit profile** → **Student details**.
3. **Start verification** → SheerID → institution name, enrollment details, upload a
   student ID / enrollment letter / class schedule if prompted.
4. While you wait, finish the profile — **upload a photo** and **fill the About section**.
   Both are required for the welcome package, and they earn your first two badges.

Payout: 12-month Skill Builder subscription on verification, $10 credits at 7 badges,
$20 at 14 badges, $100 certification voucher at 21.

Rejections are almost always an ineligible institution, an unclear scan, or enrollment
dates that don't show *current* enrollment. Retry with a different document.

---

## C. AWS CLI + credentials (~15 min)

### C1. Install

PowerShell:

```powershell
winget install --id Amazon.AWSCLI --accept-source-agreements --accept-package-agreements
```

Close and reopen the terminal, then confirm:

```bash
aws --version
```

If `winget` fails, download the MSI from
<https://awscli.amazonaws.com/AWSCLIV2.msi> and run it.

### C2. Make an IAM user — do not use root access keys

Root access keys cannot be scoped or easily revoked, and AWS actively warns against them.
Two minutes of IAM now saves a leaked-key incident later.

1. Console → search **IAM** → **Users** → **Create user**.
2. User name: `hackathon-cli`. Leave console access **unchecked** (this identity is for the
   CLI only).
3. Permissions → **Attach policies directly** → tick **AdministratorAccess**.
   (Broad, but scoped to a throwaway hackathon account with a hard credit ceiling.)
4. Create user → click into it → **Security credentials** tab → **Create access key**.
5. Use case: **Command Line Interface (CLI)** → acknowledge the warning → Create.
6. **Download the .csv.** The secret key is shown exactly once.

### C3. Configure

```bash
aws configure
# AWS Access Key ID     -> from the csv
# AWS Secret Access Key -> from the csv
# Default region name   -> ap-south-1
# Default output format -> json
```

Verify:

```bash
aws sts get-caller-identity
```

The `Account` field in that output is your **12-digit account ID**. Post it in the team
chat — step F needs everyone's.

Never commit the csv or `~/.aws/credentials`. The repo `.gitignore` already blocks
`.aws-credentials`, but the real file lives in your home directory.

---

## D. Spend guardrails (~10 min) — do this before launching anything

Credits do not stop charges; they offset a bill that still gets generated. An idle GPU
notebook left over a weekend is a real card charge once credits run out.

1. Console → **Billing and Cost Management** → **Budgets** → **Create budget**.
2. **Use a template (simplified)** → **Zero spend budget** → enter your email → Create.
   This alerts the moment any real (post-credit) charge appears.
3. Create a second budget: template **Monthly cost budget**, amount `10` USD, same email.
   This one catches runaway usage before it eats the whole credit pool.
4. Billing → **Billing preferences** → enable **AWS Free Tier alerts** and
   **Receive CloudWatch billing alerts**.
5. Bookmark Billing → **Free Tier** — it shows free-tier consumption per service.

Team rule worth agreeing on now: **whoever starts a GPU instance is responsible for
stopping it.** Put it in the chat pinned message.

---

## E. SageMaker GPU quota (~5 min to file, hours to approve) — file this first

New accounts often have a quota of **0** for GPU instances. Requesting it at hour 20 of the
hackathon is too late.

1. Console → region **ap-south-1** → search **Service Quotas**.
2. **AWS services** → **Amazon SageMaker**.
3. Search `ml.g4dn.xlarge` and request an increase on both of these:
   - `ml.g4dn.xlarge for training job usage` → request **2**
   - `ml.g4dn.xlarge for notebook instance usage` → request **1**
4. Also request `ml.g5.xlarge for training job usage` → **1**, as a fallback if g4dn is
   capacity-constrained.
5. Justification: "Student team participating in the Amazon ML Challenge 2026, 25–27 Sep.
   Training a multimodal model for the competition."

Small increases on new accounts are often auto-approved in minutes; some go to a human.
Check Service Quotas → **Quota request history**.

> **Have a backup.** Kaggle Notebooks give 30 free GPU hours/week per account and Colab
> gives a free T4 — across four people that is a lot of parallel experimentation with zero
> setup and zero credit burn. Use AWS for the long training runs and the artifact relay;
> use Kaggle for fast iteration. Create/verify Kaggle accounts tonight (phone verification
> is required to enable GPU, and that takes a few minutes you won't have on Day 1).

---

## F. Shared bucket — done

**Superseded.** The bucket exists: `amazon-cuda-commandos-2026` in `ap-south-1`,
cross-account, every member granted read+write.

Setup, auth and the helper script are in
[`TEAM_BUCKET.md`](TEAM_BUCKET.md). Short version:

```bash
aws login --region ap-south-1 --profile amlc   # not a static access key
./aws/s3.sh ls
```

The four `aws/0*.sh` scripts this section used to describe were written for a
bucket we ended up not creating ourselves; `aws/s3.sh` replaced them.

---

## G. Local environment (~30 min, mostly downloads)

Your global Python has **pandas 3.0.3**, which breaks a lot of code written against
pandas 2.x — including most public competition notebooks. Use a venv.

Git Bash:

```bash
cd /c/Users/Priyanshu/OneDrive/Desktop/All_projects/amazon_ml_challenge
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Check whether you actually have a usable GPU (my earlier check ran in a sandbox and was
inconclusive):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

- `True` → you have a local GPU; do first-pass training locally and save credits.
- `False` with no NVIDIA card → all training goes to AWS/Kaggle. Plan accordingly.

Pre-download the backbones now, so Day 1 isn't spent watching progress bars:

```bash
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
for m in ["microsoft/deberta-v3-base",
          "distilbert-base-uncased",
          "sentence-transformers/all-MiniLM-L6-v2"]:
    print("fetching", m)
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)

import timm
for m in ["convnext_tiny", "tf_efficientnet_b0"]:
    print("fetching", m)
    timm.create_model(m, pretrained=True)
print("done")
PY
```

That lands in `~/.cache/huggingface` and `~/.cache/torch`. Roughly 2 GB.

Smoke-test the metric so you know the harness works:

```bash
python -c "import sys; sys.path.insert(0,'src'); from metrics import smape; print(smape([100],[120]))"
# expect 18.1818...
```

---

## H. Team coordination (~10 min)

Agree on these in the chat *tonight*, because renegotiating them at hour 30 costs a
submission slot:

- **Region**: `ap-south-1`. Everyone.
- **Bucket**: `hackathon-<team-name>`.
- **Git repo**: one private GitHub repo, everyone pushes. Data stays out of it.
- **Framework version**: pin the exact torch + transformers versions now. A checkpoint
  trained on torch 2.5 will not load cleanly in a 2.2 container.
- **Roles**: someone owns the CV harness and submission pipeline end-to-end, someone owns
  text, someone owns images, someone owns the approach document. The document is graded
  and is the usual thing teams leave to the last 40 minutes.
- **Submission log**: a shared sheet — every submission's local CV, public LB score, and
  what changed. Without it you cannot tell which of hour 50's six ideas helped.

---

## Timetable for tonight

| Time | What |
|---|---|
| now | Start A (AWS account) and B (SheerID) in parallel — both have external waits |
| +30 min | C: install CLI, make IAM user, `aws sts get-caller-identity`, post account ID |
| +45 min | **E: file the SageMaker quota request** — longest external wait |
| +55 min | D: budgets and alerts |
| +70 min | F: leader creates bucket, everyone tests read access |
| +80 min | G: venv + pip install + weight downloads (leave it running) |
| +100 min | H: team agreements, then sleep. Day 1 starts at 09:00 and runs 72 hours. |

Sleep is a competitive advantage in a 72-hour event. Do not pull an all-nighter the night
*before* it starts.
