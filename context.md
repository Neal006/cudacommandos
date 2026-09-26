# What we've built, step by step

## The problem, in plain words

Imagine three phone books from three companies, all listing the same shops but written differently:

- **Book 1 (S1)** is the clean master list. Each business appears once. Example: "Sharma Sweets, 12 MG Road, Pune".
- **Books 2 and 3 (S2, S3)** are messy. The same shop might appear as "SHARMA SWEETS PVT LTD, MG Rd", as "शर्मा स्वीट्स" (Hindi script), or three times with typos. About a quarter of their entries are "distractors" that match nothing in Book 1.

**Our job:** for every business in Book 1, list which entries in Books 2 and 3 are the same business. Some businesses have zero matches. The correct answer for those is "none".

The only information we get per entry is a **name**, an **address** and a **country** (US, India, France). There are no phone numbers or map coordinates.

**The data sizes:**
- **Training data (with answers):** 2.2M businesses in Book 1 and about 10M entries in Books 2 and 3. It covers only the US and India.
- **Test data (no answers; we're graded on it):** 1.73M businesses in Book 1 and about 10M entries in Books 2 and 3. It covers the US (38%), India (47%) and **France (15%). France never appears in the training data.**

## How we're graded

For each Book 1 business, the grader compares our list with the true list and gives it a score from 0 to 1. The final score is the average over all businesses.

The score is called **F0.5**. It combines two things:
- **Precision:** of the matches we claimed, how many were right?
- **Recall:** of the true matches, how many did we find?

F0.5 cares **twice as much about precision**. A wrong match hurts more than a missed one. So the system is built to be careful: when unsure, leave it out.

**Where we stand:** our current leaderboard score is **0.943** (rank 933). The target is **0.985**.

## The pipeline: 7 stages

Think of it as a factory line.

### Stage 1: Loading
We read the huge text files and convert them into a compact, fast format (Parquet), using fast tools (Polars) and running several parts in parallel. This cut memory use by about half, which matters on a laptop.

*In plain terms:* we unpack the boxes and put everything on labelled shelves.

### Stage 2: Cleaning ("normalization")
We make every name and address comparable:
- **Lowercase, strip accents:** "Café" becomes "cafe".
- **Remove legal words:** "Pvt Ltd", "LLC" and "SARL" are dropped, but we remember them separately as a clue.
- **Expand abbreviations:** "Rd" becomes "road" and "Av" becomes "avenue". France has its own rule tables.
- **Transliterate:** Hindi, Kannada and other Indian scripts are converted to Latin letters, so "शर्मा" becomes "sharma". We measured that this recovers about 94% of shared words.
- **"Skeleton" key:** a rough sound-alike version of the name, so spelling variants collapse together.

*In plain terms:* everyone is forced to write in the same handwriting before we compare them.

### Stage 3: Shortlisting ("blocking")
Comparing 1.7M businesses against 10M entries one by one would take forever: about 17 trillion pairs. So for each Book 1 business we quickly pull the **30 most similar-looking entries** from the same country and ignore everything else.

We do this with **TF-IDF**, a scoring method that treats **rare shared words as strong evidence**. Sharing "Sharma" means something; sharing "road" means almost nothing.

- The test data ends up with **about 52 million candidate pairs** instead of trillions.
- **Cost:** about 5% of true matches never make the shortlist of 30. Since those can never be recovered, **the best possible score is about 0.99**. That's our "ceiling".
- On the test data this step takes about 52 minutes.

*In plain terms:* before interviewing suspects, the detective narrows a city of millions down to 30 per case.

### Stage 4: Measuring similarity ("features")
For each of the 52M pairs we compute about 46 numbers describing how alike they are:
- How similar the names are, measured several ways (order-insensitive, partial match, etc.).
- How similar the addresses are, and whether the **house numbers agree**.
- Whether the transliterated or skeleton versions match.
- Whether the legal forms are compatible ("Ltd" vs "Pvt Ltd" is fine; "LLC" vs "Trust" is suspicious).
- How **generic** the name is ("Sai Medical" is common; "Xylo Quantics" is unique).
- How this candidate **ranks** among the 30 for this business.

*In plain terms:* a fingerprint comparison sheet for each suspect.

### Stage 5: The judge, in two rounds (machine learning)

**Round 1.** A **LightGBM** model reads the 46 numbers and outputs a probability, such as "92% likely the same business". LightGBM builds hundreds of small yes/no decision trees and adds up their votes. It learned from the training answers.

**Round 2.** A second model looks at the **context**, not just the pair:
- "Is this entry also claimed strongly by a *different* Book 1 business?" This matters because each entry belongs to at most one business.
- "How does this candidate compare to the best one?"
- "Does it look like the other entries we already matched?" (peer similarity)

*In plain terms:* round 1 judges each suspect alone; round 2 looks at the whole lineup.

### Stage 6: Making the final call ("decision layer")
We turn probabilities into yes/no answers:
1. **Calibrate:** make sure "80%" really means right 80% of the time.
2. **Assign:** since an entry can belong to only one business, give disputed entries to the strongest claimant.
3. **Expected-F choice:** for each business, pick the set of matches that **maximizes the expected score**, rather than using one fixed cutoff for everyone.

### Stage 7: Writing and checking the output
We write the answer file (one row per Book 1 business, with its matches) plus the shortlist file the organizers audit. Then we run the official validator and our own checker.

## Safety net: mlguard
A separate watchdog program written in **Rust** (a fast, crash-resistant language). It:
- **Watches training live** and stops it if the model starts memorizing instead of learning ("overfitting").
- **Checks the final file** for problems like missing rows, bad IDs, or suspiciously empty output.
- **Runs automatically on GitHub** whenever code changes.

*Why:* no file gets uploaded without passing it.

## How we measure ourselves before submitting
We can't see the test answers, so we use **cross-validation**. We split the training businesses into 5 groups, train on 4, test on the 5th, and rotate. The resulting score is called **OOF** ("out-of-fold").

| Version | OOF score |
|---|---|
| Original simple pipeline | 0.927 |
| New features (v2) | 0.950 |
| + round 2 context | 0.953 |
| + decision layer | 0.9532 |

By country, India scores 0.940 and the US 0.962.

## Why the leaderboard (0.943) is lower than our internal score (0.953)
Most likely:
1. **The test set has more India**, which is our weaker country.
2. **France was never seen in training.** Working backwards from the numbers suggests France is scoring around 0.90. That's an estimate, not measured.

Analysis of this gap is still in progress.

## What's built but not yet in use
- **GPU "reranker":** a small AI language model (e5-small) that reads a pair of records and judges whether they match. It's used only on uncertain cases. In one test it added about +0.009, but **we don't trust that gain yet** because it may have seen some records in training that it was then tested on. It also needs the laptop GPU, which was broken last boot.
- **A teammate's branch** that adds sentence-embedding features (another AI model's view of text similarity). It hasn't been evaluated in the main pipeline yet.

## One-sentence versions for any audience
- **Non-technical:** "We built a system that reads three messy business directories and figures out which listings are the same shop, even across spelling differences and different alphabets."
- **Semi-technical:** "Clean the text, shortlist 30 likely matches per business, score each pair with a trained model, then use context and the one-owner rule to make final picks."
- **Technical:** "Normalization with transliteration and skeleton keys, TF-IDF top-k blocking (ceiling 0.99), a 46-feature LightGBM stage 1, a stage-2 GBDT with competition and peer features, calibrated partition-aware expected-F0.5 decisions, all gated by a Rust run and submission checker."

## Caveats
- The exact code-level details (precise feature list, how France is handled at scoring time) are still being traced.
- The France ≈ 0.90 figure is back-calculated, not measured.
- The reranker's +0.009 is unverified.