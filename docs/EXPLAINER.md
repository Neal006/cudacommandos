# How this all works — from fundamentals up

Written to be read start to finish by someone who has not touched the code.
No prior knowledge of entity resolution assumed. Every number in here was
measured from our data, not estimated.

---

## 1. The problem, in one paragraph

Three separate lists describe the same businesses. **Source 1** is Amazon's
clean, deduplicated list — one row per real business. **Source 2** and
**Source 3** are messy lists from elsewhere: same businesses, but with typos,
abbreviations, missing address pieces, different word order, sometimes written
in a different script entirely. None of the lists share an ID.

For every business in Source 1, find all the rows in Sources 2 and 3 that
describe that same business.

```
SOURCE 1  (clean reference)          SOURCES 2 & 3  (noisy)
S1-925783039                         S2-185405451  "Orelees Barber Shop"
"Orelee's Barbershop"        <---->                "1795 Westchester Dr"
"1795 Westchester Drive,             S3-286983356  "ORELEE'S BARBER SHOP"
 High Point, NC"                                   "1795 Westchester Drive, NC"
```

Same shop, three spellings, no shared identifier. That's the whole job.

This is a classic, well-studied CS problem called **entity resolution** (or
record linkage). It shows up whenever two databases about the same real-world
things have to be joined without a common key — merging customer lists,
deduplicating medical records, linking census data.

## 2. Why it's hard

### 2.1 Scale

```
test Source 1    1,732,544 businesses
test Source 2+3  9,969,589 candidate rows
```

If you compared every Source-1 business against every candidate row, that's
**1,732,544 × 9,969,589 ≈ 1.7 × 10¹³ comparisons** — seventeen trillion.

At a very optimistic one million comparisons per second, that's **200 days**.
We have three. So brute force is not "slow", it is *impossible*, and the
entire design follows from that one fact.

### 2.2 Noise

Real examples of what the same business looks like across sources:

| Kind | Source 1 | Source 2/3 |
|---|---|---|
| Legal suffix | `Acme Private Limited` | `Acme Pvt. Ltd` |
| Abbreviation | `Mahatma Gandhi Road` | `M.G. Rd` |
| Possessive | `Orelee's Barbershop` | `Orelees Barber Shop` |
| Landmark address | `12 Main Road, Bengaluru` | `12 Main Rd, Near SBI ATM` |
| Different script | `Maharashtra Traders` | `महाराष्ट्र ट्रेडर्स` |
| Missing address | `12 Main Rd, Bengaluru` | *(blank)* |
| Name is a URL | `Acme Corporation` | `www.acme.com` |

Only **20%** of true matching pairs have identical names after normalization.
So "just compare the strings" fails for four out of five real matches.

### 2.3 Two countries in training, three in testing

```
train:   US 60.0%   India 40.0%   France  0%
test:    US 38.3%   India 46.8%   France 15.0%
```

**France is 15% of the test set and appears nowhere in training.** About
260,000 businesses whose naming and address conventions the model has never
seen. Anything we hard-code for US or India will not transfer.

## 3. How you're scored: F0.5

For each Source-1 business you output a list of matching IDs. Two things can
go wrong:

- **Precision** — of the IDs you listed, how many were right?
- **Recall** — of the right IDs, how many did you list?

Normally these are balanced (that's "F1"). Here they are **not**. The metric
is **F0.5**, which weights precision **twice** as heavily as recall.

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Why? Because in the real world, **wrongly merging two different businesses is
worse than missing a link.** A missed link means incomplete data. A wrong
merge means one business's reviews, orders and ratings get attached to a
different business. That's corruption, not just a gap.

Worked example:

```
you predicted   [S2-00047, S2-00193, S3-00812]     3 guesses
truth is        [S2-00047,           S3-00812]     2 correct
precision       2/3 = 0.667      (one wrong guess)
recall          2/2 = 1.000      (found them all)
F0.5            0.714
```

You found everything, but one bad guess dropped you to 0.714.

The score is computed **per business, then averaged** ("macro"). Every
business counts the same whether it has one match or nine.

### The singleton trap

Some businesses have *no* match at all. Rules:

| Truth | You predict | Score |
|---|---|---|
| nothing | nothing | **1.0** |
| nothing | anything | **0.0** |
| something | nothing | **0.0** |

Predicting an empty list scores a perfect 1.0 when you're right. That makes
"say nothing when unsure" feel very safe.

**It's a trap.** We measured it: only **5.6%** of businesses have no match.
The other 94.4% average **3.46 matches each**. Being too cautious loses far
more than it protects.

## 4. The core idea: two stages

Since seventeen trillion comparisons are impossible, every entity-resolution
system splits into two stages:

```
        10,000,000 candidates per business
                    |
        STAGE 1: BLOCKING          "cheap and rough"
        throw away the obvious non-matches
                    |
              30 candidates per business
                    |
        STAGE 2: MATCHING          "expensive and careful"
        score each survivor properly, keep the good ones
                    |
              ~3 final matches
```

**An analogy.** You're looking for one person in a city of ten million. You
don't interview ten million people. You first narrow to "lives in this
neighbourhood, roughly this age" — cheap, rough, keeps everyone plausible.
*Then* you interview the few hundred who survive.

Blocking is the neighbourhood filter. Matching is the interview.

### The iron rule of blocking

> **If blocking throws away a true match, no amount of clever modelling gets
> it back.** Stage 2 only ever sees what stage 1 hands it.

This is why the very first thing we measured was: *what fraction of true
matches survive blocking?* That number is a hard ceiling on the final score.

## 5. Stage 1: how our blocking works

### 5.1 Normalize the text

Before comparing anything, squash away differences that don't matter.
(`src/normalize.py`)

```
"Acme Pvt. Ltd"                    -> "acme"
"ACME Private Limited"             -> "acme"
"12 M.G. Rd, Near SBI ATM, 560001" -> "12 m g road sbi atm 560001"
"Café Dupont"                      -> "cafe dupont"      (accents folded)
```

We strip legal suffixes (`Ltd`, `Pvt`, `SARL`), expand street abbreviations
(`Rd` → `road`), fold accents, and remove possessives.

A rule that fires on **both** sides of a pair is harmless — both become
"acme", they still match. Only *asymmetric* rewrites hurt. That principle
caught two real bugs in our own code (§8).

### 5.2 TF-IDF: score words by how rare they are

The key insight: **rare words identify, common words don't.**

If two businesses both contain "restaurant", that tells you almost nothing —
thousands do. If both contain "Orelee", that's nearly conclusive.

TF-IDF ("term frequency × inverse document frequency") turns each record into
a list of numbers where rare words get big weights and common words get tiny
ones. Two records are similar if they share *heavy* (rare) words.

We also **discard** any word appearing in more than 1% of records. Those words
produce enormous lists of matches and almost no signal. This is the single
setting that makes the whole thing computationally possible.

### 5.3 Sparse matrix multiplication

With each record as a vector of word weights, similarity between two records
is a **dot product**. Doing that for 1.7M × 10M pairs at once is impossible —
but almost every pair shares *zero* rare words, so almost every similarity is
exactly zero.

A **sparse matrix** stores only the non-zero entries. Multiplying two sparse
matrices only does work where records actually share a word. That turns an
impossible calculation into a few hours.

We process Source-1 records in chunks of 2,000 so memory stays bounded, and
keep the **top 30** candidates for each. (`src/blocking.py`)

### 5.4 Country partitioning

We only compare records that share a country label. Roughly a 3× reduction,
and near-certainly correct — a Bengaluru shop doesn't match a Texas one.

Important: country is treated as an **opaque string**, never a fixed list.
That's what lets France — unseen in training — work automatically.

## 6. Stage 2: the matcher

For each surviving candidate pair we compute about 30 numbers describing *how*
similar they are (`src/features.py`):

| Group | Examples |
|---|---|
| Name similarity | edit distance, token-sort ratio, Jaro-Winkler |
| Token overlap | Jaccard, containment |
| Address | same measures on the address |
| Numbers | do the house/unit numbers agree? (postal codes barely exist here — India 0.2%, US 11%, France 0.4%) |
| Acronyms | does "ich" match "indian coffee house"? |
| **Competition** | is this the *best* candidate for this business? how far ahead of second place? |

That last group matters more than it looks. A similarity of 0.8 that ranks
**first with a big gap** is very different from a 0.8 that ranks seventh.

These 30 numbers go into **LightGBM**, a gradient-boosted decision tree model.
It learns, from the training labels, which combinations mean "same business".

Then a **threshold**: keep every candidate scoring above some cutoff. We sweep
all possible cutoffs and pick the one maximising F0.5 on held-out data.

### Why not a neural network / LLM?

- The signal here is string similarity, which trees handle excellently
- LightGBM trains in minutes on a CPU; a transformer needs hours and a GPU
- Rules cap the model at 8B parameters, MIT/Apache licensed
- Decision trees are auditable — and the top 100 teams get their pipeline
  reviewed

A neural model *could* help in one specific place — see §9.

## 7. Guarding against fooling ourselves

**GroupKFold.** When splitting data for validation, all pairs belonging to one
business go into the same fold. Otherwise the model sees some of an entity's
answers while being tested on the rest, and reports a score it can't reproduce.

**Two leaderboards.** A public one during the challenge (a sample of test) and
a private one after (the rest). **Final ranking uses the private one.** So
when your local validation and the public board disagree, believe validation —
the public board is a small sample you can accidentally tune yourself onto.

**Five submissions per day.** About 15 total. Every upload should be
deliberate, and every one gets logged in `docs/EXPERIMENTS.md` with what
changed — otherwise a moving score tells you nothing.

## 8. What we've actually measured

### Run 001 — how good is blocking?

Recall = fraction of true matches that survive blocking.

| candidates kept (K) | recall | best possible F0.5 |
|---:|---:|---:|
| 10 | 0.9187 | 0.983 |
| 20 | 0.9411 | 0.988 |
| **30** | **0.9499** | **0.990** |
| 50 | 0.9586 | 0.992 |

The obvious reading — "recall is still climbing, keep more candidates" — is
**wrong**, and this is the single most useful thing we've worked out.

Recall is not the score. Because F0.5 weights precision double, a perfect
matcher limited only by blocking scores `1.25R / (0.25 + R)`. Losing 4% of
recall costs under 1% of score. Going from K=30 to K=50 buys **+0.002** and
costs **35 million extra pairs** to process.

**Conclusion: blocking is not the bottleneck. Matcher precision is.**

### Run 002 — where do the misses happen?

Neal's review challenged run 001: it was measured on a 60/40 US/India mix, but
test is 38/47/15. If misses concentrate in India, the number is optimistic.

He was right that it was masked:

| K | India | US |
|---:|---:|---:|
| 30 | **0.9132** | **0.9735** |

India is 6 points behind. But the conclusion survives — blocking doesn't learn
from labels, so France (Latin script, accents we already fold) behaves like
US, not India. Weighted by the real test mix, recall is ~0.946 vs 0.950. The
ceiling moves 0.990 → 0.989. **K=30 stands.**

The surprise was *what* gets missed:

| Property of missed record | in misses | in matches | over-represented |
|---|---:|---:|---:|
| **no address at all** | **24.9%** | **3.2%** | **7.8×** |
| Indic script | 26.8% | 7.5% | 3.6× |
| name is a URL | 10.9% | 4.5% | 2.4× |

Records with a blank address are only ~3% of the data but a **quarter** of all
misses. That's our design flaw: we build the search text as
`name + " " + address`, so a record with no address contributes only its name,
has fewer rare words, and gets pushed out of the top 30 by records matching on
address noise.

Keep it in proportion though: ordinary Latin records are still **60.7%** of
misses. Most missed pairs are just hard, not a category bug. Fixing all three
categories perfectly is worth **+0.005** of ceiling.

### Two bugs we found in our own normalizer

Both silent, both caught by writing test cases:

```
"Orelee's Barbershop"  ->  "orelee south barbershop"
```
The apostrophe stripped to a bare `s`, which then hit our `s → south`
abbreviation rule. Every possessive English business name was corrupted.

```
"J. S. Motors"  ->  "j south motors"
```
Street/direction abbreviations were being applied to *names*, not just
addresses.

Fixed: possessives strip before tokenizing, and address abbreviations never
touch names.

## 9. Where things stand

**Done:** dataset verified, blocking built and measured, candidates cached,
docs and experiment log, merged onto `main`.

**Not done:** the matcher has never been trained. There is no F0.5 score and
no submission file. That is the single gap.

Ranked by value:

1. **Train the matcher.** Everything else is guesswork until a real score
   exists. First honest number arrives ~40 min into the run.
2. **Better decision rule.** One global threshold for all businesses is crude.
   Per-entity "expected F0.5" reasoning should beat it.
3. **Assignment constraint.** Each Source-2/3 record realistically belongs to
   at most one Source-1 business. Nothing enforces that yet.
4. **Blocking fixes** (address / transliteration / URL) — worth ≤ +0.005,
   so behind the above.
5. **Embeddings** (Krisha's branch) — a small neural model that understands
   *meaning*, not just letters. Aimed at the Indic-script gap. This is the one
   place a GPU genuinely helps.

## 10. Map of the code

```
src/config.py         all knobs in one place
src/normalize.py      text cleaning (two speeds: cheap for blocking, full for features)
src/blocking.py       TF-IDF + sparse matmul candidate generation
src/features.py       ~30 similarity numbers per pair
src/metrics.py        F0.5 exactly as the organisers compute it
src/data.py           TSV reading/writing, submission format enforcement
src/run_pipeline.py   glues it together end to end
src/analyze_blocking.py   the run-002 diagnostic

docs/master-plan/     the team's design + experiment plan (Neal)
docs/EXPERIMENTS.md   append-only log of every run — add submissions here
docs/DATA_BRIEF.md    measured facts about the data
docs/SUBMISSION.md    output format and the rules that get you rejected
context/              every original source document
```

## 11. How long does training take, and where should it run?

All figures below come from measurements on this laptop (10 physical cores,
25.5 GB RAM, RTX 3050), not estimates.

### Measured rates

```
blocking, test France     1,066 queries/sec   (259k queries vs 1.43M records)
blocking, test US           800 queries/sec   (663k queries vs 3.82M records)
blocking, test India        397 queries/sec   (810k queries vs 4.72M records)
feature computation     ~11,700 pairs/sec
```

**Blocking throughput is not a property of the machine.** It is a property of
the index being searched: bigger index, or denser posting lists for the
query's tokens, means each query touches more of the matrix. Notice that US
searches *fewer* records than India and is still twice as fast — Indian names
and addresses share more tokens.

An earlier draft of this section extrapolated the test phase from the
*training* run's rate (75–150 q/s) and predicted 255 minutes. The real run
took **52**. The estimate was not arithmetically wrong; it applied a rate
measured against one index to a different one. Quote q/s with the index it
came from, or don't quote it.

### Full run on the laptop, stage by stage

Measured end to end in run 007 (2026-09-26), except the two rows marked ⁺,
which the run did not reach:

| Stage | Work | Time |
|---|---|---:|
| Load train | 12.5M rows of TSV | 5 min |
| Train blocking | 150k queries | 8 min |
| Featurize train | 4.5M pairs | 2 min |
| Stage-1 LightGBM | 5 folds | 33 min |
| Stage-2 LightGBM | 5 folds | 28 min |
| Decision-layer tuning | 288 combinations | 7 min |
| Load test | 11.7M rows | 5 min |
| **Test blocking** | **1.73M queries** | **52 min** |
| **Featurize + score test**⁺ | **52M pairs, two passes** | **~80 min** |
| Predict + write⁺ | 52M rows | ~13 min |
| | **total** | **≈ 3.9 hours** |

The shape of the problem changed with the numbers. Blocking is no longer the
dominant cost — **model training is**, at 68 of the 145 minutes that were
actually measured. Test scoring is the other big block, and it is the one
stage still carrying an estimate rather than a measurement.

### The thing that decides everything: this workload is single-threaded CPU

| Component | Uses GPU? | Uses many cores? |
|---|---|---|
| TF-IDF / sparse matmul (scipy) | no | **no** |
| Top-K selection (Python loop) | no | **no** |
| String similarity (rapidfuzz) | no | **no** |
| LightGBM training | no | yes |

LightGBM is the parallel one, and now that blocking has come down to 52
minutes it is **68 minutes of the 234** — no longer a rounding error, and the
single biggest measured stage.

**A GPU still does nothing here.** Your RTX 3050 sits idle for the entire
run, and so would a T4 on Colab or SageMaker: LightGBM's GPU build helps
mainly on wide dense matrices, and 46 features over 4.5M rows is neither.
Renting GPU hardware for *this* pipeline buys you nothing. The one place a
GPU would earn its keep is the Indic-script gap — embedding ~12M records is
hours on CPU and 20–40 minutes on a T4 — which is Krisha's branch, not this
one.

### Comparison

| Where | Spec | Est. time | Cost | Verdict |
|---|---|---:|---:|---|
| **This laptop** | 10 cores, 25 GB | **~4 h** | free | Baseline (measured) |
| Colab (free) | 2 vCPU, 13 GB, T4 | **6–9 h** | free | **Worse.** Fewer cores, slower CPUs, GPU unused. **12-hour session cap with idle disconnects**, and 2.4 GB of data to upload each session |
| Colab Pro | 2–4 vCPU, T4/L4 | 5–7 h | ~₹1k/mo | Fixes the disconnect risk, not the speed |
| SageMaker `ml.g4dn.xlarge` | 4 vCPU, 16 GB, T4 | 5–6 h | ~$6 | No faster. GPU wasted, and 16 GB is *less* RAM than the laptop |
| SageMaker `ml.m5.4xlarge` | 16 vCPU, 64 GB | 2–3 h | ~$6.5 | The only one that genuinely wins: more cores for the pools, and RAM headroom that removes the chunking entirely |

**Renting hardware does not meaningfully help**, because the bottleneck is
single-thread Python, and cloud CPUs are not faster per-core than a modern
laptop.

### What actually helps: parallelism (a code change, not a hardware change)

Both slow stages are *embarrassingly parallel* — every chunk of queries is
independent. Running them across processes instead of one:

**This is already done.** `config.WORKERS` drives multiprocessing pools in
`ingest.py` and `features_v2.py`, and the 52-minute test blocking above is the
parallel number, not the serial one.

The remaining lever is RAM, not cores. On Windows the pools start with
`spawn`, so every worker is a fresh interpreter that re-imports pandas, numpy,
scipy and rapidfuzz — 250–400 MB resident before it does any work. That caps
us at 4 workers on a 25 GB box; asking for 15 killed run 005 during pool
startup. A 64 GB instance runs 12 and the cap disappears.

| Setup | Workers | Blocking | Score test | Total |
|---|---:|---:|---:|---:|
| Laptop, serial (the old estimate) | 1 | 255 min | 90 min | ~7 h |
| **Laptop today** | **4 (RAM-capped)** | **52 min** | ~80 min | **~4 h** |
| `ml.m5.4xlarge` | 12 (64 GB) | ~25 min | ~30 min | **~2.5 h** |

**Recommended order:** keep running on the laptop — it is the fastest machine
we have access to for this workload, and it is free. Move to `m5.4xlarge`
only if several experiments need to run at once.

### It's only slow once

Candidates are cached to disk after the first run. Later experiments —
different features, different threshold, different model — skip blocking
entirely:

```
first run        ~4 h      (train 2h11m + test blocking 52m + scoring)
every run after  ~80 min   (cached candidates and stats; train + score only)
```

Which is also why sharing that cache on S3 matters: one person pays for test
blocking, everyone else pulls the parquet. After run 007 that is 758 MB of
candidates plus a 153 MB stats pickle — together they are 56 minutes of
compute that nobody else has to spend.

```bash
./aws/s3.sh share-cache          # whoever ran it
./aws/s3.sh get-cache priyanshu  # everyone else
```

### Where a GPU *would* earn its place

One thing only: **embeddings** (Krisha's branch). A small neural model that
maps text to vectors capturing *meaning* rather than spelling — which is the
one thing that could close the Indic-script gap, since a Devanagari name and
its Latin form share zero characters.

Embedding ~12M records is genuinely GPU work: hours on CPU, perhaps 20–40
minutes on a T4. **That** is what the SageMaker quota is worth spending on.

## 12. Jargon

| Term | Meaning |
|---|---|
| **Entity resolution** | Deciding which records describe the same real thing |
| **Blocking** | Cheaply shrinking the candidate set before careful comparison |
| **Recall ceiling** | Best score achievable given what blocking kept |
| **TF-IDF** | Weighting words by rarity |
| **Sparse matrix** | Storing only non-zero values |
| **Precision / Recall** | Were your guesses right / did you find them all |
| **F0.5** | Combined score weighting precision 2× |
| **Macro average** | Score each business, then average |
| **Singleton** | A business with no match |
| **LightGBM** | Gradient-boosted decision trees |
| **OOF** | Out-of-fold — scored on data the model didn't train on |
| **GroupKFold** | Splitting so one entity never straddles folds |
| **Candidate pair** | A (Source-1, Source-2/3) pair worth scoring |
