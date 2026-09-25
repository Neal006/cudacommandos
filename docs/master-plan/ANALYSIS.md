# Analysis — measured facts, the question register, and the hypotheses we attacked

Every number in §1 was measured on 2026-09-25 from the released files
(`tools/eda/design_eda.py`). Numbers that are guesses are marked **(guess)**.
Numbers from `docs/DATA_BRIEF.md` (Priyanshu) were re-measured and agree.

---

## 1. Measured facts that drive the design

### 1.1 Structure of the ground truth

| # | Fact | Value | Design consequence |
|---|---|---|---|
| F1 | S2/S3 records linked to **more than one** S1 | **0** (max 1 of 7,638,365 links) | GT is a **partition**. Each S2/S3 record belongs to ≤1 S1 → one-to-many **assignment constraint** at decision time (HLD §4, L6). |
| F2 | S2 records linked to any S1 | 73.4% (26.6% are distractors) | Distractors are ~1 in 4 records → the matcher must say "no" often. |
| F3 | S3 records linked to any S1 | 74.6% (25.4% distractors) | Same. |
| F4 | Links per entity | mean 3.46, singletons 5.58% | Macro F0.5 is dominated by multi-link entities. |
| F5 | S2 links per entity | 1:789k 2:653k 3:334k 4:119k 5:24k (max 5) | Each source contains **several duplicates of the same business** → matches form tight clusters (peer-support features, L5). |
| F6 | S3 links per entity | 1:716k … 5:35k 6:2.8k (max 6) | Same. |
| F7 | Positive pairs whose countries differ | **0** of 7.64M | Blocking **within country** loses nothing on train. |
| F8 | Spearman(S1 id number, matched id number) | 0.00014 | IDs are random → **no ID-order leakage**. |
| F9 | S2 and S3 numeric id parts that collide | 26,801 | Never strip the `S2-`/`S3-` prefix — ids are only unique *with* the prefix. |
| F10 | Test ids also in train (S1, S2) | 0, 0 | Train and test are disjoint record sets. |
| F11 | Test S1 normalized names also seen in train S1 | 34.7% | Same generator and vocabulary (generic names repeat), but no record reuse. |
| F12 | Singleton share by country (train) | US 5.58%, India 5.59% | Singleton rate is not country-dependent → a big country gap in predicted singletons on test is a **bug signal** (mlguard `test_singleton_drift`). |

### 1.2 Countries and scale

| Split | S1 | S2 | S3 | US / India / France (S1) |
|---|---:|---:|---:|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | 60.0 / 40.0 / — |
| test | 1,732,544 | 4,887,273 | 5,082,316 | 38.3 / 46.8 / 15.0 |

Test S2 per country: US 1.87M, India 2.31M, France 0.70M, which is ~2.7–2.85 S2
records per S1 in *every* country, France included. **The French data comes from the same
generator** (strong evidence, not proof), so French match cardinality should look like
US/India: predicted links/entity for France far from ~3 means the pipeline is broken
for France.

### 1.3 Surface agreement on true pairs (1M-pair sample)

| Measure | US | India | All |
|---|---:|---:|---:|
| exact normalized name equal | | | 20.4% |
| exact normalized address equal | | | 8.3% |
| mean name token Jaccard | 0.68 | 0.54 | |
| positives sharing **no** name token | 8.2% | **23.9%** | 14.5% |
| positives sharing no address token | 4.8% | 4.0% | 4.5% |
| positives sharing a name OR an address token | | | **99.985%** |
| first house number equal | 83.0% | 73.1% | |
| any number shared | 78.1% | 82.7% | |
| other side has no number in address | 12.2% | 12.2% | |

**The blocking gap is not in the data.** Token-union blocking can reach ~99.98% pair recall.
The current ceiling (0.95 @K=30) comes from truncating at K and pruning at `max_df=0.01`.

### 1.4 Scripts: the hidden India problem

Share of **India** S2 names containing each script (test S2 is the same to 3 decimals):

| Devanagari | Telugu | Kannada | Tamil | Gujarati | Bengali | Malayalam | Oriya | Gurmukhi |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13.4% | 2.0% | 1.8% | 1.7% | 1.5% | 1.5% | 0.9% | 0.4% | 0.3% |

About 23% of India S2 names and 13% of India S3 names are in a **native Indic script**.
The state is often in native script too (`महाराष्ट्र`, `தமிழ்நாடு`, `ગુજરાત`, `കേരളം`).
S1 is 100% Latin. India is 47% of test, so **roughly 11% of test S2 and 6% of test S3 records
(≈8% of all test S2+S3 records) cannot share a name token with their S1** until they are
transliterated. This is the single biggest
recall and feature gap we found.

**Already tested (Hope #2's first test, done).** A deterministic transliteration +
consonant-skeleton key (`tools/eda/skeleton_prototype.py`, anyascii ISC) makes **94.4%** of
20,000 native-script India positive pairs share ≥1 core-name token with their S1 name (raw
tokens: ~0%); 43.1% get an identical skeleton. Transliteration moves from "idea" to P0.

### 1.5 Noise catalogue (seen in the data)

| Noise | Example (real rows) | Where |
|---|---|---|
| Case, spacing | `FOUNDATION EXCEL AGENCY PRIVATE  LIMITED` | S2 everywhere |
| Token order | `EVE'S ASSOCIATES HIGHLAND`, `private foundation excel agency limited` | S2/S3 |
| Legal suffix moved/dropped/added | `LLC Moncada…`, `Lee and Lawson` vs `… LLC`, `PVT. ASTOR TRADING LTD.` | all |
| Injected accents | `Léarning`, `Prívate`, `Fóod`, `Àmicale`, `Pàrenthese` | all countries |
| Typos | `Tetlecommunication`, `Lwson`, `Pbne Ave`, `Froaune` | all |
| Domain as name | `heassociates.com`, `deltatelecommunication.com`, `léo.com` | ~3% of S2/S3 |
| Handle as name | `@nizarmarine` | S2 |
| Junk prefix | `-- Holloway…`, `<< Team Ecole` | ~1.7–2.5% of S2/S3, 0.14% of S1 |
| Suffix noise words | `Lee and Lawson LLC Center`, `… Service` | S3 |
| Native script name/state | `राम मार्केटिंग प्राइवेट लिमिटेड`, `குளோபல் பிசினஸ்` | India S2/S3 |
| Address reorder | `IA, Iowa City, 1064 Newton Rd`, `Ohio, 991 Pierpont Avenue, Cleveland` | S1 and S3 |
| State abbrev ↔ full | `NY` ↔ `New York`, `MH` ↔ `Maharashtra` | S3 |
| House number corruption | `9914` → `914` / `991`, `599` → `599 1/2`, `No. 21` → `#C-21` | S2/S3 |
| Components missing | no house number (12%), empty address (~3%) | S2/S3 |
| Extra components | `PMB 9531`, `PO Box 3545`, `HN 753` | S2/S3 |
| Different locality level | `POTTERSVILLE` vs `Horicon`, `THANE` added | S2 |
| French abbreviations | `R.`/`R` = rue, `AV` = avenue, `ALLÉE`, `CITÉ`, `B` = bis | test France S2 |
| French admin level | S1 uses region (`Hauts-de-France`), S2 uses department (`Nord`, `Gironde`, `Loire-Atlantique`) | test France |
| Embedded CSV quotes | `""NIAGARA"" BUILDING` | 6 lines in train S2 |

### 1.6 Hard negatives

- 872,822 S1 entities (39.6%) share their normalized name with another S1 entity
  (`primary care group` ×253, `pediatric group` ×222, …).
- **117,160 S1 entities share their exact address with another S1 entity** (malls,
  medical buildings), so the address alone cannot decide.
- **80,757 S1 entities share name *and* locality with another S1 entity.** Only the
  street and house number separate them. These are the pairs where F0.5 gets lost.
- S1 has 0 exact (name, address) duplicates.

---

## 2. Question register

Every question we asked while designing, with the answer and how we know it.
Status: ✅ answered by data · 🧠 answered by reasoning/spec · 🧪 needs an experiment (id in
EXPERIMENT_PLAN.md) · 📨 needs the organizers (Google Form). **No question needed a GitHub
issue to Priyanshu**, because we measured everything that depends on the data ourselves.

### Data

| Q | Question | Answer | Status |
|---|---|---|---|
| Q1 | Can one S2/S3 record belong to several S1? | No, never (F1). | ✅ |
| Q2 | How many S2/S3 records are distractors? | ~26% (F2, F3). | ✅ |
| Q3 | Does a source contain duplicates of the same business? | Yes, up to 5 (S2) / 6 (S3) per entity (F5, F6). | ✅ |
| Q4 | Do matches ever cross countries? | Never in train (F7). | ✅ train / 📨 test |
| Q5 | Do IDs leak order or linkage? | No (F8). Prefix is needed for uniqueness (F9). | ✅ |
| Q6 | Is there train/test record overlap? | No (F10). Generic names repeat (F11). | ✅ |
| Q7 | How far apart are true pairs on the surface? | Only 20% have equal names; 24% of India pairs share no name token (§1.3). | ✅ |
| Q8 | What makes negatives hard? | Generic names, shared addresses, same name + same locality (§1.6). | ✅ |
| Q9 | Are singletons different in kind? | Same rate per country (F12); they look like normal records (sampled). | ✅ |
| Q10 | Which scripts must we handle? | 9 Indic scripts plus Latin with accents (§1.4). | ✅ |
| Q11 | Can we block on postal codes? | No. India has 0% PIN codes; US has a ZIP in ~10% of records (96% agree when both present); France ~0.4%. | ✅ |
| Q12 | Is France from the same generator? | S2 per S1 ratio matches (§1.2), same noise styles appear (accents, abbreviations, reorder). | ✅ (strong evidence) |
| Q13 | Will pandas split rows wrongly on quotes? | 6 quoted lines in train S2 → read with the quoting that `validate_submission.py` and pandas default agree on; assert row counts (EDGE_CASES D3). | ✅ |
| Q14 | Is the country label clean? | 0 nulls, exactly {US, India} train, {US, India, France} test. | ✅ |

### Blocking

| Q | Question | Answer | Status |
|---|---|---|---|
| B1 | Where does the 5% blocking recall loss live? | **Answered (E03, run 002).** Not in the data (99.985% share a token). Enrichment vs a covered baseline: empty address **7.8×** (24.9% of misses), all Indic combined **3.6×** (26.8%), domain **2.4×** (10.9%). Latin is still 60.7% of misses in absolute terms — the loss is *distributed*, not concentrated. | ✅ |
| B2 | Is within-country blocking safe? | Yes on train (F7). | ✅ |
| B3 | Is K=30 right? | Prefer per-pass K with a union over one global K. Look at ceiling per country, not global. | 🧪 E02, E04 |
| B4 | Should blocking be reverse too (S2→S1)? | Partition means every S2/S3 record needs its best S1. A reverse top-3 pass gives both recall and competition features. | 🧪 E05 |
| B5 | Can we afford more passes in time? | Current: 4–5 h on the laptop for test. Needs parallel chunks, or an AWS large-RAM box. | 🧪 E01 |

### Features / model

| Q | Question | Answer | Status |
|---|---|---|---|
| M1 | How do we make features work for France with zero labels? | Only country-agnostic similarity features plus country-keyed normalizers with a generic fallback. Measure transfer with leave-one-country-out. | 🧠 + 🧪 E10 |
| M2 | GBDT or transformer? | GBDT on engineered features first (fast, strong for ER). A cross-encoder only for the uncertain band if GPU time allows. | 🧠 + 🧪 E14 |
| M3 | Is a 150k S1 training sample enough? | For pairwise features yes. For **competition/assignment** features no: with random sampling 93% of candidates' true owners are absent (see H2). | 🧪 E06 |
| M4 | Global threshold or per-entity decision? | Per-entity expected-F0.5 decision on calibrated probabilities, compared against the global threshold. | 🧪 E08 |
| M5 | Does the assignment constraint help? | Should raise precision; its size is unknown. | 🧪 E07 |
| M6 | Is `country` allowed as a feature? | Only as equality (`country_eq`), never as a category. Country-keyed normalization tables with a generic fallback are allowed. | 🧠 (spec) |
| M7 | Is the source (S2 vs S3) a legitimate feature? | Yes. It is structural (different noise styles), not an identity leak. The numeric id part is banned. | 🧠 |

### Rules (📨 ask via the organizers' Google Form)

| Q | Question | Our default until answered |
|---|---|---|
| R1 | Which submission counts for the private leaderboard: best public, last, or a chosen one? | Keep the last upload = our best OOF model. |
| R2 | Are offline transliteration libraries (anyascii ISC, indic-transliteration MIT) and hand-written abbreviation tables "external data"? | We treat them as allowed: they are code and linguistic rules, not lookups of business identity. Document them. |
| R3 | Is transductive use of **unlabeled** test records allowed (IDF fit, self-supervised pairs)? | IDF over train+test text: yes. Synthetic training pairs built from test S1: **not used** unless confirmed. |
| R4 | Does the MIT/Apache ≤8B rule apply to every model in the pipeline (e.g. the blocking embedder) or only the final matcher? | Apply it to every model. |
| R5 | Are test matches guaranteed within-country, like train? | Assume yes; keep a cross-country diagnostic pass (E04). |

---

## 3. Weakest-hypothesis audit

*(There is no `/weakest-hypothesis` skill installed; this applies the method by hand.
We list what the current plan silently relies on, rank by likely damage × how untested
it is, and attack the weakest first.)*

| Rank | Hypothesis the current pipeline relies on | Evidence for | Why it may be false | Test | Kill criterion |
|---|---|---|---|---|---|
| **1** | **"A random 150k-S1 sample trains a matcher that behaves like the full test run."** | Pairwise features don't care who else is in the sample. | With 150k of 2.2M S1 sampled against the **full** S2/S3 index, ~93% of the S2/S3 records a sampled entity competes with have their true owner missing. Competition features (reverse rank, assignment) and the threshold are learned in a world with far fewer rival claimants than test. | E06: train on geo-cluster samples (whole localities) vs random; compare OOF on a fully blocked locality slice. | If the random-sample threshold/decision differs by more than 0.005 F0.5 on the full slice → switch to geo-cluster sampling. |
| **2** | "A global score threshold is the right decision rule." | Simple, tuned on OOF. | The metric is per-entity macro F0.5. The best cut depends on how many strong candidates an entity has. Singletons need P(no match) reasoning. | E08: expected-F0.5 per entity vs global threshold, same OOF scores. | If expected-F doesn't beat global by ≥0.003 → keep global (simpler). |
| **3** | "Blocking is not the bottleneck (ceiling 0.99)." | Measured global recall 0.95 @K30. | Measured on a 60/40 US/India mix, but test is 38/47/15. If the misses sit in India/Indic-script records, test recall is lower than the train sample shows. The ceiling formula also assumes a perfect matcher. | E03: recall ceiling split by country × script × domain-name × empty-address. | **FIRED (run 002): India 0.9132 < 0.93.** Translit + address pass are P0. But see the verdict below — the hypothesis survives. |
| **4** | "Features learned on US/India transfer to France." | Features are similarities, not vocabulary. | Normalizer rules for FR are untested; `r`→`rue` is excluded; departments vs regions; `bis`. | E10 leave-one-country-out; E11 FR normalizer unit tests on real test France rows; mlguard drift on France predicted cardinality. | France predicted links/entity outside 0.6–1.4× OOF → FR normalization is broken. |
| 5 | "Word-token TF-IDF is enough for blocking." | 99.985% share a token. | Share ≠ rank in top-K; generic tokens pruned by max_df; Indic scripts. | E02, E03. | — |
| 6 | "Train and test candidate distributions match." | Same generator. | Test is India-heavy (denser posting lists) → more candidates per entity pass K → rank features shift. | Compare rank/gap feature distributions train vs test (mlguard drift on features, E13). | KS statistic > 0.1 on a top feature. |

Hypotheses 1–3 get fixed before any model tuning.

### Verdict on hypothesis 3 (E03 / run 002, 2026-09-25)

Equal 15k-per-country sample against the full 10.3M index. Full entry in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) run 002.

| K | India | US | ALL |
|---:|---:|---:|---:|
| 10 | 0.8720 | 0.9511 | 0.9116 |
| 20 | 0.9007 | 0.9686 | 0.9347 |
| 30 | 0.9132 | 0.9735 | 0.9434 |

**The kill criterion fired.** India is 0.9132 at K=30, below the 0.93 line, and
lags US by 6pp. The concern was well founded: the 60/40 sample did mask it.

**But the hypothesis itself survives**, for a reason the criterion did not
anticipate. Blocking is *unsupervised* — the vocabulary is fit on the corpus,
not learned from labels — so France's absence from training costs it nothing,
and French names are Latin with accents the normalizer already folds. France
should track US, not India. Weighting the measured recalls by the real test
mix:

```
0.383(0.9735) + 0.468(0.9132) + 0.150(~0.97) ≈ 0.9457
F0.5 ceiling: 0.9896 (run 001)  ->  0.9887 (test-weighted)
```

A **0.001** move. `TOP_K = 30` stands and blocking is still not the bottleneck.

**E03's own success bar — "a slice holding > 40% of misses" — was not met by
any actionable slice.** Empty address 24.9%, all Indic combined 26.8%, domain
10.9%; latin is 60.7%. The loss is distributed. What makes the first three
worth fixing is *enrichment* against a covered baseline (7.8× / 3.6× / 2.4×),
not their share.

**Size of the prize.** If all three fixes landed perfectly and without overlap:
+1.41pp (address) +1.52pp (Indic) +0.62pp (domain) → ~0.97 recall → ceiling
0.9939, against 0.9887 today. **+0.005 total.**

So translit + address pass are P0 *within blocking*, as the criterion says —
but the whole blocking backlog is capped at half a point of ceiling while the
matcher remains unmeasured. Order of work: get a real OOF F0.5 first, then
revisit these if the matcher's own per-country breakdown shows India dragging.

One correction to the B1 guess: the guess named Indic-script and domain-name
records. **Empty address is the strongest single signal of the three** (7.8×
vs 3.6×), and it is a design flaw rather than a data property — `_blob` is
`core_name + " " + core_address`, so a record with no address contributes only
its name, carries fewer rare tokens, and loses top-K slots to records matching
on address noise. ~3% of S2/S3 lack an address; they are a quarter of misses.

---

## 4. Pandora's Box: opening the problem up

### 🔓 Lifting the lid
**Problem.** For 1.73M reference businesses, pick out which of ~10M noisy records from
two other sources are the same business. Precision counts double, and 15% of the test is a
country we never saw labelled.
**Goal.** Macro F0.5 ≥ 0.95 on OOF (guess at a strong target), with no country slice
more than 0.03 below the aggregate, and the full test run finishing in under 6 h.
**Hidden assumptions.**
1. "We decide each (S1, candidate) pair independently." But GT is a partition, and the
   true matches of an entity are duplicates of each other.
2. "Matching is text similarity." But the structure (who else claims this record, how many
   S2 copies agree) carries signal the text doesn't.
3. "France must be learned." It only has to be *normalized*; the similarity function is
   language-agnostic.
4. "Native-script names need a multilingual model." A deterministic transliteration plus
   a phonetic skeleton may close most of that gap.
5. "Blocking and matching are separate stages." Blocking pass membership and ranks are
   strong features.
**Hard limits.** 25 GB RAM / 16 threads locally (AWS available). About 2 days left. 5
submissions a day. No external lookups. Models MIT/Apache ≤8B. Output format fixed.

### 📦 Opening the box
1. `[Break an assumption]` **Assignment decoding.** Treat the output as a bipartite
   assignment: every S2/S3 record goes to at most one S1 (its best claimant), then choose
   each entity's set by expected F0.5.
2. `[Borrow from another field: population genetics / record linkage]` **Fellegi–Sunter
   EM weights** fitted *unsupervised on test France* candidates, used as extra features or
   as a France fallback.
3. `[Borrow from nature: immune-system clonal consensus]` **Peer support.** A true S2
   copy agrees with the *other* true copies. Score each candidate by its similarity to the
   entity's other strong candidates.
4. `[Inversion]` "How do we over-merge as badly as possible?" Match on generic names
   (`primary care group`) with shared addresses (medical buildings). Reversed: add a
   name-IDF feature and a co-location flag (how many S1s share this address) so the model
   leans on house number and street exactly there.
5. `[First principles]` **Transliterate, then compare consonant skeletons.** Indic → Latin
   (anyascii / indic-transliteration), then fold voicing (g→k, b→p, d→t) and drop vowels.
   `Global Business` and `குளோபல் பிசினஸ்` (`kulopal picinas`) collide on `klpl psns`.
6. `[Scale shift ×1000 cheaper]` **Rust / rapidfuzz `cpdist` feature engine.**
   Featurizing ~100M pairs in Python loops is the real time risk. Vectorized C++/Rust scorers
   make it minutes.
7. `[Remove a limit: unlimited labels for France]` **Self-supervised French pairs.** Apply
   the noise operators we observed (case, accent injection, `rue`→`R.`, token shuffle,
   suffix drop, typo) to French S1 records and fine-tune on them. *Strange; needs the R3
   ruling.*
8. `[Combine two things]` **Blocking = TF-IDF ∪ dense ANN.** A multilingual-e5-small
   (MIT) embedding of `name | street` with FAISS HNSW per country recovers pairs that share
   no surviving token.
9. `[Change the user: an auditor]` **Explainable decisions.** Log the top-3 features per
   decision for the methodology document and error analysis.
10. `[Add a limit: 1 hour for all of test]` **Two-tier cascade.** A cheap LR on 8 features
    discards 80% of candidates; the GBDT scores the rest. *Strange: deliberately weaker
    first stage.*

### ⚠️ What escaped

| Idea | Status | Biggest risk | Value |
|---|---|---|---|
| 1 Assignment + expected-F decoding | Grounded (bipartite assignment; F-measure-optimal decoding is standard decision theory) | Calibration errors mislead expected-F; the independence assumption between candidates | **High** |
| 2 Fellegi–Sunter EM on France | Grounded (classical record-linkage model) | Adds complexity and may duplicate GBDT signal | Medium |
| 3 Peer-support features | Grounded | Needs stage-1 OOF scores → a stacking leak if done in-fold | **High** (guess) |
| 4 Name-IDF + co-location features | Grounded | Small; none | **High** for hard negatives |
| 5 Transliteration + skeleton | Grounded (deterministic mapping) | Tamil script loses voicing, so the skeleton must fold it | **High** for India |
| 6 Vectorized feature engine | Grounded (`rapidfuzz.process.cpdist`, multi-threaded) | Engineering time | **High** (enabler) |
| 7 Synthetic French pairs | Stretch | Rule ambiguity (R3); synthetic ≠ real noise | Medium |
| 8 Dense ANN blocking | Grounded | Time to embed 10M records on CPU (~hours, guess); needs GPU/AWS | Medium |
| 9 Explanations | Grounded | None | Low for score, high for the doc |
| 10 Cascade | Grounded | Recall lost at stage 1 | Medium (only if time-bound) |

### 🌟 Hope that remained
1. **Assignment + expected-F0.5 decoding (idea 1).** It won because it attacks F0.5 directly
   and uses the one structural fact (partition) nobody exploits yet, with zero new features.
   *First test:* on existing OOF scores, compare global threshold → + assignment → +
   expected-F (E07, E08), 30 minutes of compute. *Success:* ≥ +0.005 OOF. *Kill:* < +0.001.
2. **Transliteration + address pass + skeleton features (idea 5).** It won because the
   biggest measured gap (24% of India pairs share no name token) sits on 47% of test.
   *First test:* E03 recall split by script before/after translit on a 50k India sample.
   *Success:* Indic-script recall +10 points. *Kill:* < +2 points (then the address pass alone
   is enough).
3. **Peer-support + competition stage-2 (ideas 3 + 4).** It won because it targets the
   hard negatives (80k same-name-same-locality). *First test:* E09 stage-2 GBDT on OOF
   stage-1 scores. *Success:* ≥ +0.003. *Kill:* no gain on the hard-negative slice.

### 🔒 Closing the lid
Keep Priyanshu's blocking + GBDT backbone and add three things in this order: (1) the
assignment + expected-F decision layer on existing OOF scores, (2) transliteration with an
address-only blocking pass for India, and (3) a vectorized feature engine so the full test
run fits the window. Then add stage-2 peer/competition features. First action: run E01–E03
(per-slice recall) and E07–E08 (decoding) tonight, because they need no new model. The main
risk to watch is France. Every run is gated by mlguard's per-country cardinality drift
check before it reaches the leaderboard.
