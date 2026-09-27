# Asking for help: Amazon ML Challenge 2026, Business Entity Resolution

Team **cudacommandos**. Written 2026-09-27 for someone who has never seen this challenge.
Reading time is about 15 minutes. The questions we want your view on are in section 9.

---

## 1. The problem in one paragraph

We get three lists of businesses (name, address, country). **Source 1** is a clean list where every business appears once. **Source 2** and **Source 3** are messy lists from other systems, where the same business can appear several times with typos, abbreviations, different scripts, or no address at all. For every business in Source 1 we must list all the Source 2 and Source 3 records that are the same real-world business. Some Source 1 businesses have no copies at all, and then the right answer is an empty list.

A small made-up example of one business:

| Source | Name | Address |
|---|---|---|
| S1 | Sri Ram Medicals Pvt Ltd | 12 MG Road, Bengaluru |
| S2 | Sri Ram Medicals | 12 M.G. Rd, Bangalore |
| S2 | श्री राम मेडिकल्स | 12 MG Road |
| S3 | SRI RAM MEDICALS PRIVATE LIMITED | (empty) |

## 2. How we are scored

- For each Source 1 business we compute **F0.5** between our predicted list and the true list. F0.5 weights precision twice as much as recall, so a wrong link hurts more than a missed one.
- If the true list is empty, we get 1.0 only if we also predict nothing, otherwise 0.
- The final score is the plain average of that number over every Source 1 business (**macro F0.5**). A business with 1 copy counts as much as one with 8.
- We must also submit the **candidate list** we considered for each business, and every predicted match must be inside it. That forces a two-step design: first shortlist, then decide.
- Rules: 5 leaderboard submissions per day, no external data or web lookups, and only open models (MIT or Apache license, 8B parameters or smaller).

## 3. The data, with real numbers

| File | Rows |
|---|---:|
| train Source 1 | 2,206,821 |
| train Source 2 | 5,034,616 |
| train Source 3 | 5,285,603 |
| train ground truth | 2,206,821 (one row per Source 1 business) |
| test Source 1 | 1,732,544 |
| test Source 2 | 4,887,273 |
| test Source 3 | 5,082,316 |

Things that shape every decision:

1. **Scale.** The test set is 1.73 million businesses against 9.97 million records. Comparing everything with everything is about 10^13 pairs, which is impossible. We must shortlist.
2. **Most businesses do have copies.** Only 5.6% have none. The typical business has 2 to 5 copies (mean 3.46). Being too cautious loses more than it saves.
3. **The country mix shifts between train and test.**

   | | US | India | France |
   |---|---:|---:|---:|
   | Train | 60% | 40% | 0% |
   | Test | 38% | 47% | 15% |

   France never appears in training, yet it is about 260,000 test businesses. India, our hardest country, becomes the majority.
4. **India is hard.** About 23% of Indian Source 2 names are written in native scripts (Devanagari, Tamil and 7 others). Transliterating them to Latin letters and building a consonant "skeleton" key lets 94% of them share a token with the English version.
5. **Messy fields.** About 3% of Source 2/3 records have no address. India has no postal codes and US ZIP codes show up only about 10% of the time, so we never rely on postal codes.
6. **Useful structure.** No messy record ever belongs to two Source 1 businesses (the answer is a partition). There are zero cross-country matches. Each source repeats a business up to 5 or 6 times.

## 4. What we built (the architecture)

```
raw TSV files
   |
   v
[1] Clean and normalise      lower case, strip accents, expand "Rd"->"road",
                             drop legal words (Pvt Ltd, SARL, LLC), transliterate
                             Indian scripts, build a consonant skeleton key
   |
   v
[2] Shortlist (blocking)     word TF-IDF per country, keep the top 30 records
                             per Source 1 business
    + recall passes (new)    pull in exact-name "siblings" of good candidates,
                             and direct lookups for rare names
   |
   v
[3] Stage 1 model            LightGBM on about 40 pair features: fuzzy name and
                             address similarity, token overlap weighted by
                             rarity, skeleton match, length and empty flags
   |
   v
[4] Stage 2 model            LightGBM again, now with context: how many Source 1
                             businesses compete for this record, the gap to the
                             best rival, the scores of this record's siblings
   |
   v
[5] Reranker (optional)      a small multilingual cross-encoder (e5-small,
                             fine-tuned) rescores only the uncertain middle band
   |
   v
[6] Calibrate and decide     turn scores into honest probabilities, then per
                             business pick the list size that maximises the
                             EXPECTED F0.5, instead of one global cutoff
   |
   v
[7] Write both files and run the official validator
```

Why it is shaped like this:

- **Shortlisting decides the ceiling.** If the true match is not in the top 30, no model can recover it. With a perfect matcher our shortlist tops out around 0.990.
- **Word TF-IDF, not character n-grams, for shortlisting.** Character n-grams explode memory on 10 million records. We use them as features on the shortlist instead.
- **Two model stages.** Stage 1 judges each pair alone. Stage 2 sees the neighbourhood: if a record is also a strong match for another business, that is evidence against this one.
- **The decision layer is built for the metric.** Because of the empty-list rule and F0.5, the best number of links differs per business. We compute it exactly from the calibrated probabilities.
- **Country is treated as an unknown label**, never as a fixed US/India list, so France goes through the same general rules plus French lookup tables (SARL, SAS, Rue, Bd).

## 5. What we have achieved

Offline scores come from 5-fold cross validation grouped by business, so no business is on both the train and test side of a fold.

| Step | Offline macro F0.5 |
|---|---:|
| First simple pipeline | 0.927 |
| New vectorised features (stage 1) | 0.950 |
| + stage 2 context model | 0.9526 |
| + decision layer | 0.9532 |
| + reranker (30k sample only, leak check pending) | about 0.960 |

- Per country at 150k businesses: **US 0.962, India 0.940**. India is where the loss is.
- **Leaderboard: 0.953.** Offline scores run about 0.010 above the leaderboard.
- Engineering wins: test shortlisting went from an estimated 4 hours to 52 minutes, and test scoring now runs in chunks at about 1.5 GB peak memory instead of about 19 GB.

## 6. What we are doing right now

**a) Recovering missed matches.** Shortlist recall is 0.9496, so 5% of true matches never reach the model. We studied 5,225 missed pairs and found they are mostly easy:

- 22% have exactly the same cleaned name as the Source 1 business
- 36% have the same transliterated skeleton
- 25% have an empty address

They lose in TF-IDF because common words and address tokens outvote the name. So we added two cheap passes:

- **Siblings:** take a business's top 3 candidates and add every record in the same country with the same name key. Names shared by more than 50 records are skipped ("Medical Store" would flood us).
- **Direct lookup:** match the Source 1 name key directly, only when 10 or fewer records share it.

| On a 30k sample | Recall | Extra candidates per business | Best possible F0.5 |
|---|---:|---:|---:|
| Before | 0.9496 | 0 | 0.9897 |
| After | 0.9623 | 8.2 | 0.9922 |

We also added "gang" features to stage 2: how many of a business's candidates share this record's name, and the best score among them.

First result on a 90k local check: **stage 1 went from 0.9490 to 0.9516**. Stage 2 and the final decision are still running. If stage 2 beats 0.9511 on the same sample, we run it at full size.

**b) Fixing a train/test mismatch in the context features.** We train on a 150k sample of businesses but score all 1.73 million at test time. In the sample each record competes for about 2 businesses, in test about 5.5. So the competition features look very different at test time. We now compute them on the full candidate set in both cases, and we fit the calibration and decision rule on a separate holdout scored exactly like test.

**c) Safety checks.** Score files carry a fingerprint so they can never be applied to a reordered candidate list. A model trained on the new shortlist refuses to score the old one. The reranker is guarded against seeing businesses from the training sample.

## 7. Compute and time we have

- Training box: AWS SageMaker ml.m7i.48xlarge, 192 CPU cores, 768 GB RAM, no GPU. We can run two in parallel.
- Budget: about 4 hours per run, covering training and scoring the full test set.
- Local laptop: about 25 GB RAM and a 4 GB RTX 3050, used for small checks only.

## 8. What we already tried and parked

- **Dense embedding nearest neighbours for shortlisting.** The measured ceiling gain was below +0.002. Not worth the cost at 10 million records.
- **Top 50 instead of top 30 candidates.** +0.002 ceiling for 35 million extra pairs.
- **A larger multilingual reranker (mmBERT based).** Lost the head-to-head against the small e5 reranker and was much slower.
- **Tuning the assignment rule** (giving each disputed record to one business). Worth less than +0.0001, so we stopped.
- **A Rust watchdog for training** was built and later removed to keep the stack in one language.

## 9. Where we would value your view

We are at 0.953 on the leaderboard and want 0.97 or better. Our honest estimate for the current plan is 0.958 to 0.965.

1. **Is the shortlist or the matcher the bigger lever?** The shortlist caps us near 0.99 offline, yet we lose almost 4 points below that inside the matcher, mostly in India. Where would you put the next day of effort?
2. **The offline to leaderboard gap of about 0.010.** We think it comes from France (never seen in training) and the country mix shift. Is there a better way to validate for an unseen country than holding out one of our two training countries?
3. **India.** Native-script names, no postcodes, very generic names ("Sri Sai Traders"). Any feature or pairing trick you have seen work for Indian business names?
4. **Graph thinking.** Copies of one business form clusters inside each source. We use this only lightly (siblings and gang features). Would you go further, for example clustering Source 2/3 records first and then matching clusters instead of single records?
5. **Using the CPU hours better.** Would you spend them on more training data (all 2.2M businesses instead of 150k), more boosting rounds, or a CPU-friendly cross-encoder over the whole shortlist?

## 10. Glossary

- **Blocking or shortlisting:** cheaply picking a few candidates per business before any expensive comparison.
- **Recall of the shortlist:** the share of true matches that make it into the shortlist.
- **F0.5:** a score mixing precision and recall that counts precision twice as much.
- **Macro average:** every business counts equally, whatever its size.
- **LightGBM:** a fast gradient boosted tree model, standard for tabular features.
- **Cross-encoder or reranker:** a small language model that reads both records together and scores the pair.
- **Calibration:** adjusting scores so that a 0.8 really means about 80% of such pairs are true.
- **Expected F decision:** for each business, choosing how many top candidates to keep so the expected F0.5 is highest.
- **Out-of-fold (OOF):** scores predicted by a model that never saw that business in training.
- **Transliteration skeleton:** convert any script to Latin letters, then keep a simplified consonant pattern so "Medicals", "Medikals" and "मेडिकल्स" collide.

---

Code: https://github.com/Neal006/amazonml/tree/nealultraprohopeso (start from `context_hopeso.md` and `src/run_v4.py`).
