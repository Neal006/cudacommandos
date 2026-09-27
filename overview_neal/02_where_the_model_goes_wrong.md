# Where the model goes wrong

Neal — this is the error picture as best we can measure it. I've tried to be
clear about which parts are **measured** and which are **inferred**, because
a couple of the interesting bits are the second kind.

The short version: we score 0.953 on the leaderboard against a hard ceiling of
about 0.980, and the losses split into four buckets of roughly known size.

---

## The budget

Start from what's actually achievable. Blocking recall is 0.9498, and a
perfect matcher given recall R scores `1.25R/(0.25+R)`. So:

```
perfect matcher, our candidates        0.9903  offline
minus the measured offline->LB gap     -0.010
maximum possible leaderboard score     ~0.980
where we actually are                   0.953
```

That gap of 0.027 breaks down roughly like this:

| Bucket | Size | Measured? |
|---|---|---|
| The offline→leaderboard gap | ~0.010 | measured 3x, cause found |
| Blocking never offers the right record | ~0.010 of ceiling | measured |
| Matcher picks wrong among candidates it has | ~0.007 | inferred (the remainder) |
| Entities with no candidates at all | ~0.00003 | measured, negligible |

---

## 1. Blocking never shows the model the right record

**5.02% of true pairs are never generated as candidates.** No matter how good
the matcher gets, those are gone. This is the single biggest *structural*
loss, and it's the one no amount of model work can touch.

We profiled what gets missed (experiment 002), and two things stand out:

- **Records with an empty address are over-represented among misses by 7.8x.**
- **Native-script (Indic) names are over-represented by 3.6x.**

The empty-address one is our own doing, and it's worth understanding because
it's the clearest fixable defect in the pipeline. Blocking searches on a
single blob:

```python
_blob = normalized_name + " " + normalized_address
```

If the address is empty, that blob is just the name — but it's competing in a
TF-IDF space where every other record has name *and* address tokens. The
similarity gets diluted, the record sinks below the top-30 cut, and it's never
seen again.

**Here's the detail I find most telling.** The feature set has a flag for
exactly this case, `addr_empty_any`, and the trained model gives it an
importance of **exactly 0.0**. Same for `country_eq`, `skel_eq` and
`n_candidates` — four dead features.

That isn't the model being stupid. It's that **by the time a pair reaches the
model, the damage is already done.** The empty-address records that matter are
the ones that never became candidates. The flag only fires on the survivors,
where it carries no information. The error happens upstream, in retrieval,
where no feature can reach it.

If anyone picks up one thing from this document, it should be that: a
name-only retrieval pass for empty-address records is the highest-value
retrieval fix we've identified, and it's never been built.

## 2. Hard negatives the matcher genuinely cannot separate

**80,000 Source-1 businesses share both a name and a locality with another
business.** Franchise branches, chains, and genuinely distinct businesses with
generic names on the same road.

For these, string similarity is not just weak — it's *uninformative*. Two
different "City Medical Store" on the same street are textually identical.
There is no amount of Levenshtein that separates them.

What actually does the work here is stage 2's competition features, and the
importances confirm it. In stage 2, after `p1` itself (14.35M):

```
n_strong_claims   2,521,391    <- second most important feature in the model
claim_gap           317,796
sum_p1              139,879
```

`n_strong_claims` is "how many *other* Source-1 entities also want this
record, strongly". That's the model noticing a tug-of-war. Ground truth is a
partition — each record belongs to at most one entity — so a record that two
entities both want confidently is a record where at least one of them is
wrong.

**This is the most important thing the model learned, and it isn't about text
at all.** It's about structure.

Related: **only 20% of true pairs share a normalized name.** The other 80%
need fuzzy matching, transliteration or address evidence. And **26% of S2/S3
records match nothing** — they're pure distractors, so the model spends most
of its capacity learning to say "no".

## 3. We deliberately under-predict, and that's correct

We output **3.17 links per entity** against a true average of **3.46**. That
looks like a bug and isn't.

F0.5 weights precision twice as heavily as recall. For an entity, adding a
marginal link that's 50/50 costs more in expected precision than it gains in
expected recall. The expected-F decoder works this out per entity and declines
those links on purpose.

I tested pushing it the other way — the decision sweep covers 288 combinations
including much more permissive ones — and the score goes **down**. Chasing
3.46 would lose points.

The one asymmetry worth knowing: under F0.5, an entity that *has* matches but
gets predicted empty scores **0**, not partial credit. So being too
conservative on a non-singleton is expensive. We predict 6.25% singletons
against a true rate of 5.6%, so we're slightly over-calling singletons — but
only slightly.

## 4. India is our weakest country by a clear margin

```
US      0.9717
India   0.9491      <- 2.3 points behind
France  (no labels)
```

Three compounding reasons:

- **Blocking is worse there.** India's top-candidate similarity distribution
  is below the US's, and it's the country where the native-script and
  empty-address misses concentrate.
- **No postal codes.** India has no PIN codes in this data, and US ZIPs appear
  in only ~10% of rows. So address matching has no anchor — it's pure token
  overlap on free text.
- **We under-predict there most.** India gets 3.08 links per entity against
  US's 3.27, and has the highest singleton rate (6.81% vs 5.87%).

The band reranker helped India more than anywhere else (+0.012 from 0.9369 to
0.9491), which fits: it's the country where the hard cases live.

## 5. The offline→leaderboard gap — two causes, both found

This is the bucket we understood latest and it's worth its own section,
because it's not really "the model being wrong" — it's *us measuring wrong*.

The gap has been remarkably stable:

```
submission 001    OOF 0.9532  ->  LB 0.943    -0.0102
submission 002    OOF 0.9607  ->  LB 0.951    -0.0097
submission 003    OOF 0.9626  ->  LB 0.953    -0.0096
```

Three very different models, agreeing within 0.0006. That consistency is the
tell: it's systematic, not noise.

**Cause one — contention.** Stage 2's competition features are raw counts of
how many entities compete for a record:

```
mean n_claims:   train 150k = 1.96      TEST = 5.55
```

We train with 150k entities competing and infer with 1.73M. `n_strong_claims`
— the second most important feature in the whole model — literally means
something different at inference than it did in training. And
cross-validation can't see it, because the OOF split carries the same wrong
contention as the training data.

That's what `run_v4` fixes: compute those counts over the full 2.2M-entity
frame. Measured on the running job, contention goes from 1.96 to **6.717**,
which brackets test's 5.55 instead of sitting 2.8x below it.

**Cause two — calibration (your catch, PR #7).** Every OOF score comes from
*one* fold model; every test score is the *mean of five*. Averaging shrinks
the variance, so the isotonic calibrator is fitted on one distribution and
applied to another. The expected-F decoder then converts slightly-wrong
probabilities into wrong keep/drop decisions.

Same property as contention: uniform across countries, invisible to CV. I'd
missed it entirely.

Between them, these two explain the residual far better than any
country-specific story — which is also why the France theory collapsed (see
`01_france_diagnostic.md`).

---

## What we haven't done

Being straight about the gaps in this analysis:

- **No per-case FP/FN inspection.** We have aggregate error rates and
  importances, but nobody has sat down with a sample of false positives and
  read them. That would probably surface patterns these numbers can't.
- **No leave-one-country-out.** Train on US, test on India and vice versa.
  It's the honest proxy for "how well does this transfer to a country it's
  never seen", which is France's exact situation. Never run.
- **LightGBM's hyperparameters are untouched.** All at defaults. That's a real
  lever and we simply ran out of hours.

## If you want to attack one thing

Ranked by what the numbers support:

1. **A name-only retrieval pass for empty-address records.** 7.8x
   over-represented in misses, and the dead `addr_empty_any` feature shows why
   no model change can fix it.
2. **Record-to-record sibling linking.** Sources carry 5–6 duplicate copies of
   one business. If we match one copy confidently, its near-duplicates should
   follow. Attacks both the 5% blocking loss and the 3.17-vs-3.46 shortfall,
   and it's what won the closest public analogue (Foursquare 2022).
3. **Your holdout calibration fix.** Cheapest of the three and it targets a
   measured 0.010.
