# The France diagnostic — what we found and what it actually proves

Neal — this is the write-up of the France investigation. Short version: we went
in expecting France to be our problem, and the measurements say it isn't. But
the evidence is weaker than the headline sounds, and your PR #7 comment caught
part of why, so I've tried to be careful about which bits are load-bearing.

## Why we were worried about France

France is 15% of the test set and has **zero training labels**. Every number
we produce offline — OOF, per-country F0.5, the decision-layer sweep — is
computed on US and India only. So a whole country was invisible to us, and
when submission 001 came back at 0.943 against an offline 0.9532, France was
the obvious suspect.

The arithmetic seemed to confirm it. Using the real country shares (France
14.97%, India 46.75%, US 38.27%) and assuming US and India transfer at their
OOF values:

```
0.4675 x 0.9399  +  0.3827 x 0.9620  +  0.1497 x France  =  0.943
                                              France     ~=  0.904
```

France at 0.904 would be six points below the US. That's a big, actionable
gap, and there was a plan built around it: French pseudo-labels, a
France-specific calibrator, raising France's `miss` so it predicts more links.

Before spending a day on that, we measured.

## How you can measure a country with no labels

You can't compute F0.5 for France. But you can compare France against US and
India on things that don't need labels, and ask whether France *looks* like a
country the pipeline is handling badly.

Three separate questions, deliberately kept apart because they point at
different fixes:

1. **Is blocking failing France?** How often do we find no candidates at all,
   and how good is the best candidate we do find? Needs only the candidate
   cache — no model involved.
2. **Is the matcher failing France?** How confident is the model about its
   best candidate? Needs the cached per-pair scores, still no labels.
3. **What did we actually ship for France?** Links per entity, singleton rate.

The tool is `src/diagnose_country.py`. It streams the aggregates through
polars so it can run without fighting a training job for memory.

## What came back

```
=== BLOCKING (no model involved) ===
country     entities  orphan%  cand/ent   top_sim p10     p50     p90
India        809,986  0.002%     30.00        0.8580  0.9703  1.0000
US           663,106  0.000%     30.00        0.8027  0.9637  1.0000
France       259,452  0.013%     30.00        0.9128  1.0000  1.0000

=== MATCHER (cached scores, still no labels) ===
country     top_p p10     p50     p90  conf/ent  no-conf%
India          0.9808  0.9999  0.9999      3.12    6.80%
US             0.9939  0.9999  0.9999      3.30    5.90%
France         0.9985  0.9999  0.9999      3.37    5.12%

=== SHIPPED OUTPUT ===
country     links/ent  singleton%
India           3.085      6.81%
US              3.267      5.87%
France          3.266      5.35%
```

France is at or near the top of every column. Its blocking similarity is the
highest, its model confidence is the highest, and its output distribution
matches the US almost exactly. India is the country that looks conservative —
fewest links per entity, most singletons, lowest blocking similarity.

That was the opposite of what we expected, and it immediately cancelled the
France work. Raising France's `miss` would have made it predict *more* links,
and France is already predicting the most of the three. We'd have pushed in
the wrong direction.

## Your critique, and how much of it survives

You raised two objections in PR #7. One is wrong and one is right, and the
right one matters.

**"`top_sim` is TF-IDF over a per-country index, so similarities aren't
comparable across countries."**

This one doesn't hold. The vectorizer is fit **once, globally**, before any
country partitioning — `blocking.py:176` builds it over all of S2+S3, and the
per-country split happens at line 180. Every country shares the same
vocabulary and the same IDF weights, so the cosine values are directly
comparable. Only the candidate *pool* is partitioned.

And the pool size argues the other way: France searches 1.43M records against
India's 4.72M. A smaller pool gives you fewer chances to find a very similar
record, so France should show *lower* max similarity. It shows the highest
anyway. That makes the finding stronger, not weaker.

**"A shifted matcher can be confidently wrong, so high confidence isn't
evidence of correctness."**

This one is correct and it's the important one. Confidence is not accuracy.
A model that has never seen French data could be systematically,
confidently wrong about it — high `top_p` on the wrong candidate looks
identical to high `top_p` on the right one when you have no labels.

So the honest version of the finding is narrower than the headline:

- **What we can say:** France is not failing in any way that leaves a visible
  trace. Blocking finds candidates for it, the model is not hesitant about
  them, and the output has a normal shape. There is no *sign* of breakage.
- **What we cannot say:** that France is scoring well. Nothing here measures
  correctness, and it can't without labels.

## The part that doesn't depend on any of that

There's a second argument that stands entirely on its own, and it's the one I'd
lean on.

Suppose France is fine — suppose it scores exactly as well as the US, 0.9620.
Then the expected total is:

```
0.4675 x 0.9399  +  0.3827 x 0.9620  +  0.1497 x 0.9620  =  0.9517
actual leaderboard                                          0.943
unexplained                                                 0.009
```

**Even a perfect France leaves 0.009 unaccounted for.** So France cannot be
the whole story regardless of what the tables above say. Something is costing
us roughly a point across *all* countries.

That's what redirected the work toward contention: stage 2's `n_claims` and
`n_strong_claims` are raw counts of how many entities compete for a record, and
they're 1.96 in training against 5.55 at test. `n_strong_claims` is the second
most important feature in stage 2. That's a uniform, all-country effect, which
is the shape the residual actually has.

And your PR #7 found a second one I'd missed — the calibrator is fitted on
single-model OOF scores and applied to five-model-averaged test scores. Also
uniform, also invisible to CV. Between them those two explain the residual far
better than a France-specific story does.

## What would settle France properly

Nothing we can do today, but for the record:

- **Leave-one-country-out.** Train on US only, evaluate on India, and vice
  versa. That measures how well the matcher transfers to a country it has
  never seen, which is exactly France's situation. It's the honest proxy and
  it's never been run.
- **Per-country leaderboard feedback**, if the organisers ever expose it,
  ends the argument in one submission.

## Bottom line

France was a reasonable hypothesis and it's probably not our problem. The
label-free evidence says it isn't visibly broken, the arithmetic says it can't
be the whole gap, and the France-specific fixes we had planned would most
likely have hurt.

But "probably not our problem" is where the evidence lands — not "fine". If
someone comes back later with per-country truth and France is at 0.91, nothing
here would have caught it.

---

Tool: `src/diagnose_country.py`
Run: `python src/diagnose_country.py --scores interim/testp_<key>.npy --matching <output tsv>`
Context: `docs/CHASING99.md`
