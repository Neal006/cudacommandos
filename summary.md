## Approach

Added two embedding features — sentence-transformer cosine similarity on name
and address (`all-MiniLM-L6-v2`) — then tried blending LightGBM with Logistic
Regression 50/50 instead of plain LightGBM.

## Result (OOF macro F₀.₅, 2k sample)

| Setup | Score |
|---|---|
| LightGBM + embedding features | **0.9083** |
| LightGBM + LR blend + embedding features | 0.8959 |

Ensemble hurt. Dropped the LR, kept the embeddings. Current: **0.9083**.

## Open

No-embeddings baseline not run yet — don't know how much of the 0.9083 is from
the embeddings vs. just the sample.
