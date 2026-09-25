# Models to Test & the Decision Framework

## 1. Candidates by pipeline role

Licences are from each project's published licence as we know it. **Re-check the model
card / LICENSE file before first use** and paste the link into the methodology document.
The rule: MIT or Apache-2.0, ≤ 8B parameters. We apply it to every model in the pipeline
(ANALYSIS R4).

### Matcher (core): tabular pair classifier
| Model | Licence | Size | Why try it | Cost | Priority |
|---|---|---|---|---|---|
| **LightGBM** (existing) | MIT | — | Fast, strong on similarity features, native NaN handling, gain importances | minutes on CPU | **Champion baseline** |
| XGBoost `hist` | Apache-2.0 | — | Different regularization; ensemble diversity | minutes | E16 |
| CatBoost | Apache-2.0 | — | Ordered boosting, robust defaults, less overfit | 2–3× LightGBM | E16 |
| Logistic regression (sklearn, BSD) | BSD-3 (library, not a pretrained model) | — | Calibrated sanity baseline; best-case domain transfer | seconds | reference |
| Fellegi–Sunter EM (own code) | ours | — | Unsupervised, label-free: a France fallback / extra features | minutes | E18 |

### Reranker on the uncertain band (optional, GPU)
| Model | Licence | Params | Why | Priority |
|---|---|---|---|---|
| `microsoft/mdeberta-v3-base` cross-encoder | MIT | ~280M | Multilingual (100+ languages) and handles Indic/French; strong pair classifier after fine-tuning | E14 first |
| `xlm-roberta-base` cross-encoder | MIT | ~280M | Alternative multilingual backbone | E14 alt |
| `google/byt5-small` | Apache-2.0 | ~300M | Byte-level: robust to typos and scripts, no tokenizer OOV | only if the others fail on scripts |
| `Qwen2.5-1.5B/7B-Instruct` as a zero-shot judge | Apache-2.0 (**not** the 3B, which has a different licence) | 1.5B / 7.6B | Reasoning on hard cases; slow | P3, only for ≤ 100k pairs |

### Blocking embedder (optional pass E, ANN)
| Model | Licence | Params | Why |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | ~118M | Small, multilingual, good short-text retrieval |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | ~118M | Fast CPU inference |
| `sentence-transformers/LaBSE` | Apache-2.0 | ~470M | Strong cross-script alignment, heavier |
| `BAAI/bge-m3` | MIT | ~570M | Dense+sparse hybrid; GPU only |

### Libraries (not "models", but licence-relevant)
rapidfuzz (MIT), polars (MIT), anyascii (ISC), indic-transliteration (MIT), scikit-learn
(BSD-3), scipy (BSD-3), faiss (MIT), sparse_dot_topn (**check licence before adding**).

## 2. Decision framework: champion / challenger

Every candidate run produces `runs/<id>/summary.json`. A challenger replaces the champion
only by passing **all gates**, then **winning the ladder**.

### Gates (hard, any failure = rejected)
| Gate | Rule | Enforced by |
|---|---|---|
| G1 Licence | MIT/Apache-2.0 and ≤ 8B, for every model used | review + this table |
| G2 Trust | `mlguard run` PASS: no overfit gap > 0.03, fold std ≤ 0.01, no leak signatures, threshold from OOF | mlguard |
| G3 France sanity | `mlguard submission --summary` PASS: France singleton rate and links/entity within tolerance of OOF | mlguard |
| G4 Budget | Full test inference fits the remaining window with a 2 h margin | run log timings |
| G5 Reproducible | Clean clone + cache → identical outputs (hash) | SYSTEM_ARCHITECTURE §5 |

### Ladder (lexicographic: the first rung that separates them decides)
1. **OOF macro F0.5**: the challenger wins if Δ > noise floor = max(0.002, 2·fold_std/√5).
2. **Worst-country F0.5**: higher wins (guards the India-heavy test).
3. **LOCO transfer** (US→India, India→US mean): higher wins (the France proxy).
4. **Hard-negative slice F0.5** (same name + same locality, ANALYSIS §1.6).
5. **Calibration** (ECE on OOF after isotonic): lower wins; expected-F depends on it.
6. **Simplicity / runtime**: fewer stages, faster wins.

### When to stop investing in a model family
- After 2 runs with Δ inside the noise floor → freeze the family and move on.
- If a family needs a GPU and G4 is at risk after 26 Sep 18:00 → drop it.
- If LOCO drops > 0.05 while OOF rises → it's memorizing country vocabulary; reject it
  even if OOF is higher (France is 15% of test).

### Expected ordering (priors, guesses)
1. LightGBM + features v2 + decision layer: most of the gain.
2. + stage-2 competition/peer.
3. + seed/fold averaging and a GBDT blend.
4. + cross-encoder on the uncertain band (only if G4 holds).

### Recording the decision
`runs/champion.json` = a copy of the winning `summary.json`. The commit message states the
ladder rung that decided it: `champion: 007_stage2 (rung 1, +0.0041 OOF)`.
