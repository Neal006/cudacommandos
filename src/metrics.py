"""The competition metric: macro-averaged F_0.5 over Source-1 entities.

Scored per Source-1 entity, then averaged. Singletons count: predicting an
empty list for an entity with no true matches scores 1.0, predicting anything
for it scores 0.0. That makes "say nothing when unsure" a real strategy, not
just a safe one.
"""
from typing import Dict, Iterable, Set

BETA = 0.5
_B2 = BETA * BETA  # 0.25


def f_beta_entity(true_ids: Set[str], pred_ids: Set[str], beta2: float = _B2) -> float:
    """F_beta for a single Source-1 entity."""
    if not true_ids and not pred_ids:
        return 1.0                      # correctly identified singleton
    if not true_ids or not pred_ids:
        return 0.0                      # predicted on a singleton, or missed everything
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f_beta(truth: Dict[str, Set[str]], preds: Dict[str, Set[str]]) -> float:
    """Macro F_0.5 across every Source-1 entity in `truth`.

    Entities missing from `preds` are scored as an empty prediction, which is
    what the grader effectively does to a submission that omits them (and the
    validator rejects outright).
    """
    if not truth:
        return 0.0
    total = sum(
        f_beta_entity(true_ids, preds.get(s1, set()))
        for s1, true_ids in truth.items()
    )
    return total / len(truth)


def scores_breakdown(truth: Dict[str, Set[str]], preds: Dict[str, Set[str]]) -> dict:
    """Where the score is actually coming from.

    Singletons and multi-match entities fail in different ways and want
    different thresholds; averaging them hides that.
    """
    singles = {k: v for k, v in truth.items() if not v}
    multis = {k: v for k, v in truth.items() if v}
    return {
        "macro_f05": macro_f_beta(truth, preds),
        "n_entities": len(truth),
        "n_singletons": len(singles),
        "singleton_f05": macro_f_beta(singles, preds) if singles else None,
        "matched_f05": macro_f_beta(multis, preds) if multis else None,
        "mean_pred_size": (sum(len(preds.get(k, ())) for k in truth) / len(truth)),
        "mean_true_size": (sum(len(v) for v in truth.values()) / len(truth)),
    }


def blocking_recall(truth: Dict[str, Set[str]], candidates: Dict[str, Set[str]]) -> dict:
    """Recall ceiling of the candidate set — the best any matcher could score.

    Run this before tuning the model. If the ceiling is 0.90, no amount of
    matcher work gets you past it.
    """
    total_true = sum(len(v) for v in truth.values())
    if total_true == 0:
        return {"pair_recall": None, "mean_candidates": 0.0}
    covered = sum(len(v & candidates.get(s1, set())) for s1, v in truth.items())
    n_cand = sum(len(candidates.get(k, ())) for k in truth)
    return {
        "pair_recall": covered / total_true,
        "mean_candidates": n_cand / len(truth),
        "total_true_pairs": total_true,
        "covered_true_pairs": covered,
    }


def parse_id_list(cell) -> Set[str]:
    """Parse a comma-separated matched/candidate id cell into a set.

    Empty cells, NaN and whitespace all mean 'no matches'.
    """
    if cell is None:
        return set()
    s = str(cell)
    if not s or s.lower() == "nan":
        return set()
    return {tok.strip() for tok in s.split(",") if tok.strip()}


def format_id_list(ids: Iterable[str]) -> str:
    """Sorted, comma-joined, no spaces — the format the validator expects."""
    return ",".join(sorted(set(ids)))
