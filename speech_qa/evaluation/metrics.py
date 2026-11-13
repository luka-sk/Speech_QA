"""Evaluation metrics (pure functions).

Ranking metrics use binary relevance: a speech is relevant or not.
"""

import math
from collections.abc import Collection, Iterable, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any of the first ``k`` results is relevant, else 0.0."""
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant items found among the first ``k`` results."""
    if not relevant:
        msg = "recall is undefined without relevant items"
        raise ValueError(msg)
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant result; 0.0 if there is none."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalised discounted cumulative gain of the first ``k`` results."""
    if not relevant:
        msg = "nDCG is undefined without relevant items"
        raise ValueError(msg)
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(set(relevant)), k) + 1))
    return gain / ideal


def micro_precision_recall(
    pairs: Iterable[tuple[Collection[str], Collection[str]]],
) -> tuple[float | None, float | None]:
    """Micro-averaged precision and recall over ``(predicted, expected)`` pairs.

    Returns ``None`` for a value whose denominator is zero.
    """
    true_positives = predicted_total = expected_total = 0
    for predicted, expected in pairs:
        true_positives += len(set(predicted) & set(expected))
        predicted_total += len(set(predicted))
        expected_total += len(set(expected))
    precision = true_positives / predicted_total if predicted_total else None
    recall = true_positives / expected_total if expected_total else None
    return precision, recall


def mean(values: Iterable[float]) -> float | None:
    """Arithmetic mean, or ``None`` for no values."""
    items = list(values)
    return sum(items) / len(items) if items else None


def percentile(values: Iterable[float], share: float) -> float | None:
    """Percentile by linear interpolation (``share`` 0.5 = median), ``None`` for no values."""
    items = sorted(values)
    if not items:
        return None
    position = (len(items) - 1) * share
    lower, upper = math.floor(position), math.ceil(position)
    return items[lower] + (items[upper] - items[lower]) * (position - lower)
