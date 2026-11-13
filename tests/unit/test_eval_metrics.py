import math

import pytest

from speech_qa.evaluation import metrics

RANKED = ["a", "b", "c", "d"]


def test_hit_at_k() -> None:
    assert metrics.hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert metrics.hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert metrics.hit_at_k([], {"c"}, 5) == 0.0


def test_recall_at_k() -> None:
    assert metrics.recall_at_k(RANKED, {"a", "d", "x"}, 4) == pytest.approx(2 / 3)
    assert metrics.recall_at_k(RANKED, {"d"}, 2) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        metrics.recall_at_k(RANKED, set(), 2)


def test_reciprocal_rank() -> None:
    assert metrics.reciprocal_rank(RANKED, {"b", "d"}) == 0.5
    assert metrics.reciprocal_rank(RANKED, {"x"}) == 0.0


def test_ndcg_at_k() -> None:
    assert metrics.ndcg_at_k(RANKED, {"a", "b"}, 4) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert metrics.ndcg_at_k(RANKED, {"b"}, 4) == pytest.approx(expected)
    assert metrics.ndcg_at_k(RANKED, {"x"}, 4) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        metrics.ndcg_at_k(RANKED, set(), 4)


def test_micro_precision_recall() -> None:
    precision, recall = metrics.micro_precision_recall(
        [({"A", "B"}, {"A"}), (set(), {"C"}), ({"D"}, set())]
    )

    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 2)
    assert metrics.micro_precision_recall([(set(), set())]) == (None, None)


def test_mean_and_percentile() -> None:
    assert metrics.mean([1.0, 2.0, 6.0]) == 3.0
    assert metrics.mean([]) is None
    assert metrics.percentile([1.0, 3.0, 2.0, 4.0], 0.5) == 2.5
    assert metrics.percentile([10.0, 20.0], 0.95) == pytest.approx(19.5)
    assert metrics.percentile([5.0], 0.95) == 5.0
    assert metrics.percentile([], 0.5) is None
