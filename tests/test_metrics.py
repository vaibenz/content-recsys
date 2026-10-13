import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from recsys import metrics


def test_perfect_ranking_scores_one():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    m = metrics.evaluate([y], [s])
    assert m["auc"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx((1 + 1 / 2) / 2)
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["hit@3"] == 1.0


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 30)
        if y.sum() in (0, len(y)):
            continue
        s = rng.random(30)
        assert metrics.evaluate([y], [s])["auc"] == pytest.approx(roc_auc_score(y, s))


def test_ndcg_known_value():
    y = np.array([1, 0, 0])
    order = np.array([1, 0, 2])  # positive placed second
    assert metrics.ndcg(y, order, 10) == pytest.approx(1 / np.log2(3))


def test_ties_are_not_free():
    # All scores tied: the logged order must not decide the result.
    y = np.array([1] + [0] * 9)
    s = np.zeros(10)
    aucs = [metrics.evaluate([y], [s], seed=k)["auc"] for k in range(200)]
    assert np.mean(aucs) == pytest.approx(0.5, abs=0.06)


def test_recall_at_k():
    assert metrics.recall_at_k(np.array(["a", "b", "c"]), {"b", "z"}, 2) == 0.5
