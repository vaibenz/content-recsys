"""Ranking metrics, matching the official MIND evaluation definitions.

All metrics are computed per impression and then averaged across impressions.
Ties are broken randomly (seeded) so a baseline with many tied scores is not
accidentally helped or hurt by the order candidates were logged in.
"""
from __future__ import annotations

import numpy as np


def _order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.random(len(scores)) * 1e-9
    return np.argsort(-(scores.astype(np.float64) + jitter), kind="stable")


def mrr(labels: np.ndarray, order: np.ndarray) -> float:
    ranked = labels[order]
    return float(np.sum(ranked / (np.arange(len(ranked)) + 1)) / np.sum(ranked))


def dcg(ranked_labels: np.ndarray, k: int) -> float:
    gains = 2 ** ranked_labels[:k] - 1
    return float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)))


def ndcg(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    ideal = dcg(np.sort(labels)[::-1], k)
    return dcg(labels[order], k) / ideal if ideal > 0 else 0.0


def hit_at_k(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    return float(labels[order[:k]].max())


def auc(labels: np.ndarray, order: np.ndarray) -> float:
    """ROC AUC via the Mann Whitney U statistic, using the same tie broken order."""
    n = len(labels)
    ascending_rank = np.empty(n)
    ascending_rank[order] = np.arange(n, 0, -1)
    n_pos = labels.sum()
    n_neg = n - n_pos
    return float((ascending_rank[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def impression_metrics(labels: np.ndarray, scores: np.ndarray,
                       rng: np.random.Generator) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = _order(scores, rng)
    return {
        "auc": auc(labels, order),
        "mrr": mrr(labels, order),
        "ndcg@5": ndcg(labels, order, 5),
        "ndcg@10": ndcg(labels, order, 10),
        "hit@3": hit_at_k(labels, order, 3),
    }


def evaluate(labels_list, scores_list, seed: int = 0, return_per_impression: bool = False):
    rng = np.random.default_rng(seed)
    rows = [impression_metrics(y, s, rng) for y, s in zip(labels_list, scores_list)]
    summary = summarise(rows)
    if return_per_impression:
        return summary, rows
    return summary


def summarise(rows: list[dict], mask=None) -> dict[str, float]:
    if mask is not None:
        rows = [r for r, m in zip(rows, mask) if m]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    summary["n_impressions"] = len(rows)
    return summary


def recall_at_k(ranked_ids: np.ndarray, relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    hits = sum(1 for x in ranked_ids[:k] if x in relevant)
    return hits / len(relevant)
