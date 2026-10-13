"""Quality aware re-ranking: trade a little engagement for less clickbait.

Takes the ranker's scores, standardises them within each impression, and subtracts
lambda * P(clickbait) from a separately trained content quality model
(see the content-quality-classifier repo). Sweeping lambda traces out the trade-off
curve between engagement (nDCG@10) and ecosystem health (clickbait share in top 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import evaluate


def load_quality_scores(path: str) -> dict[str, float]:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["news_id"], df["clickbait_prob"]))


def rerank(scores: list[np.ndarray], probs: list[np.ndarray], lam: float) -> list[np.ndarray]:
    out = []
    for s, p in zip(scores, probs):
        z = (s - s.mean()) / (s.std() + 1e-9)
        out.append(z - lam * p)
    return out


def top_k_clickbait_share(scores, probs, k=5, threshold=0.5, seed=0) -> float:
    rng = np.random.default_rng(seed)
    shares = []
    for s, p in zip(scores, probs):
        order = np.argsort(-(s + rng.random(len(s)) * 1e-9))[:k]
        shares.append(float((p[order] > threshold).mean()))
    return float(np.mean(shares))


def sweep(labels, scores, probs, lambdas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), seed=0):
    rows = []
    for lam in lambdas:
        s = rerank(scores, probs, lam)
        m = evaluate(labels, s, seed=seed)
        rows.append({"lambda": lam, "ndcg@10": m["ndcg@10"], "auc": m["auc"],
                     "clickbait_share_top5": top_k_clickbait_share(s, probs, seed=seed)})
    return rows
