"""Simple baselines every model has to beat."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import NewsTable


@dataclass
class PopularityStats:
    clicks: Counter
    impressions: Counter
    prior_ctr: float
    prior_strength: float = 20.0

    @classmethod
    def fit(cls, behaviors: pd.DataFrame) -> "PopularityStats":
        clicks, imps = Counter(), Counter()
        for cands, labels in zip(behaviors["candidates"], behaviors["labels"]):
            for nid, y in zip(cands, labels):
                imps[nid] += 1
                clicks[nid] += int(y)
        total_imps = sum(imps.values())
        prior = sum(clicks.values()) / max(total_imps, 1)
        return cls(clicks, imps, prior)

    def smoothed_ctr(self, nid: str) -> float:
        """Beta prior smoothing: items with few impressions shrink toward the global CTR."""
        a = self.prior_ctr * self.prior_strength
        return (self.clicks[nid] + a) / (self.impressions[nid] + self.prior_strength)


def random_scores(behaviors: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [rng.random(len(c)) for c in behaviors["candidates"]]


def popularity_scores(behaviors: pd.DataFrame, stats: PopularityStats):
    return [np.array([stats.clicks[n] for n in c], dtype=float) for c in behaviors["candidates"]]


def category_affinity(history_idx: np.ndarray, cand_idx: np.ndarray, field: np.ndarray,
                      n_values: int, alpha: float = 1.0) -> np.ndarray:
    """P(candidate's category | user's click history), Laplace smoothed."""
    hist_vals = field[history_idx[history_idx > 0]]
    counts = np.bincount(hist_vals, minlength=n_values).astype(float)
    probs = (counts + alpha) / (counts.sum() + alpha * n_values)
    return probs[field[cand_idx]]


def category_scores(behaviors: pd.DataFrame, table: NewsTable):
    out = []
    for hist, cands in zip(behaviors["history"], behaviors["candidates"]):
        h = table.lookup(hist)
        c = table.lookup(cands)
        out.append(category_affinity(h, c, table.category, table.n_categories))
    return out
