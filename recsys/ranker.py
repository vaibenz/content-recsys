"""Second stage ranker: LightGBM LambdaRank over hand built and model features.

Feature hygiene matters more than model choice here:
  * popularity statistics come only from impressions logged BEFORE the rows being scored
  * the two tower score comes from a model trained only on that earlier period
so the ranker learns how much to trust each signal under the same conditions it sees in serving.
"""
from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import PopularityStats, category_affinity
from .data import NewsTable, history_matrix

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

FEATURES = [
    "tt_score", "log_clicks", "log_imps", "smoothed_ctr", "is_cold_item",
    "cat_affinity", "subcat_affinity", "log_hist_len", "is_cold_user",
    "title_len", "category", "n_candidates",
]


def build_features(behaviors: pd.DataFrame, table: NewsTable, stats: PopularityStats,
                   tt_scores: list[np.ndarray], quality: dict[str, float] | None = None,
                   max_history: int = 50) -> pd.DataFrame:
    hist = history_matrix(behaviors["history"].tolist(), table, max_history)
    frames = []
    for i, (cands, labels) in enumerate(zip(behaviors["candidates"], behaviors["labels"])):
        c = table.lookup(cands)
        h = hist[i]
        clicks = np.array([stats.clicks[n] for n in cands], dtype=float)
        imps = np.array([stats.impressions[n] for n in cands], dtype=float)
        n_hist = len(behaviors["history"].iat[i])
        f = {
            "impression": np.full(len(c), i),
            "label": labels,
            "tt_score": tt_scores[i],
            "log_clicks": np.log1p(clicks),
            "log_imps": np.log1p(imps),
            "smoothed_ctr": np.array([stats.smoothed_ctr(n) for n in cands]),
            "is_cold_item": (imps == 0).astype(np.int8),
            "cat_affinity": category_affinity(h, c, table.category, table.n_categories),
            "subcat_affinity": category_affinity(h, c, table.subcategory, table.n_subcategories),
            "log_hist_len": np.full(len(c), np.log1p(n_hist)),
            "is_cold_user": np.full(len(c), int(n_hist == 0), dtype=np.int8),
            "title_len": table.title_len[c],
            "category": table.category[c],
            "n_candidates": np.full(len(c), len(c)),
        }
        if quality is not None:
            f["clickbait_prob"] = np.array([quality.get(n, np.nan) for n in cands])
        frames.append(pd.DataFrame(f))
    return pd.concat(frames, ignore_index=True)


def train_ranker(train_df: pd.DataFrame, features: list[str], seed: int = 0,
                 n_estimators: int = 300, valid_frac: float = 0.1) -> lgb.LGBMRanker:
    imps = train_df["impression"].unique()
    rng = np.random.default_rng(seed)
    valid_imps = set(rng.choice(imps, max(1, int(len(imps) * valid_frac)), replace=False))
    is_valid = train_df["impression"].isin(valid_imps)
    tr, va = train_df[~is_valid], train_df[is_valid]

    def groups(df):
        return df.groupby("impression", sort=False).size().to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=seed, verbose=-1,
    )
    model.fit(
        tr[features], tr["label"], group=groups(tr),
        eval_set=[(va[features], va["label"])], eval_group=[groups(va)],
        eval_at=[10], categorical_feature=["category"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    log.info("ranker: best iteration %s", model.best_iteration_)
    return model


def scores_per_impression(df: pd.DataFrame, scores: np.ndarray) -> list[np.ndarray]:
    out = pd.Series(scores, index=df.index).groupby(df["impression"], sort=True)
    return [g.to_numpy() for _, g in out]


def labels_per_impression(df: pd.DataFrame) -> list[np.ndarray]:
    return [g.to_numpy() for _, g in df.groupby("impression", sort=True)["label"]]


def feature_importance(model: lgb.LGBMRanker, features: list[str]) -> list[tuple[str, float]]:
    gain = model.booster_.feature_importance(importance_type="gain")
    total = gain.sum() or 1.0
    return sorted(((f, float(g / total)) for f, g in zip(features, gain)), key=lambda x: -x[1])
