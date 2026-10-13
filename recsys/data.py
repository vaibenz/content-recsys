"""Loading and preparing the MIND news recommendation dataset.

MIND ships two tab separated files per split:
  news.tsv       news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
  behaviors.tsv  impression_id, user_id, time, history, impressions

`history` is a space separated list of news ids the user clicked before the impression.
`impressions` is a space separated list like "N123-1 N456-0" (news id, clicked flag).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

PAD, UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def load_news(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, quoting=3,
                     dtype=str, keep_default_na=False)
    return df[["news_id", "category", "subcategory", "title", "abstract"]]


def load_behaviors(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, quoting=3,
                     dtype=str, keep_default_na=False)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].map(lambda s: s.split() if s else [])
    pairs = df["impressions"].map(lambda s: [p.rsplit("-", 1) for p in s.split()])
    df["candidates"] = pairs.map(lambda ps: [p[0] for p in ps])
    df["labels"] = pairs.map(lambda ps: np.array([int(p[1]) for p in ps], dtype=np.int8))
    df = df.drop(columns=["impressions"])
    # Impressions with no positive (or no negative) carry no ranking signal.
    has_both = df["labels"].map(lambda y: 0 < y.sum() < len(y))
    return df[has_both].reset_index(drop=True)


def time_split(behaviors: pd.DataFrame, holdout_frac: float = 0.2):
    """Split by time so later stages never see the future.

    Stage 1 (earlier) trains the two tower model and popularity statistics.
    Stage 2 (later) trains the ranker on features computed only from stage 1,
    which mirrors how features would look at serving time.
    """
    cutoff = behaviors["time"].quantile(1 - holdout_frac)
    early = behaviors[behaviors["time"] < cutoff].reset_index(drop=True)
    late = behaviors[behaviors["time"] >= cutoff].reset_index(drop=True)
    return early, late


@dataclass
class NewsTable:
    """Integer encoded news features. Row 0 is padding, so news index i >= 1."""
    ids: list[str]
    index: dict[str, int]
    title_tokens: np.ndarray      # [N+1, max_title_len]
    category: np.ndarray          # [N+1]
    subcategory: np.ndarray       # [N+1]
    title_len: np.ndarray         # [N+1]
    category_names: list[str]
    vocab_size: int
    n_categories: int
    n_subcategories: int

    def lookup(self, news_ids: list[str]) -> np.ndarray:
        return np.array([self.index.get(n, 0) for n in news_ids], dtype=np.int64)


def build_news_table(news: pd.DataFrame, max_title_len: int = 24,
                     min_freq: int = 2) -> NewsTable:
    """Encode every news item seen in any split.

    Only content (title, category) is used, never click labels, so encoding dev news
    here is not leakage: at serving time a new article's text is known on arrival.
    """
    news = news.drop_duplicates("news_id").reset_index(drop=True)
    token_lists = [tokenize(t) for t in news["title"]]
    counts = Counter(tok for toks in token_lists for tok in toks)
    vocab = {tok: i + 2 for i, tok in enumerate(sorted(t for t, c in counts.items() if c >= min_freq))}

    cats = sorted(news["category"].unique())
    subcats = sorted(news["subcategory"].unique())
    cat_idx = {c: i + 1 for i, c in enumerate(cats)}
    sub_idx = {c: i + 1 for i, c in enumerate(subcats)}

    n = len(news) + 1
    title = np.zeros((n, max_title_len), dtype=np.int64)
    title_len = np.zeros(n, dtype=np.int64)
    for row, toks in enumerate(token_lists, start=1):
        ids = [vocab.get(t, UNK) for t in toks][:max_title_len]
        title[row, :len(ids)] = ids
        title_len[row] = len(toks)

    category = np.zeros(n, dtype=np.int64)
    subcategory = np.zeros(n, dtype=np.int64)
    category[1:] = news["category"].map(cat_idx).to_numpy()
    subcategory[1:] = news["subcategory"].map(sub_idx).to_numpy()

    ids = news["news_id"].tolist()
    return NewsTable(
        ids=ids, index={nid: i + 1 for i, nid in enumerate(ids)},
        title_tokens=title, category=category, subcategory=subcategory, title_len=title_len,
        category_names=["<pad>"] + cats, vocab_size=len(vocab) + 2,
        n_categories=len(cats) + 1, n_subcategories=len(subcats) + 1,
    )


def history_matrix(histories, table: NewsTable, max_history: int = 50) -> np.ndarray:
    """Most recent `max_history` clicks per user as news indices, left aligned, 0 padded."""
    out = np.zeros((len(histories), max_history), dtype=np.int64)
    for i, hist in enumerate(histories):
        idx = table.lookup(hist[-max_history:])
        out[i, :len(idx)] = idx
    return out


def load_split(data_dir: str | Path, split: str):
    d = Path(data_dir) / split
    return load_news(d / "news.tsv"), load_behaviors(d / "behaviors.tsv")
