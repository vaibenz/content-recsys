"""Two tower retrieval model.

News tower: title words (additive attention pooling) + category + subcategory.
It never uses a news id embedding, so a brand new article gets a vector from its
content alone. That is the cold start property a content platform needs.

User tower: additive attention over the vectors of the user's recent clicks,
with a learned fallback vector for users who have no history yet.

Both towers output L2 normalised vectors; relevance is cosine similarity / temperature.
Training uses MIND style impression negatives: one clicked item and K items the user
saw in the same impression but did not click, with a softmax cross entropy loss.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NewsTable, history_matrix

log = logging.getLogger(__name__)


class AdditiveAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L] (True for real positions)
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class NewsEncoder(nn.Module):
    def __init__(self, table: NewsTable, word_dim=128, cat_dim=32, out_dim=128, dropout=0.2):
        super().__init__()
        self.register_buffer("title", torch.from_numpy(table.title_tokens))
        self.register_buffer("cat", torch.from_numpy(table.category))
        self.register_buffer("sub", torch.from_numpy(table.subcategory))
        self.word = nn.Embedding(table.vocab_size, word_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(table.n_categories, cat_dim, padding_idx=0)
        self.sub_emb = nn.Embedding(table.n_subcategories, cat_dim, padding_idx=0)
        self.attn = AdditiveAttention(word_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(word_dim + 2 * cat_dim, out_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tokens = self.title[idx]                       # [M, T]
        mask = tokens > 0
        mask[:, 0] = True                              # empty titles still get a vector
        words = self.drop(self.word(tokens))
        title_vec = self.attn(words, mask)
        feats = torch.cat([title_vec, self.cat_emb(self.cat[idx]), self.sub_emb(self.sub[idx])], -1)
        return F.normalize(self.out(self.drop(feats)), dim=-1)


class UserEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.attn = AdditiveAttention(dim)
        self.out = nn.Linear(dim, dim)
        self.cold_user = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, hist_vecs: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.attn(hist_vecs, hist_mask)
        has_hist = hist_mask.any(-1, keepdim=True)
        pooled = torch.where(has_hist, pooled, self.cold_user.expand_as(pooled))
        return F.normalize(self.out(pooled), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, table: NewsTable, dim=128, temperature=0.1):
        super().__init__()
        self.news = NewsEncoder(table, out_dim=dim)
        self.user = UserEncoder(dim)
        self.temperature = temperature

    def encode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode each unique news item once per batch, then scatter back."""
        flat = idx.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        vecs = self.news(uniq)
        return vecs[inverse].reshape(*idx.shape, -1)

    def user_vectors(self, hist: torch.Tensor) -> torch.Tensor:
        return self.user(self.encode_indices(hist), hist > 0)

    def forward(self, hist: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        u = self.user_vectors(hist)                    # [B, D]
        c = self.encode_indices(cands)                 # [B, C, D]
        return torch.einsum("bd,bcd->bc", u, c) / self.temperature


def make_training_samples(behaviors: pd.DataFrame, table: NewsTable, n_neg: int = 4,
                          max_history: int = 50, seed: int = 0):
    """One row per click: [positive, n_neg negatives from the same impression]."""
    rng = np.random.default_rng(seed)
    hist_rows, cand_rows = [], []
    for row_i, (cands, labels) in enumerate(zip(behaviors["candidates"], behaviors["labels"])):
        idx = table.lookup(cands)
        pos, neg = idx[labels == 1], idx[labels == 0]
        for p in pos:
            negs = rng.choice(neg, n_neg, replace=len(neg) < n_neg)
            cand_rows.append(np.concatenate([[p], negs]))
            hist_rows.append(row_i)
    hist = history_matrix(behaviors["history"].tolist(), table, max_history)[hist_rows]
    return hist, np.stack(cand_rows)


def train_two_tower(behaviors: pd.DataFrame, table: NewsTable, epochs=2, batch_size=256,
                    lr=1e-3, n_neg=4, max_history=50, dim=128, device="cpu", seed=0,
                    max_samples: int | None = None) -> TwoTower:
    torch.manual_seed(seed)
    hist, cands = make_training_samples(behaviors, table, n_neg, max_history, seed)
    if max_samples and len(hist) > max_samples:
        keep = np.random.default_rng(seed).choice(len(hist), max_samples, replace=False)
        hist, cands = hist[keep], cands[keep]
    log.info("two tower: %d training samples, %d epochs", len(hist), epochs)

    model = TwoTower(table, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist_t, cand_t = torch.from_numpy(hist), torch.from_numpy(cands)
    target = torch.zeros(batch_size, dtype=torch.long, device=device)  # positive is column 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(hist_t))
        total, n_batches, t0 = 0.0, 0, time.time()
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            logits = model(hist_t[b].to(device), cand_t[b].to(device))
            loss = F.cross_entropy(logits, target[:len(b)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        log.info("epoch %d  loss %.4f  (%.0fs)", epoch + 1, total / max(n_batches, 1), time.time() - t0)
    model.eval()
    return model


@torch.no_grad()
def all_news_vectors(model: TwoTower, n_news: int, device="cpu", batch=4096) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, n_news, batch):
        idx = torch.arange(s, min(s + batch, n_news), device=device)
        out.append(model.news(idx).cpu())
    vecs = torch.cat(out).numpy()
    vecs[0] = 0.0
    return vecs


@torch.no_grad()
def user_vectors(model: TwoTower, news_vecs: np.ndarray, hist: np.ndarray, device="cpu",
                 batch=2048) -> np.ndarray:
    model.eval()
    nv = torch.from_numpy(news_vecs).to(device)
    out = []
    for s in range(0, len(hist), batch):
        h = torch.from_numpy(hist[s:s + batch]).to(device)
        out.append(model.user(nv[h], h > 0).cpu())
    return torch.cat(out).numpy()


def score_impressions(model: TwoTower, behaviors: pd.DataFrame, table: NewsTable,
                      max_history=50, device="cpu"):
    news_vecs = all_news_vectors(model, len(table.ids) + 1, device)
    hist = history_matrix(behaviors["history"].tolist(), table, max_history)
    users = user_vectors(model, news_vecs, hist, device)
    scores = [news_vecs[table.lookup(c)] @ u for u, c in zip(users, behaviors["candidates"])]
    return scores, news_vecs, users
