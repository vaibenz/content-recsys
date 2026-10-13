"""Generate a tiny dataset in the exact MIND file format, for tests and CI.

Users prefer a couple of categories, titles use category specific words, and some
articles are intrinsically more clickable, so a working model should beat random.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CATS = {
    "sports": ["match", "league", "coach", "goal", "season", "striker"],
    "finance": ["stocks", "market", "rates", "earnings", "bank", "inflation"],
    "travel": ["beach", "city", "flight", "hotel", "island", "guide"],
    "food": ["recipe", "dinner", "chef", "noodles", "bakery", "spicy"],
    "health": ["sleep", "workout", "diet", "doctor", "study", "habits"],
    "tech": ["phone", "chip", "startup", "software", "robot", "cloud"],
}
FILLER = ["new", "best", "how", "why", "top", "local", "week", "people", "first", "big"]


def make(out: Path, n_users=400, n_news=900, n_train_imps=3000, n_dev_imps=800, seed=0):
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    news = []
    for i in range(n_news):
        c = cats[i % len(cats)]
        words = list(rng.choice(CATS[c], 3)) + list(rng.choice(FILLER, 2))
        rng.shuffle(words)
        news.append({"id": f"N{i + 1}", "cat": c, "sub": f"{c}_{i % 3}",
                     "title": " ".join(words).capitalize(), "appeal": rng.normal(0, 1),
                     "dev_only": i >= int(n_news * 0.8)})
    pref = {f"U{u}": rng.choice(len(cats), 2, replace=False) for u in range(n_users)}

    def click_prob(user, item):
        match = 1.0 if cats.index(item["cat"]) in pref[user] else 0.0
        return 1 / (1 + np.exp(-(2.2 * match + 0.6 * item["appeal"] - 2.0)))

    def behaviors(n_imps, start, days, pool, imp_offset):
        rows = []
        for k in range(n_imps):
            user = f"U{rng.integers(n_users)}"
            liked = [n for n in news if cats.index(n["cat"]) in pref[user] and not n["dev_only"]]
            n_hist = 0 if rng.random() < 0.08 else int(rng.integers(1, 15))
            hist = [n["id"] for n in rng.choice(liked, n_hist)] if n_hist else []
            cands = list(rng.choice(pool, int(rng.integers(8, 20)), replace=False))
            labels = [int(rng.random() < click_prob(user, c)) for c in cands]
            if sum(labels) == 0:
                labels[int(np.argmax([click_prob(user, c) for c in cands]))] = 1
            t = start + timedelta(seconds=float(rng.uniform(0, days * 86400)))
            imps = " ".join(f"{c['id']}-{y}" for c, y in zip(cands, labels))
            rows.append((imp_offset + k + 1, user, t.strftime("%m/%d/%Y %I:%M:%S %p").lstrip("0"),
                         " ".join(hist), imps))
        return sorted(rows, key=lambda r: datetime.strptime(r[2], "%m/%d/%Y %I:%M:%S %p"))

    train_pool = [n for n in news if not n["dev_only"]]
    dev_pool = news[int(n_news * 0.5):]  # mix of older items and brand new (cold) items
    splits = {
        "train": behaviors(n_train_imps, datetime(2019, 11, 9), 6, train_pool, 0),
        "dev": behaviors(n_dev_imps, datetime(2019, 11, 15), 1, dev_pool, n_train_imps),
    }
    for split, rows in splits.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "news.tsv", "w") as f:
            for n in news:
                f.write("\t".join([n["id"], n["cat"], n["sub"], n["title"], "", "", "[]", "[]"]) + "\n")
        with open(d / "behaviors.tsv", "w") as f:
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make(Path(a.out), seed=a.seed)
    print(f"wrote synthetic MIND format data to {a.out}")
