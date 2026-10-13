# Content Recommendation and Ranking System

A two stage news recommender on [MIND](https://msnews.github.io), Microsoft's real click log
dataset, built around the problems a content platform actually has: ranking for engagement,
recommending articles nobody has clicked yet, serving users with no history, and deciding
whether a model improvement is real before shipping it.

![tests](https://github.com/vaibenz/content-recsys/actions/workflows/ci.yml/badge.svg)

```
all fresh articles ──► Two tower retrieval ──► LightGBM LambdaRank ──► quality aware re-rank ──► feed
                       (PyTorch, content only)  (12 features)           (optional, lambda sweep)
```

## Results (MIND small, dev split)

> Run `make data && make run`, then copy the tables from `results/RESULTS.md` here.

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Random | | | | |
| Popularity | | | | |
| Category affinity | | | | |
| Two tower | | | | |
| LightGBM ranker | | | | |
| Ranker without two tower (ablation) | | | | |

Simulated A/B test, popularity (control) vs ranker (treatment), outcome hit@3 per user:
lift ___, 95% CI [___, ___]. A/A check: CI should contain 0.

## Design decisions

**No leakage across time.** The training log is split by timestamp. The two tower model and
popularity statistics learn from the earlier 80%; the ranker trains on the later 20% using
features computed only from the earlier part; dev is the following day. The ranker therefore
learns how much to trust each signal under the same conditions it sees at serving time.

**Cold start by construction.** The news tower encodes title words, category and subcategory
and never uses an article ID embedding, so a brand new article has a meaningful vector the
moment it is published. Users with no history get a learned fallback vector, and the ranker
sees explicit `is_cold_user` and `is_cold_item` flags. Results are broken down by cold user
and cold item segments, because an average can hide a model that fails exactly the users a
new platform most needs to keep.

**Two tower model.** Additive attention over title words; additive attention over the user's
last 50 clicks; cosine similarity with temperature 0.1. Trained with MIND style impression
negatives (one click plus four articles the same user saw and skipped) and softmax cross entropy.
Retrieval is evaluated as Recall@K against the whole pool of fresh articles, not just the
logged candidates, because that is the job retrieval does.

**Ranker.** LightGBM LambdaRank optimising nDCG directly, with early stopping on a held out
group of impressions. Features: two tower score, smoothed CTR with a Beta prior, log clicks and
impressions, category and subcategory affinity, history length, cold flags, title length,
category, candidate count. An ablation retrains without the two tower score to measure what
the learned representation adds over hand built features.

**Evaluation.** AUC, MRR, nDCG@5 and nDCG@10 per impression, matching MIND's official
definitions, with random tie breaking so baselines with many tied scores are not flattered by
log order.

**Experimentation.** Users are hashed into arms and each arm ranks with its own policy. Outcomes
are averaged per user before testing, since the user is the randomisation unit and treating
correlated impressions as independent overstates significance. I report a bootstrap 95% CI and
Welch's t test, plus an A/A test as a sanity check on the whole setup. This is offline replay on
logged data, so it estimates lift only among articles users were actually shown.

**Quality versus engagement.** With scores from my
[content-quality-classifier](https://github.com/vaibenz/content-quality-classifier), the final
score becomes `standardised ranker score - lambda * P(clickbait)`. Sweeping lambda traces the
trade-off curve between nDCG@10 and the clickbait share of the top 5 slots.

## Run it

```bash
pip install -r requirements.txt
make test            # unit tests + end to end run on synthetic MIND format data
make data            # downloads MIND small (about 50k users, 65k articles)
make run             # full pipeline, writes results/RESULTS.md and results/metrics.json

# optional quality trade-off
make score-mind      # in the content-quality-classifier repo, cloned next to this one
make run-quality
```

The full run is CPU friendly. If it is slow, cap the two tower training set with
`--max-tt-samples 100000` or pass `--device cuda`.

## Layout

```
recsys/data.py        MIND parsing, time split, news encoding
recsys/baselines.py   random, popularity (smoothed CTR), category affinity
recsys/two_tower.py   news and user towers, training, scoring
recsys/ranker.py      feature building and LightGBM LambdaRank
recsys/metrics.py     AUC, MRR, nDCG, hit@k, recall@k
recsys/experiment.py  user bucketing, bootstrap CI, Welch test
recsys/quality.py     quality aware re-ranking and lambda sweep
scripts/run_pipeline.py   end to end run and report
scripts/make_synthetic.py tiny MIND format dataset for tests and CI
```

## What I would do next

* Swap the from scratch word encoder for a frozen sentence transformer and measure the cold item segment.
* Add a diversity term alongside quality, and track category coverage in the top 10.
* Estimate long term effects (retention) rather than clicks, which offline replay cannot see.
