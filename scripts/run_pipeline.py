"""End to end: baselines -> two tower -> LightGBM ranker -> A/B simulation -> report.

Usage:
  python scripts/run_pipeline.py --data-dir data/MINDsmall
  python scripts/run_pipeline.py --data-dir data/synthetic --quick      # smoke test
  python scripts/run_pipeline.py --data-dir data/MINDsmall --quality-scores data/mind_quality.tsv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import baselines, data, experiment, metrics, quality, ranker, two_tower  # noqa: E402

log = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="folder with train/ and dev/ subfolders")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-tt-samples", type=int, default=None, help="cap two tower training rows")
    p.add_argument("--max-rank-impressions", type=int, default=30000,
                   help="cap stage 2 impressions used to train the ranker")
    p.add_argument("--max-eval-impressions", type=int, default=None)
    p.add_argument("--quality-scores", default=None, help="tsv: news_id, clickbait_prob")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="tiny settings for CI")
    return p.parse_args()


def sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n, random_state=seed).sort_values("time").reset_index(drop=True)


def retrieval_eval(news_vecs, user_vecs, dev: pd.DataFrame, table, stats, ks=(10, 50, 100),
                   max_users=5000, seed=0):
    """Recall@K when retrieving from every article shown in the dev period (the fresh pool)."""
    pool_ids = sorted({n for c in dev["candidates"] for n in c})
    pool_idx = table.lookup(pool_ids)
    pool_vecs = news_vecs[pool_idx]
    pop = np.array([stats.clicks[n] for n in pool_ids], dtype=float)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(dev), min(max_users, len(dev)), replace=False)
    pool_arr = np.array(pool_ids)
    pop_rank = pool_arr[np.argsort(-(pop + rng.random(len(pop)) * 1e-9))]
    res = {f"two_tower_recall@{k}": [] for k in ks} | {f"popularity_recall@{k}": [] for k in ks}
    for r in rows:
        clicked = {n for n, y in zip(dev["candidates"].iat[r], dev["labels"].iat[r]) if y}
        tt_rank = pool_arr[np.argsort(-(pool_vecs @ user_vecs[r]))]
        for k in ks:
            res[f"two_tower_recall@{k}"].append(metrics.recall_at_k(tt_rank, clicked, k))
            res[f"popularity_recall@{k}"].append(metrics.recall_at_k(pop_rank, clicked, k))
    out = {k: float(np.nanmean(v)) for k, v in res.items()}
    out["pool_size"] = len(pool_ids)
    return out


def segment_eval(dev, per_impression, stats):
    """Break results down for the two cold start problems: new users and new articles."""
    cold_user = dev["history"].map(len).to_numpy() == 0
    pos_cold_item = np.array([
        any(stats.impressions[n] == 0 for n, y in zip(c, l) if y)
        for c, l in zip(dev["candidates"], dev["labels"])
    ])
    segments = {"all": np.ones(len(dev), bool), "cold_user": cold_user, "warm_user": ~cold_user,
                "clicked_cold_item": pos_cold_item}
    out = {}
    for seg, mask in segments.items():
        if mask.sum() == 0:
            continue
        out[seg] = {"n_impressions": int(mask.sum())}
        for name, rows in per_impression.items():
            out[seg][name] = round(metrics.summarise(rows, mask)["ndcg@10"], 4)
    return out


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.quick:
        args.epochs, args.max_rank_impressions = 1, 2000
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    news_tr, beh_tr = data.load_split(args.data_dir, "train")
    news_dev, beh_dev = data.load_split(args.data_dir, "dev")
    beh_dev = sample(beh_dev, args.max_eval_impressions, args.seed)
    table = data.build_news_table(pd.concat([news_tr, news_dev]))
    stage1, stage2 = data.time_split(beh_tr, holdout_frac=0.2)
    stage2 = sample(stage2, args.max_rank_impressions, args.seed)
    log.info("news %d | stage1 %d | stage2 %d | dev %d impressions",
             len(table.ids), len(stage1), len(stage2), len(beh_dev))

    labels = beh_dev["labels"].tolist()
    stats_stage1 = baselines.PopularityStats.fit(stage1)
    stats_full = baselines.PopularityStats.fit(beh_tr)  # all logged history before dev

    results = {"config": vars(args), "data": {
        "n_news": len(table.ids), "n_stage1": len(stage1), "n_stage2": len(stage2),
        "n_dev": len(beh_dev)}}
    scores = {
        "random": baselines.random_scores(beh_dev, args.seed),
        "popularity": baselines.popularity_scores(beh_dev, stats_full),
        "category_affinity": baselines.category_scores(beh_dev, table),
    }

    log.info("training two tower model")
    model = two_tower.train_two_tower(stage1, table, epochs=args.epochs, batch_size=args.batch_size,
                                      device=args.device, seed=args.seed,
                                      max_samples=args.max_tt_samples)
    scores["two_tower"], news_vecs, dev_users = two_tower.score_impressions(
        model, beh_dev, table, device=args.device)
    tt_stage2, _, _ = two_tower.score_impressions(model, stage2, table, device=args.device)

    log.info("building ranker features")
    qual = quality.load_quality_scores(args.quality_scores) if args.quality_scores else None
    f_train = ranker.build_features(stage2, table, stats_stage1, tt_stage2, qual)
    f_dev = ranker.build_features(beh_dev, table, stats_full, scores["two_tower"], qual)
    features = list(ranker.FEATURES)
    lgbm = ranker.train_ranker(f_train, features, seed=args.seed)
    dev_scores = lgbm.predict(f_dev[features])
    scores["lightgbm_ranker"] = ranker.scores_per_impression(f_dev, dev_scores)
    results["feature_importance"] = ranker.feature_importance(lgbm, features)

    # Ablation: how much does the learned two tower signal add on top of hand built features?
    no_tt = [f for f in features if f != "tt_score"]
    lgbm_no_tt = ranker.train_ranker(f_train, no_tt, seed=args.seed)
    scores["ranker_without_two_tower"] = ranker.scores_per_impression(
        f_dev, lgbm_no_tt.predict(f_dev[no_tt]))

    log.info("evaluating")
    per_imp, results["offline"] = {}, {}
    for name, s in scores.items():
        results["offline"][name], per_imp[name] = metrics.evaluate(
            labels, s, seed=args.seed, return_per_impression=True)
    results["retrieval"] = retrieval_eval(news_vecs, dev_users, beh_dev, table, stats_full, seed=args.seed)
    results["segments"] = segment_eval(beh_dev, per_imp, stats_full)

    # A/B simulation: popularity (control) vs ranker (treatment), plus an A/A sanity check.
    hit_pop = [r["hit@3"] for r in per_imp["popularity"]]
    hit_rank = [r["hit@3"] for r in per_imp["lightgbm_ranker"]]
    users = beh_dev["user_id"].tolist()
    results["ab_test"] = experiment.simulate(users, hit_pop, hit_rank, seed=args.seed)
    results["aa_test"] = experiment.simulate(users, hit_pop, hit_pop, salt="aa", seed=args.seed)

    if qual is not None:
        probs = [np.nan_to_num(g, nan=np.nanmean(f_dev["clickbait_prob"]))
                 for _, g in f_dev.groupby("impression", sort=True)["clickbait_prob"]]
        results["quality_tradeoff"] = quality.sweep(labels, scores["lightgbm_ranker"], probs, seed=args.seed)

    results["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    write_report(results, out)
    log.info("done in %.1f min, wrote %s", results["runtime_minutes"], out)


def write_report(r: dict, out: Path):
    lines = ["# Results", "", f"Dev impressions evaluated: {r['data']['n_dev']:,}  ",
             f"News items: {r['data']['n_news']:,}", "", "## Offline ranking (MIND dev)", "",
             "| Model | AUC | MRR | nDCG@5 | nDCG@10 |", "|---|---|---|---|---|"]
    for name, m in r["offline"].items():
        lines.append(f"| {name} | {m['auc']:.4f} | {m['mrr']:.4f} | {m['ndcg@5']:.4f} | {m['ndcg@10']:.4f} |")
    lines += ["", "## Retrieval from the full fresh pool", "",
              f"Pool size: {r['retrieval']['pool_size']:,} articles", "",
              "| K | Two tower recall | Popularity recall |", "|---|---|---|"]
    for k in (10, 50, 100):
        lines.append(f"| {k} | {r['retrieval'][f'two_tower_recall@{k}']:.4f} | "
                     f"{r['retrieval'][f'popularity_recall@{k}']:.4f} |")
    lines += ["", "## Cold start segments (nDCG@10)", ""]
    segs = r["segments"]
    models = [k for k in next(iter(segs.values())) if k != "n_impressions"]
    lines += ["| Segment | n | " + " | ".join(models) + " |", "|---" * (len(models) + 2) + "|"]
    for seg, row in segs.items():
        lines.append(f"| {seg} | {row['n_impressions']:,} | " + " | ".join(f"{row[m]:.4f}" for m in models) + " |")
    lines += ["", "## Ranker feature importance (share of gain)", ""]
    lines += [f"- {f}: {g:.3f}" for f, g in r["feature_importance"]]
    for key, title in (("ab_test", "Simulated A/B test: popularity (control) vs ranker (treatment)"),
                       ("aa_test", "A/A sanity check (same policy in both arms)")):
        a = r[key]
        lines += ["", f"## {title}", "", "Outcome: hit@3 averaged per user (user is the randomisation unit).", "",
                  f"- users: {a['n_users_control']:,} control, {a['n_users_treatment']:,} treatment",
                  f"- control {a['control_mean']:.4f}, treatment {a['treatment_mean']:.4f}",
                  f"- lift {a['abs_lift']:+.4f} ({a['rel_lift']:+.1%}), "
                  f"95% bootstrap CI [{a['ci95_low']:+.4f}, {a['ci95_high']:+.4f}], Welch p = {a['p_value']:.2g}"]
    if "quality_tradeoff" in r:
        lines += ["", "## Engagement vs clickbait trade-off", "",
                  "| lambda | nDCG@10 | clickbait share in top 5 |", "|---|---|---|"]
        for row in r["quality_tradeoff"]:
            lines.append(f"| {row['lambda']} | {row['ndcg@10']:.4f} | {row['clickbait_share_top5']:.3f} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
