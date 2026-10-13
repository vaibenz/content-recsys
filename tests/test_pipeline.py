import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic import make  # noqa: E402

from recsys import data  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("mind")
    make(d, n_users=200, n_news=400, n_train_imps=1200, n_dev_imps=300)
    return d


def test_loading_and_time_split(synthetic):
    news, beh = data.load_split(synthetic, "train")
    assert {"news_id", "category", "title"} <= set(news.columns)
    assert all(0 < y.sum() < len(y) for y in beh["labels"])
    early, late = data.time_split(beh, 0.2)
    assert early["time"].max() <= late["time"].min()


def test_news_table_handles_unknown_ids(synthetic):
    news, _ = data.load_split(synthetic, "train")
    table = data.build_news_table(news)
    idx = table.lookup(["N1", "does-not-exist"])
    assert idx[0] > 0 and idx[1] == 0
    assert table.title_tokens[0].sum() == 0


def test_end_to_end_beats_random(synthetic, tmp_path):
    out = tmp_path / "results"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
                    "--data-dir", str(synthetic), "--out", str(out), "--quick"], check=True)
    r = json.loads((out / "metrics.json").read_text())
    assert r["offline"]["lightgbm_ranker"]["auc"] > r["offline"]["random"]["auc"] + 0.1
    assert r["offline"]["two_tower"]["auc"] > 0.6
    assert (out / "RESULTS.md").exists()
