import numpy as np

from recsys import experiment


def test_bucket_is_deterministic_and_balanced():
    users = [f"U{i}" for i in range(4000)]
    arms = [experiment.bucket(u) for u in users]
    assert arms == [experiment.bucket(u) for u in users]
    assert 0.45 < np.mean(arms) < 0.55


def test_detects_real_lift_and_not_noise():
    rng = np.random.default_rng(0)
    users = [f"U{i}" for i in range(3000) for _ in range(3)]
    control = rng.random(len(users)) < 0.30
    treatment = rng.random(len(users)) < 0.40
    real = experiment.simulate(users, control.astype(float), treatment.astype(float))
    assert real["significant"] and real["ci95_low"] > 0

    aa = experiment.simulate(users, control.astype(float), control.astype(float), salt="aa")
    assert aa["ci95_low"] < 0 < aa["ci95_high"]
