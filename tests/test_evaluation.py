import numpy as np

from spamguard.evaluation import (bootstrap_ci, choose_threshold, compute_metrics, expected_calibration_error,
                                  mcnemar_p, recall_at_precision)

RNG = np.random.default_rng(0)
Y = np.r_[np.ones(100), np.zeros(900)].astype(int)
P = np.clip(np.where(Y == 1, RNG.beta(8, 2, 1000), RNG.beta(1, 12, 1000)), 0, 1)


def test_perfect_classifier_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["accuracy"] == m["f1"] == m["mcc"] == m["pr_auc"] == m["roc_auc"] == 1.0 and m["fp"] == m["fn"] == 0


def test_confusion_counts_add_up():
    m = compute_metrics(Y, P, 0.5)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(Y)
    assert 0 <= m["ece"] <= 1 and 0 <= m["brier"] <= 1


def test_min_precision_threshold_is_respected():
    thr, _ = choose_threshold(Y, P, "min_precision", min_precision=0.95)
    assert compute_metrics(Y, P, thr)["precision"] >= 0.95


def test_threshold_strategies_run_and_trade_off():
    t_prec, _ = choose_threshold(Y, P, "min_precision", min_precision=0.99)
    t_f1, _ = choose_threshold(Y, P, "max_f1")
    t_cost, _ = choose_threshold(Y, P, "min_cost", cost_fp=10, cost_fn=1)
    assert 0 < t_f1 <= 1 and 0 < t_cost <= 1 and t_prec >= t_f1 - 1e-9
    assert recall_at_precision(Y, P, 0.9) >= recall_at_precision(Y, P, 0.99)


def test_unreachable_precision_falls_back():
    _, info = choose_threshold([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6], "min_precision", min_precision=0.99)
    assert "fell back" in info["note"]


def test_ece_perfectly_calibrated_is_small():
    p = RNG.uniform(0, 1, 20000); y = (RNG.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_bootstrap_ci_brackets_point_estimate():
    ci = bootstrap_ci(Y, P, 0.5, n=200)
    m = compute_metrics(Y, P, 0.5)
    assert ci["recall"]["low"] <= m["recall"] <= ci["recall"]["high"]


def test_mcnemar_identical_classifiers():
    pred = (P >= .5).astype(int)
    assert mcnemar_p(Y, pred, pred) == 1.0
