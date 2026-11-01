import numpy as np
import pandas as pd

from spamguard.monitoring import drift_report, psi

RNG = np.random.default_rng(1)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    a = RNG.normal(0, 1, 5000)
    assert psi(a, RNG.normal(0, 1, 5000)) < 0.05
    assert psi(a, RNG.normal(2, 1, 5000)) > 0.25


def test_drift_report_flags_shifted_feature():
    ref = pd.DataFrame({"n_chars": RNG.normal(80, 20, 1000), "score": RNG.beta(1, 8, 1000)})
    cur = pd.DataFrame({"n_chars": RNG.normal(160, 20, 1000), "score": RNG.beta(1, 8, 1000)})
    rep = drift_report(ref, cur)
    assert rep["status"] == "alert" and "n_chars" in rep["drifted_features"] and "score" not in rep["drifted_features"]


def test_drift_report_needs_enough_data():
    assert drift_report(pd.DataFrame({"n_chars": [1, 2]}), pd.DataFrame({"n_chars": [1]}))["status"] == "insufficient_data"
