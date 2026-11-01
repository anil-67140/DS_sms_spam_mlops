import json

import pytest
from fastapi.testclient import TestClient

from spamguard.api import create_app
from spamguard.explain import explain_message


def test_training_produced_all_artifacts(trained_home):
    home, summary = trained_home
    for f in ("reports/leaderboard_full.csv", "reports/metrics.json", "reports/MODEL_CARD.md", "reports/summary.json",
              "reports/reference_features.csv", "reports/figures/pr_all_models.png",
              "reports/figures/confusion_champion.png", "models/champion.json", "checkpoints/multinomial_nb.joblib"):
        assert (home / f).exists(), f
    assert summary["champion"] != "dummy_prior" and summary["metrics_test"]["pr_auc"] > 0.9


def test_second_run_resumes_from_checkpoints(trained_home, synthetic_csv, capsys):
    """Re-running must skip every finished model (this is what protects you from Colab disconnects)."""
    from spamguard.config import load_params
    from spamguard.data import prepare
    from spamguard.train import fit_or_resume
    params = load_params(overrides={"data": {"source": "local", "local_path": str(synthetic_csv)},
                                    "features": {"min_df": 1}, "training": {"n_jobs": 1, "n_iter": 2, "heavy_n_iter": 1}})
    splits = prepare(params)
    info = fit_or_resume("multinomial_nb", params, splits, splits["manifest"], resume=True)
    assert "pipeline" in info and info["fit_seconds"] >= 0
    # a changed hyper-parameter budget must invalidate the checkpoint key, not silently reuse stale results
    from spamguard.train import _run_key
    p2 = load_params(overrides={"training": {"n_iter": 99}})
    assert _run_key(splits["manifest"], params, "multinomial_nb") != _run_key(splits["manifest"], p2, "multinomial_nb")


def test_explanation_flags_spam_tokens(predictor):
    ex = explain_message(predictor.predict_proba, "WINNER claim your free prize call 09061701234 now see you tomorrow")
    assert ex["top_spam_tokens"] and len(ex["tokens"]) == 11


@pytest.fixture()
def client(predictor):
    with TestClient(create_app(predictor, enable_ui=False)) as c:
        yield c


def test_health_ready_info(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["status"] == "ready"
    assert 0 < client.get("/model-info").json()["threshold"] <= 1


def test_predict_spam_and_ham(client):
    spam = client.post("/predict", json={"text": "WINNER claim your free prize call 09061701234 now"}).json()
    ham = client.post("/predict", json={"text": "are we meeting tomorrow for dinner"}).json()
    assert spam["label"] == "spam" and ham["label"] == "ham" and spam["spam_probability"] > ham["spam_probability"]
    assert spam["model_version"] and "prediction_id" in spam


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "x" * 2001}, {}, {"text": 5}])
def test_predict_input_validation(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_batch_and_limits(client):
    r = client.post("/predict/batch", json={"texts": ["free prize now call 09061701234", "ok see you"]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["ok", "   "]}).status_code == 422


def test_feedback_is_persisted_and_retrain_skips_when_too_few(client, trained_home):
    home, _ = trained_home
    assert client.post("/feedback", json={"text": "new scam msg", "true_label": "spam"}).status_code == 201
    lines = (home / "data" / "feedback" / "feedback.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[-1])["true_label"] == "spam"
    from spamguard.retrain import retrain
    assert retrain()["status"] == "skipped"


def test_api_key_enforced(predictor, monkeypatch):
    monkeypatch.setenv("SPAMGUARD_API_KEY", "s3cret")
    with TestClient(create_app(predictor, enable_ui=False)) as c:
        assert c.post("/predict", json={"text": "hi"}).status_code == 401
        assert c.post("/predict", json={"text": "hi"}, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200   # probes stay open


def test_prediction_log_has_no_raw_text(client, trained_home):
    home, _ = trained_home
    client.post("/predict", json={"text": "super private message 12345"})
    log = (home / "logs" / "predictions.jsonl").read_text()
    assert "super private" not in log and "text_hash" in log


def test_metrics_endpoint(client):
    client.post("/predict", json={"text": "hello there"})
    assert "spamguard_predictions_total" in client.get("/metrics").text


def test_no_model_gives_503(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAMGUARD_HOME", str(tmp_path))
    with TestClient(create_app(None, enable_ui=False)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503 and c.post("/predict", json={"text": "hi"}).status_code == 503


def test_retrain_gate_with_feedback(predictor, trained_home):
    """Human feedback -> challenger -> champion/challenger gate on the frozen test set."""
    from spamguard.retrain import retrain
    from tests.conftest import make_dataset
    extra = make_dataset(80, seed=123)
    for text, y in zip(extra["text"], extra["label"]):
        predictor.record_feedback("fb " + text, "spam" if y else "ham", source="test")
    res = retrain(promote=True)
    assert res["status"] == "evaluated" and res["feedback_rows_used"] >= 50
    assert set(res["challenger"]) >= {"precision", "recall", "f1", "pr_auc", "fpr"}
    if res["passes_gate"]:
        assert res["promoted"] and res["new_version"] != res["champion"]["version"]
