import pytest
from sklearn.dummy import DummyClassifier

from spamguard.registry import list_versions, load_bundle, promote_version, save_bundle


def _meta(v): return {"version": v, "threshold": 0.5, "model_name": "x", "metrics_test": {"pr_auc": .9}}


def test_roundtrip_promote_and_rollback(tmp_path):
    m = DummyClassifier().fit([[0], [1]], [0, 1])
    save_bundle(m, _meta("v1"), tmp_path); save_bundle(m, _meta("v2"), tmp_path)
    assert load_bundle(models_dir=tmp_path).version == "v2"
    promote_version("v1", tmp_path)                                  # rollback is one call
    assert load_bundle(models_dir=tmp_path).version == "v1" and len(list_versions(tmp_path)) == 2


def test_tampered_model_is_rejected(tmp_path):
    save_bundle(DummyClassifier().fit([[0], [1]], [0, 1]), _meta("v1"), tmp_path)
    with open(tmp_path / "versions" / "v1" / "model.joblib", "ab") as f: f.write(b"tamper")
    with pytest.raises(RuntimeError, match="Checksum"):
        load_bundle("v1", tmp_path)


def test_unknown_version_cannot_be_promoted(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote_version("nope", tmp_path)
