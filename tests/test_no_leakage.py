from spamguard.config import DEFAULT_PARAMS
from spamguard.models import build_pipeline


def test_vectoriser_is_fitted_on_train_only():
    """The original notebook fitted TF-IDF on ALL data. In our pipeline, test-only tokens must not enter the vocab."""
    train = ["free prize call now", "see you at lunch", "win cash today", "call me later please"] * 5
    y = [1, 0, 1, 0] * 5
    pipe = build_pipeline("multinomial_nb", {**DEFAULT_PARAMS["features"], "min_df": 1, "use_char": False}, n_jobs=1)
    pipe.fit(train, y)
    vocab = pipe.named_steps["features"].transformer_list[0][1].vocabulary_
    assert "zzzonlyintest" not in vocab
    pipe.predict_proba(["zzzonlyintest free"])          # unseen tokens handled gracefully
    assert "zzzonlyintest" not in vocab                  # ...and predicting never mutates the vocabulary


def test_trained_run_has_disjoint_splits(trained_home):
    from spamguard.data import load_splits
    s = load_splits()
    assert not (set(s["train"].text) & set(s["test"].text)) and not (set(s["val"].text) & set(s["test"].text))
