import numpy as np

from spamguard.preprocessing import HandcraftedFeatures, build_features, meta_features, normalize_text


def test_entities_are_masked():
    t = normalize_text("WIN £500 now! Call 09061701461 or visit www.prize.com, text 87121 or mail a@b.com")
    for tok in ("moneytoken", "phonetoken", "urltoken", "shortcodetoken", "emailtoken"):
        assert tok in t


def test_zero_width_and_case_normalised():
    assert normalize_text("FR\u200bEE") == "free"


def test_non_string_input_is_safe():
    assert normalize_text(None) == "" and normalize_text(123) == "numtoken"


def test_handcrafted_features_bounded_and_sparse():
    X = HandcraftedFeatures().fit_transform(["HELLO!!! 12345 http://x.com", "ok", ""])
    assert X.shape == (3, 10) and X.min() >= 0 and X.max() <= 1


def test_meta_features_columns():
    m = meta_features(["a b c", "WIN!"])
    assert m.loc[1, "exclaim_count"] == 1 and m.loc[0, "n_words"] == 3


def test_feature_union_output_width(tmp_path):
    from spamguard.config import DEFAULT_PARAMS
    f = build_features({**DEFAULT_PARAMS["features"], "min_df": 1})
    X = f.fit_transform(["free prize now", "see you soon", "call me later"])
    assert X.shape[0] == 3 and X.shape[1] > 10 and np.isfinite(X.toarray()).all()
