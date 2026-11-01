import pandas as pd

from spamguard.data import clean, standardize, stratified_split


def test_standardize_uci_columns_and_labels():
    df = standardize(pd.DataFrame({"v1": ["ham", "spam", "weird"], "v2": ["a", "b", "c"], "Unnamed: 2": [1, 2, 3]}))
    assert list(df["label"]) == [0, 1] and set(df.columns) == {"text", "label"}


def test_clean_removes_duplicates_and_conflicts():
    df = pd.DataFrame({"text": ["Hello  there", "hello there", "dup", "dup", "x"], "label": [0, 0, 1, 0, 1]})
    out, rep = clean(df)
    assert rep["conflicting_label_texts_removed"] == 1
    assert out["text"].str.lower().str.replace(r"\s+", " ", regex=True).is_unique
    assert "dup" not in out["text"].tolist()


def test_clean_drops_empty_and_overlong():
    out, _ = clean(pd.DataFrame({"text": ["", "  ", "ok", "x" * 5000], "label": [0, 1, 0, 1]}), max_chars=1000)
    assert out["text"].tolist() == ["ok"]


def test_split_is_stratified_disjoint_and_complete():
    df = pd.DataFrame({"text": [f"m{i}" for i in range(1000)], "label": [1 if i % 8 == 0 else 0 for i in range(1000)]})
    tr, va, te = stratified_split(df, 0.15, 0.15, 42)
    assert len(tr) + len(va) + len(te) == 1000
    assert not (set(tr.text) & set(va.text)) and not (set(tr.text) & set(te.text)) and not (set(va.text) & set(te.text))
    for part in (tr, va, te):
        assert abs(part["label"].mean() - df["label"].mean()) < 0.01
