"""Text normalisation + feature engineering.

Everything here is *stateless* or lives inside an sklearn ``Pipeline``. Because the
vectorisers are fitted inside the pipeline, cross-validation and train/test separation
cannot leak vocabulary / IDF statistics (the original notebook fitted TF-IDF on the
full dataset before splitting).
"""
from __future__ import annotations

import re
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# ----------------------------------------------------------------------------- regexes
_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|co\.uk|info|biz|ly|xyz)\b\S*)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"(?:[£$€]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|pounds?|dollars?|rs|pkr)\b)", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s\-().]{8,}\d)(?!\w)")
_SHORTCODE = re.compile(r"(?<!\d)\d{4,6}(?!\d)")
_NUMBER = re.compile(r"\d+")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_NON_ALNUM = re.compile(r"[^a-z0-9_\s]")
_WS = re.compile(r"\s+")

# Leet / homoglyph folding: "fr33", "w1n", "cl@im" -> "free", "win", "claim" (applied inside words only)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize_text(text: str, stem: bool = False) -> str:
    """Lowercase, mask volatile entities (url/email/phone/money/shortcode/number)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _ZERO_WIDTH.sub("", text).lower()
    t = _EMAIL.sub(" emailtoken ", t)   # before URL: the bare-domain URL rule would swallow emails
    t = _URL.sub(" urltoken ", t)
    t = _MONEY.sub(" moneytoken ", t)
    t = _PHONE.sub(" phonetoken ", t)
    t = _SHORTCODE.sub(" shortcodetoken ", t)
    t = _NUMBER.sub(" numtoken ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if stem:
        from nltk.stem.porter import PorterStemmer  # no corpus download required
        ps = PorterStemmer()
        t = " ".join(ps.stem(w) for w in t.split())
    return t


def fold_obfuscation(text: str) -> str:
    """Optional helper used by robustness tests: map leetspeak back to letters inside words."""
    return re.sub(r"(?<=[a-z])[013457@$](?=[a-z])|(?<=[013457@$])[a-z](?=[01345@$7])",
                  lambda m: m.group(0).translate(_LEET), text.lower())


# ------------------------------------------------------------------ hand-crafted features
META_COLUMNS = ["n_chars", "n_words", "digit_ratio", "upper_ratio", "punct_ratio",
                "exclaim_count", "has_url", "has_phone", "has_money", "has_shortcode"]


def meta_features(texts: Iterable[str]) -> pd.DataFrame:
    """Interpretable per-message statistics (also used for drift monitoring)."""
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ("" if t is None else str(t))
        n = max(len(t), 1)
        rows.append({
            "n_chars": len(t),
            "n_words": len(t.split()),
            "digit_ratio": sum(c.isdigit() for c in t) / n,
            "upper_ratio": sum(c.isupper() for c in t) / n,
            "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
            "exclaim_count": t.count("!"),
            "has_url": int(bool(_URL.search(t))),
            "has_phone": int(bool(_PHONE.search(t))),
            "has_money": int(bool(_MONEY.search(t))),
            "has_shortcode": int(bool(_SHORTCODE.search(t))),
        })
    return pd.DataFrame(rows, columns=META_COLUMNS)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Bounded [0, 1] numeric features -> no fitted scaler needed (so nothing to leak)."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):  # stateless => always "fitted"
        return True

    def transform(self, X):
        m = meta_features(list(X))
        out = np.column_stack([
            np.log1p(m["n_chars"]) / np.log1p(1000),
            np.log1p(m["n_words"]) / np.log1p(200),
            m["digit_ratio"].clip(0, 1),
            m["upper_ratio"].clip(0, 1),
            m["punct_ratio"].clip(0, 1),
            np.log1p(m["exclaim_count"]) / np.log1p(10),
            m["has_url"], m["has_phone"], m["has_money"], m["has_shortcode"],
        ]).astype(np.float64)
        return sparse.csr_matrix(np.clip(out, 0.0, 1.0))

    def get_feature_names_out(self, input_features=None):
        return np.array(["meta_" + c for c in META_COLUMNS])


def build_features(cfg: dict) -> FeatureUnion:
    """word TF-IDF (1-2gram) + char TF-IDF (2-4gram, robust to obfuscation) + meta features."""
    prep = partial(normalize_text, stem=bool(cfg.get("stem", False)))
    parts = [("word", TfidfVectorizer(
        preprocessor=prep, ngram_range=tuple(cfg["word_ngram"]), min_df=cfg["min_df"],
        max_features=cfg["word_max_features"], sublinear_tf=True, stop_words=cfg.get("stop_words"),
        dtype=np.float64))]
    if cfg.get("use_char", True):
        parts.append(("char", TfidfVectorizer(
            preprocessor=prep, analyzer="char_wb", ngram_range=tuple(cfg["char_ngram"]),
            min_df=cfg["min_df"], max_features=cfg["char_max_features"], sublinear_tf=True,
            dtype=np.float64)))
    if cfg.get("use_handcrafted", True):
        parts.append(("meta", HandcraftedFeatures()))
    return FeatureUnion(parts)


def oov_rate(texts: Iterable[str], word_vectorizer: TfidfVectorizer) -> np.ndarray:
    """Share of tokens unseen at training time (a cheap, effective vocabulary-drift signal)."""
    analyzer, vocab = word_vectorizer.build_analyzer(), word_vectorizer.vocabulary_
    out = []
    for t in texts:
        toks = [w for w in analyzer(t) if " " not in w]  # unigrams only
        out.append(0.0 if not toks else sum(w not in vocab for w in toks) / len(toks))
    return np.asarray(out)
