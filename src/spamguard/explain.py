"""Model-agnostic explanations (no extra dependencies).

* ``explain_message``  - occlusion / leave-one-token-out attribution. Works for *every* model in the
  zoo, including SVMs, forests and boosting. Positive weight = pushes toward SPAM.
* ``global_top_features`` - strongest n-grams of a linear model (what the model learned overall).
"""
from __future__ import annotations

import numpy as np

MAX_TOKENS = 80


def explain_message(predict_proba, text: str, top_k: int = 8) -> dict:
    tokens = text.split()[:MAX_TOKENS]
    if not tokens:
        return {"base_probability": 0.0, "tokens": [], "top_spam_tokens": [], "top_ham_tokens": []}
    variants = [text] + [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    probs = np.asarray(predict_proba(variants), dtype=float)
    base = float(probs[0])
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    weights = logit(probs[0]) - logit(probs[1:])   # log-odds scale: robust to saturated probabilities
    rows = [{"token": t, "weight": float(w), "delta_prob": float(base - q)} for t, w, q in zip(tokens, weights, probs[1:])]
    order = np.argsort(weights)
    return {
        "base_probability": base,
        "tokens": rows,
        "top_spam_tokens": [rows[i] for i in order[::-1][:top_k] if weights[i] > 1e-4],
        "top_ham_tokens": [rows[i] for i in order[:top_k] if weights[i] < -1e-4],
    }


def global_top_features(pipeline, k: int = 25):
    """Return (names, coefficients) for a fitted linear pipeline, or None."""
    clf = pipeline.named_steps["clf"]
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return None
    names = pipeline.named_steps["features"].get_feature_names_out()
    return np.asarray(names), np.asarray(coef).ravel()
