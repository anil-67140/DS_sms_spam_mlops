"""Metrics, threshold optimisation, calibration, statistical tests, robustness and plots."""
from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             fbeta_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

EPS = 1e-15


# =============================================================== metrics
def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Every metric a spam-filter owner cares about, at one operating point."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": div(tn, tn + fp),
        "npv": div(tn, tn + fn),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "f0_5": fbeta_score(y, pred, beta=0.5, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "pr_auc": average_precision_score(y, p) if both else float("nan"),
        "roc_auc": roc_auc_score(y, p) if both else float("nan"),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def recall_at_precision(y, p, min_precision: float) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


# =============================================================== thresholds
def choose_threshold(y, proba, strategy: str = "min_precision", min_precision: float = 0.99,
                     beta: float = 0.5, cost_fp: float = 5.0, cost_fn: float = 1.0) -> tuple[float, dict]:
    """Pick the operating point on *validation* data according to a business policy."""
    y, p = np.asarray(y).astype(int), np.asarray(proba, dtype=float)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    info = {"strategy": strategy}

    def fbeta_best(b):
        f = (1 + b**2) * prec * rec / np.maximum(b**2 * prec + rec, EPS)
        return int(np.nanargmax(f))

    if strategy == "min_precision":
        ok = np.where(prec >= min_precision)[0]
        if len(ok):
            best_rec = rec[ok].max()
            cands = ok[rec[ok] >= best_rec - 1e-12]
            i = int(cands[-1])  # highest threshold among ties => most conservative
            info["note"] = f"max recall subject to precision >= {min_precision}"
        else:
            i = fbeta_best(beta)
            info["note"] = f"precision >= {min_precision} unreachable; fell back to max F{beta}"
    elif strategy == "max_f1":
        i = fbeta_best(1.0)
    elif strategy == "max_fbeta":
        i = fbeta_best(beta)
    elif strategy == "min_cost":
        fpr_, tpr_, t_ = roc_curve(y, p)
        P = y.sum(); N = len(y) - P
        cost = cost_fp * fpr_ * N + cost_fn * (1 - tpr_) * P
        j = int(np.argmin(cost[1:])) + 1  # skip the inf threshold
        t = float(t_[j]); info.update(cost=float(cost[j]), note="min expected cost")
        return float(min(max(t, 1e-6), 1.0)), info
    else:
        raise ValueError(f"unknown strategy {strategy}")
    t = float(thr[i])
    info.update(precision_at_threshold=float(prec[i]), recall_at_threshold=float(rec[i]))
    return t, info


# =============================================================== statistics
def bootstrap_ci(y, proba, threshold, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs (stratified resampling) - small test sets are noisy!"""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(proba)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keys = ["precision", "recall", "f1", "mcc", "pr_auc", "roc_auc", "fpr"]
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = compute_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi)),
                "mean": float(np.mean(v))} for k, v in samples.items()}


def mcnemar_p(y, pred_a, pred_b) -> float:
    """Exact McNemar test: are two classifiers' error patterns significantly different?"""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    b01 = int(((a == y) & (b != y)).sum()); b10 = int(((a != y) & (b == y)).sum())
    n = b01 + b10
    return 1.0 if n == 0 else float(binomtest(min(b01, b10), n, 0.5).pvalue)


# =============================================================== latency
def measure_latency_ms(predict_fn, texts, n: int = 40) -> float:
    """Median single-message latency (ms) - what an API call pays."""
    texts = list(texts)[:n]
    predict_fn(texts[:1])  # warm-up
    times = []
    for t in texts:
        s = time.perf_counter(); predict_fn([t]); times.append((time.perf_counter() - s) * 1000)
    return float(np.median(times))


# =============================================================== robustness
def _leet(t):
    return t.translate(str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}))


def _spaced(t, rng):
    return " ".join(" ".join(w) if len(w) > 4 and rng.random() < .4 else w for w in t.split())


def _zero_width(t, rng):
    return "".join(c + ("\u200b" if c.isalpha() and rng.random() < .15 else "") for c in t)


def _caps_noise(t, rng):
    return "".join(c.upper() if rng.random() < .3 else c for c in t)


def _punct_noise(t, rng):
    return " ".join(w + (rng.choice([".", "*", "-", "_"]) if rng.random() < .3 else "") for w in t.split())


def robustness_report(predict_proba, texts, y, threshold, seed: int = 42) -> pd.DataFrame:
    """Recall on SPAM after adversarial obfuscation (spammers actively evade filters)."""
    rng = random.Random(seed)
    spam = [t for t, l in zip(texts, y) if l == 1]
    attacks = {"clean (baseline)": lambda t: t, "leetspeak": _leet,
               "letter spacing": lambda t: _spaced(t, rng), "zero-width chars": lambda t: _zero_width(t, rng),
               "random caps": lambda t: _caps_noise(t, rng), "punctuation noise": lambda t: _punct_noise(t, rng)}
    rows = []
    for name, fn in attacks.items():
        p = predict_proba([fn(t) for t in spam])
        rows.append({"attack": name, "spam_recall": float((p >= threshold).mean()),
                     "mean_spam_score": float(p.mean())})
    return pd.DataFrame(rows)


def slice_report(texts, y, proba, threshold) -> pd.DataFrame:
    """Metrics by message length bucket - catches hidden weak spots."""
    s = pd.Series([len(t) for t in texts])
    buckets = pd.cut(s, [-1, 40, 100, 160, 100000], labels=["<=40", "41-100", "101-160", ">160"])
    rows = []
    for b in buckets.cat.categories:
        m = (buckets == b).to_numpy()
        if m.sum() < 5:
            continue
        r = compute_metrics(np.asarray(y)[m], np.asarray(proba)[m], threshold)
        rows.append({"length_bucket": b, "n": int(m.sum()), "spam_rate": float(np.asarray(y)[m].mean()),
                     "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                     "fp": r["fp"], "fn": r["fn"]})
    return pd.DataFrame(rows)


# =============================================================== plots
def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_class_balance(df, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    c = df["label"].value_counts().sort_index()
    ax.bar(["ham", "spam"], c.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v}\n({v / c.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Class balance"); ax.set_ylabel("messages")
    return _save(fig, path)


def plot_length_distributions(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax in zip(["n_chars", "n_words"], axes):
        for lab, color, name in [(0, "#4C78A8", "ham"), (1, "#E45756", "spam")]:
            ax.hist(df.loc[df["label"] == lab, col].clip(upper=df[col].quantile(.99)), bins=40,
                    alpha=.6, color=color, label=name, density=True)
        ax.set_title(col); ax.legend()
    return _save(fig, path)


def plot_confusion(y, proba, threshold, title, path):
    cm = confusion_matrix(y, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, norm in zip(axes, [False, True]):
        data = cm / cm.sum(axis=1, keepdims=True) if norm else cm
        ax.imshow(data, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{data[i, j]:.1%}" if norm else int(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1], ["ham", "spam"]); ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("normalised" if norm else "counts")
    fig.suptitle(f"{title} (threshold={threshold:.3f})")
    return _save(fig, path)


def _curve_plot(curves: dict, kind: str, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, (y, p) in curves.items():
        if kind == "roc":
            x_, y_, _ = roc_curve(y, p); lab = f"{name} (AUC={roc_auc_score(y, p):.4f})"
        else:
            y_, x_, _ = precision_recall_curve(y, p); lab = f"{name} (AP={average_precision_score(y, p):.4f})"
        ax.plot(x_, y_, lw=1.4, label=lab)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlim(0, .2); ax.set_ylim(.8, 1.005)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC (zoomed)")
    else:
        ax.set_xlim(.5, 1.005); ax.set_ylim(.5, 1.005)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall (zoomed)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_roc_all(curves, path): return _curve_plot(curves, "roc", path)
def plot_pr_all(curves, path): return _curve_plot(curves, "pr", path)


def plot_calibration_all(curves: dict, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, (y, p) in curves.items():
        try:
            frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
            ax.plot(mean, frac, marker="o", ms=3, lw=1, label=name)
        except ValueError:
            continue
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed spam fraction")
    ax.set_title("Reliability diagram (calibration)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_threshold_sweep(y, proba, chosen, path):
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], EPS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, prec[:-1], label="precision"); ax.plot(thr, rec[:-1], label="recall"); ax.plot(thr, f1, label="F1")
    ax.axvline(chosen, color="k", ls="--", label=f"chosen = {chosen:.3f}")
    ax.set_xlabel("decision threshold"); ax.set_title("Threshold optimisation (validation set)")
    ax.legend(); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_score_distribution(y, proba, threshold, path):
    y, p = np.asarray(y), np.asarray(proba)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(p[y == 0], bins=50, alpha=.6, color="#4C78A8", label="ham", log=True)
    ax.hist(p[y == 1], bins=50, alpha=.6, color="#E45756", label="spam", log=True)
    ax.axvline(threshold, color="k", ls="--"); ax.set_xlabel("P(spam)"); ax.set_title("Score distribution (log-count)")
    ax.legend(); return _save(fig, path)


def plot_leaderboard(df: pd.DataFrame, path):
    d = df[df["model"] != "dummy_prior"].sort_values("pr_auc")
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, .38 * len(d))), sharey=True)
    for ax, col, color in zip(axes, ["pr_auc", "f1", "recall"], ["#4C78A8", "#54A24B", "#E45756"]):
        ax.barh(d["model"], d[col], color=color)
        ax.set_xlim(max(0.0, d[col].min() - .03), 1.0); ax.set_title(col + " (test)")
        for i, v in enumerate(d[col]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    return _save(fig, path)


def plot_metric_heatmap(df: pd.DataFrame, cols, path):
    d = df.set_index("model")[cols]
    fig, ax = plt.subplots(figsize=(1 + .9 * len(cols), .45 * len(d) + 1.5))
    im = ax.imshow(d.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(d)), d.index)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im); ax.set_title("All models x key metrics (test)")
    return _save(fig, path)


def plot_speed_vs_quality(df, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["latency_ms"], df["f1"], s=60)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["latency_ms"], r["f1"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("median latency / message (ms, log)"); ax.set_ylabel("F1 (test)")
    ax.set_title("Quality vs serving cost"); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_top_features(names, coefs, path, k=20):
    order = np.argsort(coefs)
    sel = np.concatenate([order[:k], order[-k:]])
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([names[i] for i in sel], coefs[sel], color=["#4C78A8"] * k + ["#E45756"] * k)
    ax.set_title("Strongest ham (blue) / spam (red) features - Logistic Regression")
    return _save(fig, path)


def plot_robustness(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["attack"], df["spam_recall"], color="#F58518")
    ax.set_xlim(0, 1.02); ax.set_xlabel("spam recall"); ax.set_title("Robustness to obfuscation attacks")
    for i, v in enumerate(df["spam_recall"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    return _save(fig, path)
