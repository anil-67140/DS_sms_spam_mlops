"""Drift & performance monitoring.

* Data drift: PSI + two-sample KS on interpretable message features, the model's score
  distribution and out-of-vocabulary rate, compared with a *reference* captured at training time.
* Concept/performance drift: precision/recall on messages that humans labelled via the feedback loop.
* Evidently AI HTML report (optional dependency) for a rich dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import get_paths, load_params
from .evaluation import compute_metrics
from .logging_utils import get_logger

log = get_logger("spamguard.monitoring")

DRIFT_FEATURES = ["n_chars", "n_words", "digit_ratio", "upper_ratio", "punct_ratio", "exclaim_count",
                  "has_url", "has_phone", "has_money", "has_shortcode", "oov_rate", "score"]


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index (<0.1 stable, 0.1-0.25 moderate, >0.25 major shift)."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if len(np.unique(ref)) <= 3:  # binary / tiny-cardinality features
        cats = np.unique(np.concatenate([ref, cur]))
        e = np.array([(ref == c).mean() for c in cats]); a = np.array([(cur == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(ref, edges)[0] / len(ref); a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or load_params()
    mcfg = params["monitoring"]
    if len(current) < mcfg["min_samples"]:
        return {"status": "insufficient_data", "n_current": int(len(current)),
                "min_samples": mcfg["min_samples"]}
    feats = {}
    for col in [c for c in DRIFT_FEATURES if c in reference and c in current]:
        v = psi(reference[col], current[col])
        ks = ks_2samp(reference[col], current[col])
        status = "alert" if v > mcfg["psi_alert"] else "warn" if v > mcfg["psi_warn"] else "ok"
        feats[col] = {"psi": round(v, 4), "ks_stat": round(float(ks.statistic), 4),
                      "ks_pvalue": float(ks.pvalue), "status": status,
                      "ref_mean": round(float(reference[col].mean()), 4),
                      "cur_mean": round(float(current[col].mean()), 4)}
    statuses = [f["status"] for f in feats.values()]
    overall = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    out = {"status": overall, "n_reference": int(len(reference)), "n_current": int(len(current)),
           "features": feats,
           "drifted_features": [k for k, f in feats.items() if f["status"] != "ok"]}
    if "score" in reference and "score" in current:
        out["spam_rate_reference"] = float((reference["score"] >= 0.5).mean())
        out["spam_rate_current"] = float((current["score"] >= 0.5).mean())
    return out


def evidently_html(reference: pd.DataFrame, current: pd.DataFrame, out_path: Path) -> Path | None:
    """Rich HTML drift dashboard. Best-effort: never breaks the pipeline if Evidently is absent."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        cols = [c for c in DRIFT_FEATURES if c in reference and c in current]
        snap = Report([DataDriftPreset()]).run(current[cols], reference[cols])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(out_path))
        return out_path
    except Exception as exc:  # noqa: BLE001
        log.warning("Evidently report skipped: %s", exc)
        return None


def load_reference() -> pd.DataFrame:
    return pd.read_csv(get_paths().reports / "reference_features.csv")


def load_prediction_log(limit: int | None = None) -> pd.DataFrame:
    p = get_paths().logs / "predictions.jsonl"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_json(p, lines=True)
    return df.tail(limit) if limit else df


def performance_from_feedback(threshold: float | None = None) -> dict:
    """Live precision/recall on human-labelled messages (needs ground truth => feedback loop)."""
    p = get_paths().feedback / "feedback.jsonl"
    if not p.exists():
        return {"status": "no_feedback"}
    df = pd.read_json(p, lines=True)
    df = df[df["score"].notna()] if "score" in df else df
    if len(df) < 20 or df["true_label"].nunique() < 2:
        return {"status": "insufficient_feedback", "n": int(len(df))}
    y = (df["true_label"] == "spam").astype(int).to_numpy()
    m = compute_metrics(y, df["score"].to_numpy(), threshold if threshold is not None else 0.5)
    return {"status": "ok", "n": int(len(df)),
            **{k: round(m[k], 4) for k in ("precision", "recall", "f1", "fpr", "fnr", "pr_auc")}}


def save_report(report: dict, name: str = "drift_report.json") -> Path:
    path = get_paths().reports / name
    path.write_text(json.dumps(report, indent=2))
    return path


def run_check(window: int | None = 5000) -> dict:
    """CLI entry (`python -m spamguard.monitoring`): compare recent traffic with the reference."""
    ref, cur = load_reference(), load_prediction_log(window)
    rep = drift_report(ref, cur)
    save_report(rep)
    if rep["status"] != "insufficient_data":
        evidently_html(ref, cur, get_paths().reports / "drift_report.html")
    log.info("Drift status: %s (%s)", rep["status"], rep.get("drifted_features"))
    return rep


if __name__ == "__main__":
    run_check()
