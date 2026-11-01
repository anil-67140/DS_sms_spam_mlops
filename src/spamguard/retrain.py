"""Retraining with human feedback + champion/challenger gate.

    python -m spamguard.retrain            # dry-run: evaluates challenger, reports decision
    python -m spamguard.retrain --promote  # promotes the challenger only if it passes the gate

Safety rules:
  * val/test splits stay FROZEN, so champion and challenger are compared on identical data;
  * feedback texts that also occur in val/test are dropped (no leakage into the evaluation);
  * promotion requires PR-AUC not worse than `min_pr_auc_delta` AND FPR not above `max_fpr_increase`.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import evaluation as ev
from .config import get_paths, load_params
from .data import dataframe_fingerprint, load_splits
from .logging_utils import get_logger
from .registry import load_bundle, new_version, save_bundle
from .train import assemble_pipeline

log = get_logger("spamguard.retrain")


def _key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def load_feedback() -> pd.DataFrame:
    p = get_paths().feedback / "feedback.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["text", "label"])
    df = pd.read_json(p, lines=True).sort_values("ts")
    df["label"] = (df["true_label"] == "spam").astype(int)
    df = df.assign(_k=_key(df["text"])).drop_duplicates("_k", keep="last")  # latest human label wins
    return df[["text", "label"]].reset_index(drop=True)


def retrain(promote: bool = False, force: bool = False) -> dict:
    paths = get_paths()
    champ = load_bundle()
    meta, params = champ.metadata, champ.metadata.get("params") or load_params()
    rcfg, tcfg, thr_cfg = params["retrain"], params["training"], params["threshold"]
    fb = load_feedback()
    if len(fb) < rcfg["min_feedback"] and not force:
        res = {"status": "skipped", "reason": f"only {len(fb)} feedback rows (< {rcfg['min_feedback']})"}
        log.info(res); return res

    sp = load_splits(); tr, va, te = sp["train"], sp["val"], sp["test"]
    frozen = set(_key(va["text"])) | set(_key(te["text"]))
    fb = fb[~_key(fb["text"]).isin(frozen)]
    train_aug = pd.concat([tr, fb], ignore_index=True).drop_duplicates("text")
    log.info("Retraining on %d base + %d feedback rows", len(tr), len(fb))

    pipe = assemble_pipeline(meta["model_name"], meta["best_params"], meta["features"], meta["calibration"],
                             params["seed"], tcfg["n_jobs"])
    trval = pd.concat([train_aug, va], ignore_index=True)
    oof = cross_val_predict(assemble_pipeline(meta["model_name"], meta["best_params"], meta["features"],
                                              meta["calibration"], params["seed"], tcfg["n_jobs"]),
                            trval["text"], trval["label"], method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=params["seed"]))[:, 1]
    thr, info = ev.choose_threshold(trval["label"], oof, thr_cfg["strategy"], thr_cfg["min_precision"],
                                    thr_cfg["beta"], thr_cfg["cost_fp"], thr_cfg["cost_fn"])
    pipe.fit(train_aug["text"], train_aug["label"])

    yte = te["label"].to_numpy()
    m_new = ev.compute_metrics(yte, pipe.predict_proba(list(te["text"]))[:, 1], thr)
    m_old = ev.compute_metrics(yte, champ.pipeline.predict_proba(list(te["text"]))[:, 1], champ.threshold)
    ok = (m_new["pr_auc"] - m_old["pr_auc"] >= rcfg["min_pr_auc_delta"]
          and m_new["fpr"] <= m_old["fpr"] + rcfg["max_fpr_increase"])
    keys = ("precision", "recall", "f1", "pr_auc", "fpr")
    res = {"status": "evaluated", "feedback_rows_used": int(len(fb)), "passes_gate": bool(ok),
           "champion": {k: round(m_old[k], 4) for k in keys} | {"version": champ.version},
           "challenger": {k: round(m_new[k], 4) for k in keys}, "promoted": False}
    if promote and ok:
        new_meta = {**meta, "version": new_version(dataframe_fingerprint(train_aug)), "threshold": thr,
                    "threshold_policy": {**thr_cfg, **info}, "metrics_test": m_new,
                    "parent_version": champ.version, "feedback_rows": int(len(fb))}
        res["promoted"], res["new_version"] = True, save_bundle(pipe, new_meta, promote=True)
    (paths.reports / "retrain_report.json").write_text(json.dumps(res, indent=2))
    log.info("Retrain result: %s", json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--promote", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore min_feedback")
    a = ap.parse_args(); retrain(a.promote, a.force)
