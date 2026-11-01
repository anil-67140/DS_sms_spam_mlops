"""End-to-end, leakage-safe, resumable training + benchmarking pipeline.

    python -m spamguard.train                # full run (resumes from checkpoints automatically)
    python -m spamguard.train --fast         # tiny search budget (smoke test / CI)
    python -m spamguard.train --no-resume    # ignore checkpoints

Protocol (the test set is never used for any decision):
  train  -> fit + hyper-parameter search (stratified CV, PR-AUC)
  val    -> choose champion, calibration method and decision threshold
  test   -> single final, unbiased report (with bootstrap CIs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from . import evaluation as ev
from .config import get_paths, load_params
from .data import prepare
from .explain import global_top_features
from .logging_utils import get_logger
from .models import build_pipeline, model_zoo
from .preprocessing import meta_features, oov_rate
from .registry import save_bundle

warnings.filterwarnings("ignore")
log = get_logger("spamguard.train")

SINGLE_THREAD_SEARCH = {"random_forest", "extra_trees", "lightgbm", "xgboost", "knn_cosine"}


# ----------------------------------------------------------------------------- helpers
def _py(v):
    """numpy -> plain python (so params round-trip through JSON)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    return v


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _code_hash() -> str:
    """Hash of the feature/model code: editing preprocessing or the zoo invalidates old checkpoints."""
    pkg = Path(__file__).parent
    return hashlib.sha1(b"".join((pkg / f).read_bytes() for f in ("preprocessing.py", "models.py"))).hexdigest()[:8]


def _run_key(manifest: dict, params: dict, name: str) -> str:
    blob = json.dumps({"fp": manifest["fingerprint"], "code": _code_hash(), "feat": params["features"], "seed": params["seed"],
                       "train": {k: params["training"][k] for k in ("cv_folds", "n_iter", "heavy_n_iter", "scoring")},
                       "val": params["data"]["val_size"], "test": params["data"]["test_size"],
                       "model": name, "sk": sklearn.__version__}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@contextmanager
def _mlflow(experiment: str, enabled: bool):
    """Best-effort MLflow: experiment tracking must never be able to break training."""
    if not enabled:
        yield None; return
    try:
        import os
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{get_paths().home}/mlflow.db"))
        mlflow.set_experiment(experiment)
        yield mlflow
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow disabled: %s", exc); yield None


def assemble_pipeline(model_name, best_params, features, calibration, seed, n_jobs):
    """Re-create a (possibly calibrated) champion pipeline from metadata - used by retraining."""
    pipe = build_pipeline(model_name, features, seed, n_jobs, best_params or None)
    if calibration in ("sigmoid", "isotonic"):
        pipe.steps[-1] = ("clf", CalibratedClassifierCV(clone(pipe.named_steps["clf"]), method=calibration, cv=5))
    return pipe


# ----------------------------------------------------------------------------- one model
def fit_or_resume(name, params, splits, manifest, resume=True) -> dict:
    paths = get_paths()
    base = paths.checkpoints / name
    key = _run_key(manifest, params, name)
    if resume and base.with_suffix(".json").exists() and base.with_suffix(".joblib").exists():
        info = json.loads(base.with_suffix(".json").read_text())
        if info.get("key") == key:
            log.info("[%s] checkpoint found -> skipping training", name)
            info["pipeline"] = joblib.load(base.with_suffix(".joblib"))
            z = np.load(base.with_suffix(".npz"))
            info["val_proba"], info["test_proba"] = z["val"], z["test"]
            return info

    t = params["training"]; seed = params["seed"]; spec = model_zoo(seed, t["n_jobs"])[name]
    Xtr, ytr = splits["train"]["text"], splits["train"]["label"].to_numpy()
    pipe = build_pipeline(name, params["features"], seed, t["n_jobs"])
    cv = StratifiedKFold(t["cv_folds"], shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    if spec.space:
        n_iter = t["heavy_n_iter"] if spec.heavy else t["n_iter"]
        search = RandomizedSearchCV(pipe, spec.space, n_iter=n_iter, cv=cv, scoring=t["scoring"],
                                    n_jobs=1 if name in SINGLE_THREAD_SEARCH else t["n_jobs"],
                                    random_state=seed, refit=True, error_score="raise")
        search.fit(Xtr, ytr)
        best, best_params = search.best_estimator_, _py(search.best_params_)
        i = search.best_index_
        cv_mean, cv_std = float(search.best_score_), float(search.cv_results_["std_test_score"][i])
    else:
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=t["scoring"], n_jobs=1)
        best = pipe.fit(Xtr, ytr); best_params = {}
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
    fit_s = time.perf_counter() - t0

    pp = lambda texts: best.predict_proba(list(texts))[:, 1]  # noqa: E731
    val_p, test_p = pp(splits["val"]["text"]), pp(splits["test"]["text"])
    latency = ev.measure_latency_ms(pp, splits["test"]["text"])
    joblib.dump(best, base.with_suffix(".joblib"), compress=3)
    np.savez_compressed(base.with_suffix(".npz"), val=val_p, test=test_p)
    info = {"key": key, "name": name, "family": spec.family, "best_params": best_params,
            "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std, "fit_seconds": round(fit_s, 2),
            "latency_ms": round(latency, 3),
            "size_mb": round(base.with_suffix(".joblib").stat().st_size / 1e6, 2)}
    _atomic_json(base.with_suffix(".json"), info)   # json written LAST => marks checkpoint complete
    log.info("[%s] cv PR-AUC %.4f±%.4f | fit %.1fs | %.1f ms/msg | %.1f MB", name, cv_mean, cv_std,
             fit_s, latency, info["size_mb"])
    info.update(pipeline=best, val_proba=val_p, test_proba=test_p)
    return info


# ----------------------------------------------------------------------------- main
def run(params: dict | None = None, resume: bool = True, fast: bool = False, models: list[str] | None = None,
        promote: bool = True, use_mlflow: bool = True) -> dict:
    params = params or load_params()
    if fast:
        params["training"].update(n_iter=2, heavy_n_iter=1)
        params["bootstrap"]["n"] = 200
    paths, seed, tcfg, thr_cfg = get_paths(), params["seed"], params["training"], params["threshold"]
    t_start = time.perf_counter()

    # 1) data ------------------------------------------------------------------------------
    splits = prepare(params); manifest = splits["manifest"]
    tr, va, te = splits["train"], splits["val"], splits["test"]
    ytr, yva, yte = (d["label"].to_numpy() for d in (tr, va, te))
    ev.plot_class_balance(tr, paths.figures / "eda_class_balance.png")
    ev.plot_length_distributions(pd.concat([tr, meta_features(tr["text"])], axis=1),
                                 paths.figures / "eda_length_distributions.png")

    # 2) fit / resume every candidate ------------------------------------------------------
    zoo = model_zoo(seed, tcfg["n_jobs"])
    wanted = models or (list(zoo) if tcfg["models"] == "all" else tcfg["models"])
    results = {}
    for name in wanted:
        try:
            results[name] = fit_or_resume(name, params, splits, manifest, resume)
        except Exception as exc:  # noqa: BLE001  one bad model must not kill a 40-min run
            log.error("[%s] FAILED: %s", name, exc)
    # 3) leaderboard -----------------------------------------------------------------------
    rows, ops = [], {}
    for name, r in results.items():
        thr, thr_info = ev.choose_threshold(yva, r["val_proba"], thr_cfg["strategy"], thr_cfg["min_precision"],
                                            thr_cfg["beta"], thr_cfg["cost_fp"], thr_cfg["cost_fn"])
        mv, mt, m05 = (ev.compute_metrics(yva, r["val_proba"], thr), ev.compute_metrics(yte, r["test_proba"], thr),
                       ev.compute_metrics(yte, r["test_proba"], 0.5))
        ops[name] = thr
        rows.append({"model": name, "family": r["family"], "cv_pr_auc": r["cv_pr_auc_mean"],
                     "cv_pr_auc_std": r["cv_pr_auc_std"],
                     "val_pr_auc": mv["pr_auc"], "val_f1": mv["f1"],
                     "val_recall_at_precision": ev.recall_at_precision(yva, r["val_proba"], thr_cfg["min_precision"]),
                     **{k: mt[k] for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                                           "f1", "f2", "f0_5", "mcc", "cohen_kappa", "fpr", "fnr", "pr_auc",
                                           "roc_auc", "log_loss", "brier", "ece", "tn", "fp", "fn", "tp")},
                     "threshold": thr, "f1_at_0.5": m05["f1"], "precision_at_0.5": m05["precision"],
                     "recall_at_0.5": m05["recall"], "accuracy_at_0.5": m05["accuracy"],
                     "latency_ms": r["latency_ms"], "size_mb": r["size_mb"], "fit_seconds": r["fit_seconds"],
                     "best_params": json.dumps(r["best_params"], default=str)})
    board = pd.DataFrame(rows)
    board.sort_values(["pr_auc", "f1"], ascending=False).to_csv(paths.reports / "leaderboard_full.csv", index=False)
    log.info("\n%s", board.sort_values("pr_auc", ascending=False)[
        ["model", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "ece", "latency_ms"]]
        .round(4).to_string(index=False))

    # 4) champion: chosen on VALIDATION only -------------------------------------------------
    cand = board[board["model"].map(lambda n: zoo[n].selectable)].copy()
    cand["_vp"] = cand["val_pr_auc"].round(3)
    champ_name = cand.sort_values(["_vp", "val_recall_at_precision", "latency_ms"],
                                  ascending=[False, False, True]).iloc[0]["model"]
    champ = results[champ_name]; log.info("Champion candidate: %s", champ_name)

    # 5) calibration (raw vs sigmoid vs isotonic) picked by validation log-loss ---------
    final_pipe, calibration = champ["pipeline"], "none"
    val_p, test_p = champ["val_proba"], champ["test_proba"]
    if tcfg["calibrate_champion"]:
        trials = {"none": (champ["pipeline"], val_p, test_p)}
        for method in ("sigmoid", "isotonic"):
            try:
                cp = assemble_pipeline(champ_name, champ["best_params"], params["features"], method, seed, tcfg["n_jobs"])
                cp.fit(tr["text"], ytr)
                trials[method] = (cp, cp.predict_proba(list(va["text"]))[:, 1], cp.predict_proba(list(te["text"]))[:, 1])
            except Exception as exc:  # noqa: BLE001
                log.warning("calibration %s failed: %s", method, exc)
        calib_table = {m: {"val_brier": float(ev.brier_score_loss(yva, v[1])), "val_ece": ev.expected_calibration_error(yva, v[1]),
                           "val_log_loss": float(ev.log_loss(yva, np.clip(v[1], 1e-15, 1 - 1e-15)))} for m, v in trials.items()}
        # log-loss (strictly proper, punishes the exact-0/1 plateaus of isotonic) decides; Brier/ECE are reported
        calibration = min(calib_table, key=lambda m: calib_table[m]["val_log_loss"])
        final_pipe, val_p, test_p = trials[calibration]
        log.info("Calibration table: %s -> using '%s'", json.dumps(calib_table), calibration)
        (paths.reports / "calibration_comparison.json").write_text(json.dumps(calib_table, indent=2))

    # Threshold: tuned on 5-fold out-of-fold predictions over train+val (~5x more positives than val
    # alone), because a precision floor tuned on ~100 validation spam messages is far too noisy.
    if thr_cfg.get("use_oof", True):
        trval = pd.concat([tr, va], ignore_index=True)
        y_trval = trval["label"].to_numpy()
        oof_pipe = assemble_pipeline(champ_name, champ["best_params"], params["features"], calibration, seed, tcfg["n_jobs"])
        oof = cross_val_predict(oof_pipe, trval["text"], y_trval, method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
        thr_y, thr_p, tuned_on = y_trval, oof, f"5-fold OOF on train+val (n={len(trval)})"
    else:
        thr_y, thr_p, tuned_on = yva, val_p, f"validation set (n={len(yva)})"
    thr, thr_info = ev.choose_threshold(thr_y, thr_p, thr_cfg["strategy"], thr_cfg["min_precision"], thr_cfg["beta"],
                                        thr_cfg["cost_fp"], thr_cfg["cost_fn"])
    thr_info["tuned_on"] = tuned_on
    m_val, m_test = ev.compute_metrics(yva, val_p, thr), ev.compute_metrics(yte, test_p, thr)
    ci = ev.bootstrap_ci(yte, test_p, thr, n=params["bootstrap"]["n"], seed=seed)
    log.info("CHAMPION %s [%s calibration] thr=%.4f | test P=%.4f R=%.4f F1=%.4f PR-AUC=%.4f", champ_name,
             calibration, thr, m_test["precision"], m_test["recall"], m_test["f1"], m_test["pr_auc"])

    # 6) deep-dive reports ------------------------------------------------------------------
    pred_te = (test_p >= thr).astype(int)
    sig = {n: ev.mcnemar_p(yte, pred_te, (r["test_proba"] >= ops[n]).astype(int))
           for n, r in results.items() if n != champ_name}
    board["mcnemar_p_vs_champion"] = board["model"].map(sig)
    board["is_champion"] = board["model"] == champ_name
    board = board.sort_values(["pr_auc", "f1"], ascending=False)
    board.to_csv(paths.reports / "leaderboard_full.csv", index=False)
    board.drop(columns=["best_params"]).round(4).to_markdown(paths.reports / "leaderboard.md", index=False)

    te_out = te.assign(score=test_p, pred=pred_te)
    te_out[(te_out["pred"] == 1) & (te_out["label"] == 0)].sort_values("score", ascending=False) \
        .to_csv(paths.reports / "errors_false_positives.csv", index=False)
    te_out[(te_out["pred"] == 0) & (te_out["label"] == 1)].sort_values("score") \
        .to_csv(paths.reports / "errors_false_negatives.csv", index=False)
    ev.slice_report(te["text"], yte, test_p, thr).round(4).to_csv(paths.reports / "slice_metrics.csv", index=False)
    pp = lambda texts: final_pipe.predict_proba(list(texts))[:, 1]  # noqa: E731
    rob = ev.robustness_report(pp, te["text"].tolist(), yte, thr, seed)
    rob.round(4).to_csv(paths.reports / "robustness.csv", index=False)

    F = paths.figures
    all_curves = {n: (yte, r["test_proba"]) for n, r in results.items()}
    ev.plot_roc_all(all_curves, F / "roc_all_models.png")
    ev.plot_pr_all(all_curves, F / "pr_all_models.png")
    ev.plot_calibration_all({**all_curves, f"CHAMPION ({calibration})": (yte, test_p)}, F / "calibration_all_models.png")
    ev.plot_confusion(yte, test_p, thr, f"{champ_name} - test", F / "confusion_champion.png")
    ev.plot_threshold_sweep(thr_y, thr_p, thr, F / "threshold_sweep_champion.png")
    ev.plot_score_distribution(yte, test_p, thr, F / "score_distribution_champion.png")
    ev.plot_leaderboard(board, F / "leaderboard_bars.png")
    ev.plot_metric_heatmap(board, ["accuracy", "precision", "recall", "f1", "mcc", "pr_auc", "roc_auc", "specificity"],
                           F / "metric_heatmap.png")
    ev.plot_speed_vs_quality(board, F / "speed_vs_quality.png")
    ev.plot_robustness(rob, F / "robustness.png")
    if "logistic_regression" in results:
        gf = global_top_features(results["logistic_regression"]["pipeline"])
        if gf is not None:
            ev.plot_top_features(*gf, F / "top_features_logreg.png")
            names, coef = gf; o = np.argsort(coef)
            pd.DataFrame({"feature": names[o[::-1][:50]], "coef": coef[o[::-1][:50]]}).to_csv(
                paths.reports / "top_spam_features.csv", index=False)

    # 7) drift reference (VALIDATION data = unseen by training, like production traffic) ------
    ref = meta_features(va["text"])
    ref["oov_rate"] = oov_rate(va["text"], final_pipe.named_steps["features"].transformer_list[0][1])
    ref["score"] = val_p
    ref.to_csv(paths.reports / "reference_features.csv", index=False)

    # 8) register ---------------------------------------------------------------------------
    metadata = {"model_name": champ_name, "best_params": champ["best_params"], "calibration": calibration,
                "threshold": thr, "threshold_policy": {**thr_cfg, **thr_info}, "features": params["features"],
                "seed": seed, "data_fingerprint": manifest["fingerprint"], "data_manifest": manifest,
                "metrics_val": m_val, "metrics_test": m_test, "bootstrap_ci_test": ci,
                "n_models_compared": len(results), "params": params}
    version = save_bundle(final_pipe, metadata, promote=promote)
    summary = {"version": version, "champion": champ_name, "calibration": calibration, "threshold": thr,
               "metrics_test": m_test, "bootstrap_ci_test": ci, "n_models": len(results),
               "runtime_minutes": round((time.perf_counter() - t_start) / 60, 2)}
    _atomic_json(paths.reports / "summary.json", summary)
    _atomic_json(paths.reports / "metrics.json", {k: round(float(m_test[k]), 5) for k in
                 ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "brier", "ece", "fpr", "fnr")})
    write_model_card(paths.reports / "MODEL_CARD.md", metadata, board, rob, ci, version)

    # 9) experiment tracking ----------------------------------------------------------------
    with _mlflow("spamguard", use_mlflow) as mlf:
        if mlf:
            try:
                for _, row in board.iterrows():
                    with mlf.start_run(run_name=row["model"]):
                        mlf.log_params({"model": row["model"], "family": row["family"],
                                        **{k: str(v) for k, v in json.loads(row["best_params"]).items()}})
                        mlf.log_metrics({f"test_{k}": float(row[k]) for k in ("accuracy", "precision", "recall", "f1",
                                         "pr_auc", "roc_auc", "mcc", "brier", "ece", "fpr", "fnr")} |
                                        {"cv_pr_auc": float(row["cv_pr_auc"]), "latency_ms": float(row["latency_ms"])})
                with mlf.start_run(run_name=f"CHAMPION-{version}"):
                    mlf.log_params({"model": champ_name, "calibration": calibration, "threshold": thr,
                                    "data_fingerprint": manifest["fingerprint"]})
                    mlf.log_metrics({f"test_{k}": float(v) for k, v in m_test.items() if isinstance(v, float)})
                    mlf.log_artifacts(str(paths.figures), artifact_path="figures")
                    mlf.log_artifact(str(paths.reports / "leaderboard_full.csv"))
                    mlf.log_artifact(str(paths.reports / "MODEL_CARD.md"))
                    mlf.sklearn.log_model(final_pipe, name="model", serialization_format="cloudpickle")
            except Exception as exc:  # noqa: BLE001
                log.warning("MLflow logging failed (non-fatal): %s", exc)
    log.info("Done in %.1f min. Version %s promoted=%s", summary["runtime_minutes"], version, promote)
    return summary


def write_model_card(path: Path, meta: dict, board: pd.DataFrame, rob: pd.DataFrame, ci: dict, version: str):
    m, d = meta["metrics_test"], meta["data_manifest"]
    top = board.head(5)[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc"]].round(4).to_markdown(index=False)
    f = lambda k: f"{m[k]:.4f} (95% CI {ci[k]['low']:.4f}-{ci[k]['high']:.4f})"  # noqa: E731
    path.write_text(f"""# Model card - SpamGuard `{version}`

**Task:** binary classification of SMS messages (spam vs ham). **Champion:** `{meta['model_name']}`
(calibration: `{meta['calibration']}`), {meta['n_models_compared']} models compared.

## Data
Source `{d['source']}`; {d['rows_after_cleaning']} messages after cleaning ({d['duplicates_removed']} duplicates and
{d['conflicting_label_texts_removed']} label-conflicts removed); spam rate {d['spam_rate']:.1%}.
Stratified split: {json.dumps(d['splits'])}.

## Decision policy
Threshold **{meta['threshold']:.4f}** chosen ({meta['threshold_policy'].get('tuned_on', 'validation set')}) with strategy `{meta['threshold_policy']['strategy']}`
({meta['threshold_policy'].get('note', '')}). Blocking a legitimate message is treated as costlier than missing spam.

## Test performance (held out, used once)
| metric | value |
|---|---|
| Precision | {f('precision')} |
| Recall | {f('recall')} |
| F1 | {f('f1')} |
| MCC | {f('mcc')} |
| PR-AUC | {f('pr_auc')} |
| ROC-AUC | {f('roc_auc')} |
| False-positive rate | {f('fpr')} |
| Brier / ECE | {m['brier']:.4f} / {m['ece']:.4f} |
| Confusion (TN/FP/FN/TP) | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |

### Top 5 models
{top}

### Robustness (spam recall under evasion attacks)
{rob.round(3).to_markdown(index=False)}

## Limitations & ethical notes
* Trained on English SMS (2000s UK/Singapore-era corpus): new scams (crypto, delivery, bank-OTP phishing) and other
  languages are likely under-represented -> monitor drift and collect feedback.
* The test set is small; differences between the top models are usually inside the confidence interval.
* Messages are personal data: prediction logs store hashes + numeric features, never raw text by default.
* A spam flag should trigger *soft* actions (move to junk folder) rather than silent deletion.
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true"); ap.add_argument("--models", help="comma separated subset")
    ap.add_argument("--source", choices=["uci", "multilingual", "local"]); ap.add_argument("--local-path")
    a = ap.parse_args()
    over = {"data": {k: v for k, v in (("source", a.source), ("local_path", a.local_path)) if v}}
    run(load_params(a.params, over), resume=not a.no_resume, fast=a.fast,
        models=a.models.split(",") if a.models else None, promote=not a.no_promote, use_mlflow=not a.no_mlflow)


if __name__ == "__main__":
    main()
