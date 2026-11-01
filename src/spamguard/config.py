"""Central configuration.

* All runtime artefacts live under ``SPAMGUARD_HOME`` (defaults to the project root).
  In Google Colab this points at a Google Drive folder, so every output survives
  runtime disconnects.
* Hyper-parameters live in ``params.yaml`` (tracked by DVC) and are deep-merged over
  ``DEFAULT_PARAMS`` below, so a missing key never crashes the pipeline.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PARAMS: dict[str, Any] = {
    "seed": 42,
    "data": {
        "source": "uci",              # uci | multilingual | local
        "local_path": None,           # CSV path when source == local
        "fallback_to_sample": False,  # use bundled sample if Kaggle is unreachable (CI / offline)
        "val_size": 0.15,
        "test_size": 0.15,
        "max_chars": 1000,
    },
    "features": {
        "word_ngram": [1, 2],
        "word_max_features": 20000,
        "char_ngram": [2, 4],
        "char_max_features": 30000,
        "min_df": 2,
        "stem": False,
        "stop_words": None,
        "use_char": True,
        "use_handcrafted": True,
    },
    "training": {
        "models": "all",              # "all" or a list of model names
        "cv_folds": 3,
        "n_iter": 6,                  # random-search iterations (light models)
        "heavy_n_iter": 3,            # random-search iterations (RF/ET/SVM/XGB/...)
        "scoring": "average_precision",
        "n_jobs": -1,
        "calibrate_champion": True,
    },
    "threshold": {
        "strategy": "min_precision",  # min_precision | max_fbeta | max_f1 | min_cost
        "min_precision": 0.99,        # business rule: <= 1% of flagged messages may be legit
        "beta": 0.5,
        "cost_fp": 5.0,               # blocking a legit SMS is 5x worse than missing a spam
        "cost_fn": 1.0,
        "use_oof": True,              # tune champion threshold on out-of-fold preds (robust)
    },
    "bootstrap": {"n": 1000},
    "monitoring": {"psi_warn": 0.1, "psi_alert": 0.25, "min_samples": 100},
    "retrain": {"min_feedback": 50, "min_pr_auc_delta": -0.002, "max_fpr_increase": 0.005},
}


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def raw(self) -> Path: return self.home / "data" / "raw"
    @property
    def processed(self) -> Path: return self.home / "data" / "processed"
    @property
    def feedback(self) -> Path: return self.home / "data" / "feedback"
    @property
    def models(self) -> Path: return self.home / "models"
    @property
    def reports(self) -> Path: return self.home / "reports"
    @property
    def figures(self) -> Path: return self.home / "reports" / "figures"
    @property
    def checkpoints(self) -> Path: return self.home / "checkpoints"
    @property
    def logs(self) -> Path: return self.home / "logs"
    @property
    def mlruns(self) -> Path: return self.home / "mlruns"

    def create(self) -> "Paths":
        for p in (self.raw, self.processed, self.feedback, self.models, self.reports,
                  self.figures, self.checkpoints, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(create: bool = True) -> Paths:
    """Resolve paths lazily so tests / Colab can change SPAMGUARD_HOME at any time."""
    home = Path(os.getenv("SPAMGUARD_HOME", PROJECT_ROOT)).expanduser().resolve()
    paths = Paths(home)
    return paths.create() if create else paths


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_params(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path or os.getenv("SPAMGUARD_PARAMS", PROJECT_ROOT / "params.yaml"))
    user: dict = {}
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
    return _deep_merge(_deep_merge(DEFAULT_PARAMS, user), overrides or {})
