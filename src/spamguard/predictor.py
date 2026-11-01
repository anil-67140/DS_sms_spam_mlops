"""Thread-safe inference wrapper with privacy-aware prediction logging."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from .config import get_paths
from .explain import explain_message
from .preprocessing import meta_features, oov_rate
from .registry import Bundle, load_bundle


class Predictor:
    def __init__(self, bundle: Bundle, log_predictions: bool = True):
        self.bundle = bundle
        self.log_predictions = log_predictions
        self._lock = threading.Lock()
        try:
            self._word_vec = bundle.pipeline.named_steps["features"].transformer_list[0][1]
        except Exception:  # noqa: BLE001
            self._word_vec = None

    @classmethod
    def from_champion(cls, version: str | None = None, **kw) -> "Predictor":
        return cls(load_bundle(version), **kw)

    # ------------------------------------------------------------------ core
    @property
    def version(self) -> str: return self.bundle.version
    @property
    def threshold(self) -> float: return self.bundle.threshold

    def predict_proba(self, texts) -> np.ndarray:
        return np.asarray(self.bundle.pipeline.predict_proba(list(texts)))[:, 1]

    def predict(self, texts, explain: bool = False) -> list[dict]:
        texts = list(texts)
        t0 = time.perf_counter()
        probs = self.predict_proba(texts)
        latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        out = []
        for text, p in zip(texts, probs):
            pid = uuid.uuid4().hex[:12]
            rec = {"prediction_id": pid, "label": "spam" if p >= self.threshold else "ham",
                   "spam_probability": round(float(p), 6), "threshold": round(self.threshold, 6),
                   "model_version": self.version, "latency_ms": round(latency, 3)}
            if explain:
                rec["explanation"] = explain_message(self.predict_proba, text)
            out.append(rec)
        if self.log_predictions:
            self._log(texts, out)
        return out

    # ------------------------------------------------------------------ logging
    def _log(self, texts, results):
        """Logs features + scores, NOT raw text (SMS is personal data) unless LOG_RAW_TEXT=1."""
        try:
            feats = meta_features(texts)
            feats["oov_rate"] = oov_rate(texts, self._word_vec) if self._word_vec is not None else 0.0
            lines = []
            for (text, r), (_, f) in zip(zip(texts, results), feats.iterrows()):
                row = {"ts": datetime.now(timezone.utc).isoformat(), "prediction_id": r["prediction_id"],
                       "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                       "score": r["spam_probability"], "label": r["label"], "model_version": r["model_version"],
                       **{k: float(v) for k, v in f.items()}}
                if os.getenv("LOG_RAW_TEXT") == "1":
                    row["text"] = text
                lines.append(json.dumps(row))
            with self._lock, open(get_paths().logs / "predictions.jsonl", "a") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception:  # noqa: BLE001  (logging must never break serving)
            pass

    # ------------------------------------------------------------------ feedback
    def record_feedback(self, text: str, true_label: str, prediction_id: str | None = None,
                        score: float | None = None, source: str = "api") -> dict:
        if score is None:
            score = float(self.predict_proba([text])[0])
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "text": text, "true_label": true_label,
               "predicted_label": "spam" if score >= self.threshold else "ham", "score": score,
               "model_version": self.version, "prediction_id": prediction_id, "source": source}
        with self._lock, open(get_paths().feedback / "feedback.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec
