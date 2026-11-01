"""FastAPI inference service (+ Gradio UI mounted at /ui).

    uvicorn spamguard.api:app --host 0.0.0.0 --port 8000
Docs: /docs   Health: /health /ready   Metrics (Prometheus text): /metrics
Set SPAMGUARD_API_KEY to require an `X-API-Key` header on prediction endpoints.
"""
from __future__ import annotations

import os
import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .logging_utils import get_logger
from .predictor import Predictor

log = get_logger("spamguard.api")
MAX_LEN = 2000


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LEN, examples=["WINNER! Claim your free prize now"])
    explain: bool = False


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LEN)
    true_label: Literal["spam", "ham"]
    prediction_id: str | None = None


class _Metrics:
    def __init__(self):
        self.req = defaultdict(int); self.pred = defaultdict(int); self.lat_sum = 0.0; self.lat_n = 0

    def render(self, version: str) -> str:
        lines = ["# TYPE spamguard_requests_total counter"]
        lines += [f'spamguard_requests_total{{path="{p}",status="{s}"}} {n}' for (p, s), n in self.req.items()]
        lines += ["# TYPE spamguard_predictions_total counter"]
        lines += [f'spamguard_predictions_total{{label="{k}"}} {v}' for k, v in self.pred.items()]
        lines += ["# TYPE spamguard_request_latency_seconds_sum counter", f"spamguard_request_latency_seconds_sum {self.lat_sum:.6f}",
                  "# TYPE spamguard_request_latency_seconds_count counter", f"spamguard_request_latency_seconds_count {self.lat_n}",
                  f'spamguard_model_info{{version="{version}"}} 1']
        return "\n".join(lines) + "\n"


def create_app(predictor: Predictor | None = None, enable_ui: bool | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            try:
                app.state.predictor = Predictor.from_champion()
                log.info("Loaded champion model %s", app.state.predictor.version)
            except Exception as exc:  # noqa: BLE001
                app.state.predictor = None; log.error("No model loaded: %s", exc)
        yield

    api = FastAPI(title="SpamGuard SMS Spam API", version=__version__, lifespan=lifespan,
                  description="Calibrated spam probability, threshold-based decision, explanations, feedback loop.")
    api.state.predictor = predictor
    api.state.metrics = _Metrics()

    def auth(x_api_key: str | None = Header(default=None)):
        key = os.getenv("SPAMGUARD_API_KEY")
        if key and not secrets.compare_digest(x_api_key or "", key):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def model(request: Request) -> Predictor:
        p = api.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded - train one first (python -m spamguard.train)")
        return p

    @api.middleware("http")
    async def instrument(request: Request, call_next):
        rid, t0 = request.headers.get("x-request-id", uuid.uuid4().hex[:12]), time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        m = api.state.metrics
        m.req[(request.url.path, response.status_code)] += 1; m.lat_sum += dt; m.lat_n += 1
        response.headers["x-request-id"] = rid
        response.headers["x-process-time-ms"] = f"{dt * 1000:.2f}"
        return response

    @api.get("/health", tags=["ops"])
    def health(): return {"status": "ok", "version": __version__}

    @api.get("/ready", tags=["ops"])
    def ready(request: Request):
        p = api.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready", "model_version": p.version}

    @api.get("/model-info", tags=["ops"], dependencies=[Depends(auth)])
    def model_info(p: Predictor = Depends(model)):
        md = p.bundle.metadata
        return {"version": p.version, "model": md.get("model_name"), "calibration": md.get("calibration"),
                "threshold": p.threshold, "created_utc": md.get("created_utc"), "sha256": md.get("sha256"),
                "metrics_test": {k: md["metrics_test"][k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc")},
                "data_fingerprint": md.get("data_fingerprint")}

    @api.post("/predict", tags=["inference"], dependencies=[Depends(auth)])
    def predict(req: PredictRequest, request: Request, p: Predictor = Depends(model)):
        out = p.predict([req.text], explain=req.explain)[0]
        api.state.metrics.pred[out["label"]] += 1
        return out

    @api.post("/predict/batch", tags=["inference"], dependencies=[Depends(auth)])
    def predict_batch(req: BatchRequest, request: Request, p: Predictor = Depends(model)):
        if any(not t.strip() or len(t) > MAX_LEN for t in req.texts):
            raise HTTPException(422, f"every text must be non-empty and <= {MAX_LEN} chars")
        out = p.predict(req.texts)
        for o in out:
            api.state.metrics.pred[o["label"]] += 1
        return {"model_version": p.version, "results": out}

    @api.post("/feedback", tags=["feedback"], dependencies=[Depends(auth)], status_code=201)
    def feedback(req: FeedbackRequest, p: Predictor = Depends(model)):
        rec = p.record_feedback(req.text, req.true_label, req.prediction_id)
        return {"status": "recorded", "was_correct": rec["predicted_label"] == rec["true_label"]}

    @api.get("/drift", tags=["monitoring"], dependencies=[Depends(auth)])
    def drift(window: int = 5000):
        from .monitoring import drift_report, load_prediction_log, load_reference
        try:
            return drift_report(load_reference(), load_prediction_log(window))
        except FileNotFoundError:
            raise HTTPException(404, "reference data missing - run training first")

    @api.get("/performance", tags=["monitoring"], dependencies=[Depends(auth)])
    def performance(p: Predictor = Depends(model)):
        from .monitoring import performance_from_feedback
        return performance_from_feedback(p.threshold)

    @api.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics(request: Request):
        from fastapi.responses import PlainTextResponse
        p = api.state.predictor
        return PlainTextResponse(api.state.metrics.render(p.version if p else "none"))

    if enable_ui if enable_ui is not None else os.getenv("SPAMGUARD_ENABLE_UI", "1") == "1":
        try:
            import gradio as gr
            from .app_gradio import build_ui
            gr.mount_gradio_app(api, build_ui(lambda: api.state.predictor), path="/ui")
        except Exception as exc:  # noqa: BLE001
            log.warning("Gradio UI not mounted: %s", exc)
    return api


app = create_app()
