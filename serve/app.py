"""Inference service.

Serves the model the pipeline produced, and exposes exactly the surface a load test
and an orchestrator need: predict, batch predict, three health probes, metrics, and
the model's own identity.

The latency accounting is deliberate and is what the load test consumes. Every request
records three numbers separately - preprocessing, model forward, and total - because
"the endpoint is slow" is not an actionable finding, and the difference between those
three numbers is what tells you whether to optimise the model, the serialisation, or
the framework.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .health import HealthState
from .model_registry import ModelRegistry
from .warmup import warm

logging.basicConfig(level=os.environ.get("FORGE_LOG_LEVEL", "INFO"))
log = logging.getLogger("forge.serve")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("FORGE_ARTIFACTS", ROOT / "artifacts"))
DASHBOARD = ROOT / "serve" / "index.html"

# Torch spawns one thread per core by default. Under a load test that means N
# concurrent requests each trying to use N cores, and the resulting oversubscription
# makes throughput WORSE as concurrency rises. Pinned to 1: the service scales by
# handling more requests, not by making one request use the whole machine.
torch.set_num_threads(int(os.environ.get("FORGE_TORCH_THREADS", "1")))


class PredictRequest(BaseModel):
    # 784 raw greyscale pixels in 0..255. Deliberately not an image upload: the load
    # test must be able to generate valid payloads without shipping image files, and
    # decoding PNGs would put an image codec in the latency path being measured.
    pixels: List[float] = Field(..., min_length=784, max_length=784)


class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., min_length=1, max_length=256)


class Metrics:
    """Latency accounting, split by stage."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.count = 0
        self.errors = 0
        self.total_ms = 0.0
        self.preprocess_ms = 0.0
        self.inference_ms = 0.0
        self.samples: List[float] = []          # bounded reservoir for percentiles
        self.max_samples = 5000

    def record(self, total_ms: float, pre_ms: float, inf_ms: float, n: int = 1) -> None:
        with self.lock:
            self.count += n
            self.total_ms += total_ms
            self.preprocess_ms += pre_ms
            self.inference_ms += inf_ms
            self.samples.append(total_ms)
            if len(self.samples) > self.max_samples:
                # Drop the oldest half rather than one at a time: a deque of 5000
                # floats churned per request is measurable overhead in a service whose
                # whole latency budget is ~2ms.
                del self.samples[: self.max_samples // 2]

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            n, samples = self.count, sorted(self.samples)
            total, pre, inf = self.total_ms, self.preprocess_ms, self.inference_ms
            errors = self.errors

        def pct(p: float) -> float:
            if not samples:
                return 0.0
            return round(samples[min(len(samples) - 1, int(len(samples) * p))], 3)

        return {
            "requests": n,
            "errors": errors,
            "mean_total_ms": round(total / n, 3) if n else 0.0,
            "mean_preprocess_ms": round(pre / n, 3) if n else 0.0,
            "mean_inference_ms": round(inf / n, 3) if n else 0.0,
            # This ratio is the finding the load test is looking for: if inference is
            # a small fraction of total, the model is not the bottleneck and
            # optimising it would be wasted effort.
            "inference_share": round(inf / total, 3) if total else 0.0,
            "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "max_ms": round(samples[-1], 3) if samples else 0.0,
        }


registry = ModelRegistry(ARTIFACTS)
health = HealthState()
metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = registry.load()
        health.model_loaded = True
        health.warmup_report = warm(model.module)
        health.warmed = True
        log.info("serving model %s", model.version)
    except Exception as exc:                    # noqa: BLE001
        # Start anyway, but never become READY. The process staying up with a failing
        # readiness probe is what lets an orchestrator keep the previous version
        # serving while this one is diagnosed; exiting here would just crash-loop.
        health.last_error = str(exc)
        log.exception("startup failed; service will report NOT READY")
    yield


app = FastAPI(title="Forge Inference", version="1.0.0", lifespan=lifespan)


def _predict(tensor: torch.Tensor) -> tuple:
    model = registry.require()
    t0 = time.perf_counter()
    x = ((tensor / 255.0) - model.mean) / model.std
    x = x.view(-1, 1, 28, 28)
    pre_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    with torch.no_grad():
        logits = model.module(x)
    inf_ms = (time.perf_counter() - t1) * 1000

    # Temperature from the manifest, fitted on validation during evaluation. Serving
    # raw softmax would report a confidence the calibration report says is wrong.
    probs = F.softmax(logits / model.temperature, dim=1)
    return probs, pre_ms, inf_ms, model


@app.post("/predict")
async def predict(req: PredictRequest):
    started = time.perf_counter()
    try:
        probs, pre_ms, inf_ms, model = _predict(torch.tensor(req.pixels, dtype=torch.float32))
    except RuntimeError as exc:
        metrics.errors += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    confidence, index = probs[0].max(0)
    total_ms = (time.perf_counter() - started) * 1000
    metrics.record(total_ms, pre_ms, inf_ms)
    return {
        "prediction": model.classes[int(index)],
        "class_index": int(index),
        "confidence": round(float(confidence), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes, probs[0])},
        "model_version": model.version,
        "timing_ms": {"total": round(total_ms, 3), "preprocess": round(pre_ms, 3),
                      "inference": round(inf_ms, 3)},
    }


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    started = time.perf_counter()
    for row in req.instances:
        if len(row) != 784:
            raise HTTPException(422, f"each instance must have 784 pixels, got {len(row)}")
    try:
        probs, pre_ms, inf_ms, model = _predict(
            torch.tensor(req.instances, dtype=torch.float32))
    except RuntimeError as exc:
        metrics.errors += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    confidence, index = probs.max(1)
    total_ms = (time.perf_counter() - started) * 1000
    metrics.record(total_ms, pre_ms, inf_ms, n=len(req.instances))
    return {
        "predictions": [
            {"prediction": model.classes[int(i)], "confidence": round(float(c), 4)}
            for i, c in zip(index, confidence)
        ],
        "count": len(req.instances),
        "model_version": model.version,
        "timing_ms": {"total": round(total_ms, 3), "inference": round(inf_ms, 3),
                      "per_instance": round(total_ms / len(req.instances), 4)},
    }


# -- orchestrator probes -----------------------------------------------------

@app.get("/health/live")
async def live():
    return health.live()


@app.get("/health/ready")
async def ready():
    body = health.ready()
    # 503 is what removes an instance from the load balancer. Returning 200 with
    # {"ready": false} would keep traffic arriving at a service that cannot serve it.
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.get("/health/startup")
async def startup():
    return health.startup()


@app.get("/metrics")
async def get_metrics():
    return {"latency": metrics.snapshot(), "model": registry.state(),
            "warmup": health.warmup_report,
            "torch_threads": torch.get_num_threads()}


@app.get("/model")
async def model_info():
    model = registry.current
    if model is None:
        raise HTTPException(503, "no model loaded")
    return {"info": model.info(), "manifest": model.manifest}


@app.post("/admin/reload")
async def reload_model():
    """Load whatever is currently in the artifacts directory. This is what a deploy
    calls after shipping a new model, and it is the reason rollback exists."""
    try:
        model = registry.load()
        health.warmup_report = warm(model.module)
        health.model_loaded = health.warmed = True
        health.last_error = ""
        return {"loaded": model.info()}
    except Exception as exc:                    # noqa: BLE001
        health.last_error = str(exc)
        raise HTTPException(500, f"reload failed, previous model still serving: {exc}") from exc


@app.post("/admin/rollback")
async def rollback():
    model = registry.rollback()
    if model is None:
        raise HTTPException(409, "no previous model to roll back to")
    return {"rolled_back_to": model.info()}


@app.get("/", response_class=HTMLResponse)
async def index():
    if DASHBOARD.exists():
        return DASHBOARD.read_text()
    return "<h1>Forge Inference</h1><p>UI not built.</p>"
