from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"
LABEL_MAPPING_PATH = ARTIFACT_DIR / "label_mapping.json"

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
PREDICTION_COUNT = Counter("prediction_class_total", "Prediction count by class", ["label"])

app = FastAPI(title="Student Learning Classifier", version="1.0.0")
app.mount("/metrics", make_asgi_app())

_model: tf.keras.Model | None = None
_scaler: Any | None = None
_feature_columns: list[str] | None = None
_id_to_label: dict[int, str] | None = None


class PredictionRequest(BaseModel):
    features: dict[str, float | int | bool]


class PredictionResponse(BaseModel):
    prediction: str
    probability_good: float
    status: str


def load_artifacts() -> None:
    global _model, _scaler, _feature_columns, _id_to_label
    missing = [
        str(path)
        for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, LABEL_MAPPING_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")

    _model = tf.keras.models.load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    label_mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    _id_to_label = {int(value): key for key, value in label_mapping.items()}


@app.on_event("startup")
def startup() -> None:
    try:
        load_artifacts()
    except FileNotFoundError:
        # Keep API bootable for health checks before the model is trained.
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    artifacts_ready = all(
        path.exists() for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, LABEL_MAPPING_PATH]
    )
    return {"status": "ok", "artifacts_ready": artifacts_ready}


@app.get("/features")
def features() -> dict[str, Any]:
    if _feature_columns is None:
        if FEATURE_COLUMNS_PATH.exists():
            columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
        else:
            raise HTTPException(status_code=503, detail="Feature columns artifact is not available.")
    else:
        columns = _feature_columns
    return {"feature_count": len(columns), "features": columns}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    REQUEST_COUNT.inc()
    start = time.perf_counter()
    try:
        if _model is None or _scaler is None or _feature_columns is None or _id_to_label is None:
            load_artifacts()

        row = np.array([[float(payload.features.get(col, 0)) for col in _feature_columns]])
        scaled = _scaler.transform(row)
        probability_good = float(_model.predict(scaled, verbose=0).ravel()[0])
        class_id = int(probability_good >= 0.5)
        label = _id_to_label[class_id]
        PREDICTION_COUNT.labels(label=label).inc()
        return PredictionResponse(
            prediction=label,
            probability_good=probability_good,
            status="success",
        )
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)
