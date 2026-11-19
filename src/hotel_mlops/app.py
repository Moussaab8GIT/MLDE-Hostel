from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from hotel_mlops.schemas import BookingRequest, FeedbackRequest
from hotel_mlops.settings import CONFIG, MODEL_NAME, MODEL_ALIAS, TRACKING_URI, DB_PATH, REPORTS_DIR
from hotel_mlops.storage import init_database as init_trace_database, record_error, feedback_metrics, flow_reports
from hotel_mlops.monitoring import ProjectCollector

REGISTERED_MODEL_NAME = MODEL_NAME
mlflow.set_tracking_uri(TRACKING_URI)

model_version = mlflow.MlflowClient().get_model_version_by_alias(
    REGISTERED_MODEL_NAME, MODEL_ALIAS
)
model = mlflow.sklearn.load_model(
    f"models:/{REGISTERED_MODEL_NAME}/{model_version.version}"
)
FEATURE_COLUMNS = list(model.feature_names_in_)


init_trace_database(DB_PATH)

app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="Predicts whether a hotel booking will be canceled.",
    version="1.0.0",
)

# Expose request counts and latency metrics for Prometheus at /metrics.
Instrumentator().instrument(app).expose(app, include_in_schema=False)
REGISTRY.register(ProjectCollector())


@app.middleware("http")
async def persist_errors(request: Request, call_next):
    started = time.perf_counter()
    request_id = str(uuid4())
    error_type = "http_error"
    try:
        response = await call_next(request)
    except Exception as exc:
        error_type = type(exc).__name__
        logging.exception("Request failed: %s", request_id)
        response = JSONResponse(status_code=500,
                                content={"detail": "Internal service error", "request_id": request_id})
    if response.status_code >= 400:
        try:
            record_error(DB_PATH, request_id, request.url.path, response.status_code,
                         str(model_version.version), (time.perf_counter() - started) * 1000,
                         error_type)
        except sqlite3.Error:
            logging.exception("Could not persist error trace: %s", request_id)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/ops/model-quality")
def model_quality():
    config = CONFIG["monitoring"]
    return {"minimum_feedback": config["minimum_feedback"],
            "window_days": config["feedback_window_days"],
            "note": "Reported outcomes may be biased; sample count alone does not establish reliability.",
            "versions": feedback_metrics(DB_PATH, config["minimum_feedback"], config["feedback_window_days"])}


@app.get("/ops/training-runs")
def training_runs():
    return flow_reports(REPORTS_DIR)[-50:]


@app.get("/ops/alerts")
def local_alerts():
    """Persistent local failure inbox, populated even while the API is stopped."""
    return [report for report in flow_reports(REPORTS_DIR) if report["status"] == "failed"][-50:]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(model_version.version),
    }


@app.post("/predict")
def predict(booking: BookingRequest) -> dict[str, Any]:
    data = booking.model_dump(exclude_none=True)
    started = time.perf_counter()
    # Keep only the model's known features and preserve the training column order.
    features = pd.DataFrame([data]).reindex(columns=FEATURE_COLUMNS).fillna(float("nan"))
    prediction = int(model.predict(features)[0])
    classes = list(model.named_steps["classifier"].classes_)
    canceled_index = classes.index(1)
    probability = float(model.predict_proba(features)[0][canceled_index])
    duration_ms = (time.perf_counter() - started) * 1000
    prediction_id = str(uuid4())

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                prediction_id, created_at, model_version, input_json,
                predicted_canceled, cancellation_probability, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now(timezone.utc).isoformat(),
                str(model_version.version),
                pd.Series([data]).to_json(orient="records", date_format="iso"),
                prediction,
                probability,
                duration_ms,
            ),
        )

    return {
        "prediction_id": prediction_id,
        "predicted_canceled": prediction,
        "cancellation_probability": probability,
        "model_version": str(model_version.version),
        "duration_ms": round(duration_ms, 3),
    }


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest) -> dict[str, str]:
    received_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            UPDATE predictions
            SET actual_canceled = ?, feedback_received_at = ?
            WHERE prediction_id = ? AND actual_canceled IS NULL
            """,
            (feedback.actual_canceled, received_at, str(feedback.prediction_id)),
        )
        if cursor.rowcount == 0:
            exists = connection.execute(
                "SELECT 1 FROM predictions WHERE prediction_id = ?",
                (str(feedback.prediction_id),),
            ).fetchone()
            if exists is None:
                raise HTTPException(status_code=404, detail="Prediction ID not found.")
            raise HTTPException(
                status_code=409,
                detail="Feedback has already been recorded for this prediction.",
            )

    return {"status": "feedback_saved", "prediction_id": str(feedback.prediction_id)}
