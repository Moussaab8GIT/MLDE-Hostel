"""Offline API contract tests: deterministic model double, no training or MLflow server."""
import importlib
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    import mlflow
    import mlflow.sklearn
    from hotel_mlops import settings
    from hotel_mlops import monitoring
    patch = pytest.MonkeyPatch()
    path = tmp_path_factory.mktemp("api") / "predictions.db"
    class ModelDouble:
        feature_names_in_ = np.array(["LeadTime"])
        named_steps = {"classifier": SimpleNamespace(classes_=np.array([0, 1]))}
        def predict(self, features):
            return np.array([1])
        def predict_proba(self, features):
            return np.array([[.1, .9]])
    patch.setattr(settings, "DB_PATH", path)
    patch.setattr(monitoring, "DB_PATH", path)
    patch.setattr(mlflow, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda *args: SimpleNamespace(version="7")))
    def load(uri):
        assert uri == "models:/hotel-cancellation-model/7"
        return ModelDouble()
    patch.setattr(mlflow.sklearn, "load_model", load)
    module = importlib.import_module("hotel_mlops.app")
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            yield client, module, path
    finally:
        patch.undo()


def test_predict_and_feedback_are_linked_and_duplicates_rejected(api):
    client, module, path = api
    response = client.post("/predict", json={"LeadTime": 100})
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "7"
    feedback = {"prediction_id": result["prediction_id"], "actual_canceled": 1}
    assert client.post("/feedback", json=feedback).status_code == 200
    assert client.post("/feedback", json=feedback).status_code == 409
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT actual_canceled FROM predictions WHERE prediction_id=?",
                                  (result["prediction_id"],)).fetchone() == (1,)


def test_validation_failure_is_a_persisted_trace(api):
    client, module, path = api
    response = client.post("/predict", json={"LeadTime": -1})
    assert response.status_code == 422
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT status_code FROM request_errors WHERE request_id=?",
                                  (response.headers["X-Request-ID"],)).fetchone() == (422,)


def test_inference_error_returns_safe_response_and_persists_version(api, monkeypatch):
    client, module, path = api
    def fail(*args):
        raise RuntimeError("controlled model failure")
    monkeypatch.setattr(module.model, "predict", fail)
    response = client.post("/predict", json={"LeadTime": 100})
    assert response.status_code == 500
    assert "controlled model failure" not in response.text
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT error_type, model_version FROM request_errors WHERE request_id=?",
                                  (response.headers["X-Request-ID"],)).fetchone() == ("RuntimeError", "7")


def test_unknown_prediction_is_not_accepted(api):
    client, *_ = api
    assert client.post("/feedback", json={"prediction_id": "00000000-0000-0000-0000-000000000000",
                                         "actual_canceled": 0}).status_code == 404
