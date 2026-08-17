import json
from types import SimpleNamespace

import joblib
import pytest
from fastapi.testclient import TestClient

import ecommerce_buy_predictor.api.main as api
from ecommerce_buy_predictor.api.main import ModelState, app
from ecommerce_buy_predictor.models.train import ModelTrainer


@pytest.fixture
def trained_model(raw_features, raw_target):
    return ModelTrainer("LogisticRegression", {"max_iter": 300}, 42).fit(
        raw_features, raw_target
    ).model


@pytest.fixture
def client_with_model(monkeypatch, trained_model):
    monkeypatch.setattr(
        api, "load_model", lambda: ModelState(trained_model, "tests://in-memory")
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(api, "load_model", ModelState)
    with TestClient(app) as client:
        yield client


def _fake_mlflow_sklearn(load_model):
    return SimpleNamespace(load_model=load_model)


def _registry_is_down(uri):
    raise RuntimeError("no tracking server")


def test_load_model_prefers_the_registry(monkeypatch, trained_model):
    monkeypatch.setattr(
        api.mlflow, "sklearn", _fake_mlflow_sklearn(lambda uri: trained_model)
    )

    state = api.load_model()

    assert state.is_loaded
    assert state.source.startswith("models:/")


def test_load_model_falls_back_to_local_pickle(monkeypatch, tmp_path, trained_model):
    monkeypatch.setattr(api.mlflow, "sklearn", _fake_mlflow_sklearn(_registry_is_down))
    monkeypatch.setattr(api.settings, "model_dir", str(tmp_path))
    joblib.dump(trained_model, tmp_path / "model.pkl")

    state = api.load_model()

    assert state.is_loaded
    assert state.source.endswith("model.pkl")


def test_load_model_gives_up_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(api.mlflow, "sklearn", _fake_mlflow_sklearn(_registry_is_down))
    monkeypatch.setattr(api.settings, "model_dir", str(tmp_path))

    state = api.load_model()

    assert not state.is_loaded
    assert state.source is None


def test_health_reports_loaded_model(client_with_model):
    response = client_with_model.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_source": "tests://in-memory",
    }


def test_health_reports_missing_model(client_without_model):
    body = client_without_model.get("/health").json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is False


def test_predict_returns_probability(client_with_model, session_payload):
    response = client_with_model.post("/predict", json=session_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_is_consistent_with_probability(client_with_model, session_payload):
    body = client_with_model.post("/predict", json=session_payload).json()

    assert body["prediction"] == int(body["probability"] >= 0.5)


def test_high_page_values_score_higher(client_with_model, session_payload):
    cold = dict(session_payload, PageValues=0.0, ExitRates=0.18)
    hot = dict(session_payload, PageValues=90.0, ExitRates=0.005)

    cold_probability = client_with_model.post("/predict", json=cold).json()["probability"]
    hot_probability = client_with_model.post("/predict", json=hot).json()["probability"]

    assert hot_probability > cold_probability


def test_batch_predict(client_with_model, session_payload):
    response = client_with_model.post(
        "/predict/batch", json={"sessions": [session_payload, session_payload]}
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_unknown_month_is_accepted(client_with_model, session_payload):
    response = client_with_model.post("/predict", json=dict(session_payload, Month="Jan"))

    assert response.status_code == 200


def test_missing_field_is_rejected(client_with_model, session_payload):
    session_payload.pop("PageValues")

    assert client_with_model.post("/predict", json=session_payload).status_code == 422


def test_out_of_range_rate_is_rejected(client_with_model, session_payload):
    response = client_with_model.post("/predict", json=dict(session_payload, BounceRates=42))

    assert response.status_code == 422


def test_predict_without_model_returns_503(client_without_model, session_payload):
    response = client_without_model.post("/predict", json=session_payload)

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_requests_are_logged_with_latency_and_status(client_with_model, session_payload, caplog):
    with caplog.at_level("INFO", logger="ecommerce_buy_predictor.api.access"):
        client_with_model.post("/predict", json=session_payload)

    record = next(r for r in caplog.records if r.name == "ecommerce_buy_predictor.api.access")
    payload = json.loads(record.getMessage())

    assert payload["method"] == "POST"
    assert payload["path"] == "/predict"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] >= 0


def test_errors_do_not_leak_internals(client_with_model, session_payload, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("connection string secret=hunter2")

    monkeypatch.setattr(api.state.model, "predict_proba", explode)
    response = client_with_model.post("/predict", json=session_payload)

    assert response.status_code == 500
    assert "hunter2" not in response.text
