from fastapi.testclient import TestClient
from ecommerce_buy_predictor.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_no_model():
    request_data = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=request_data)

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]
