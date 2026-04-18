from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# --- Sample valid input ---
valid_payload = {
    "passenger_count": 2,
    "trip_distance": 3.5,
    "trip_duration_minutes": 15,
    "pickup_hour": 14,
    "is_weekend": 0,
    "pickup_borough_label": 1,
    "dropoff_borough_label": 2,
    "fare_amount": 12.5,
    "extra": 0.5,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0
}

# 1️ Successful single prediction
def test_single_prediction():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction_id" in data
    assert "tip_amount" in data
    assert "model_version" in data


# 2️ Successful batch prediction
def test_batch_prediction():
    response = client.post("/predict/batch", json=[valid_payload, valid_payload])
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


# 3️ Invalid input (should return 422)
def test_invalid_input():
    bad_payload = valid_payload.copy()
    bad_payload["trip_distance"] = -5

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# 4️ Health endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# 5 Edge case (small values)
def test_edge_case():
    edge_payload = valid_payload.copy()
    edge_payload["trip_distance"] = 0.1
    edge_payload["trip_duration_minutes"] = 1

    response = client.post("/predict", json=edge_payload)
    assert response.status_code == 200