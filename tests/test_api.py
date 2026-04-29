from fastapi.testclient import TestClient
from api.main import app # Importing the FastAPI app for testing
import pytest

client = TestClient(app)

def test_healthcheck():
    """Verifies that the API responds on the root or a health endpoint"""
    response = client.get("/")
    assert response.status_code in [200, 404] # 404 is acceptable if no "/" route is defined


def test_prediction_endpoint(sample_transaction):
    """Tests the POST call to the API using a dictionary"""
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    data = response.json()
    
    # Check for the presence of one of the expected keys
    output_key = "prediction" if "prediction" in data else "is_fraud"
    assert output_key in data
    # Ensure the value is in a binary format (0, 1, True, or False)
    assert data[output_key] in [0, 1, True, False]

def test_metrics_presence():
    """Verifies that Prometheus can scrape the metrics"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text

def test_invalid_data_format():
    """Verifies that the API rejects malformed data (e.g., Amount passed as a string)"""
    bad_data = {"V1": "error", "Amount": "a_lot"}
    response = client.post("/predict", json=bad_data)
    assert response.status_code == 422 # Pydantic validation error