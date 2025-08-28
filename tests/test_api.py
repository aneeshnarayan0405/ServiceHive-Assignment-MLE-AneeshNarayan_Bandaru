import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_type" in response.json()
    assert "classes" in response.json()

def test_predict_invalid_file():
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400