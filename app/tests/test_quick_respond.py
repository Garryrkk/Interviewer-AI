from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_quick_response_valid():
    payload = {"prompt": "What is the capital of France?"}
    response = client.post("/quick-respond/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["success"] is True

def test_quick_response_empty_prompt():
    payload = {"prompt": ""}
    response = client.post("/quick-respond/", json=payload)
    assert response.status_code in (200, 400)

