from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_main_feature_summary():
    payload = {"text": "Artificial Intelligence helps prepare for interviews."}
    response = client.post("/main-feature/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) <= len(payload["text"])
