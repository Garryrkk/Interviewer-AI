from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_handfree_start():
    response = client.post("/handfree/start", json={"user_id": "test_user"})
    assert response.status_code == 200
    assert response.json()["status"] == "started"

def test_handfree_stop():
    response = client.post("/handfree/stop", json={"user_id": "test_user"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
