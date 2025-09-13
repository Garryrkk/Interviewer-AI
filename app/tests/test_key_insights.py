import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.key_insights_service import KeyInsightsService
from app.schemas.key_insights import KeyInsightResponse, KeyInsight

client = TestClient(app)

def test_extract_key_insights_valid():
    text = "AI can help in healthcare by predicting diseases and improving diagnosis."
    service = KeyInsightsService()
    result = service.extract_key_insights(text)

    assert isinstance(result, list)
    assert len(result) > 0
    assert any("predicting diseases" in insight.text.lower() for insight in result)


def test_extract_key_insights_empty_input():
    service = KeyInsightsService()
    with pytest.raises(ValueError):
        service.extract_key_insights("")


def test_api_key_insights_valid_request():
    payload = {
        "text": "AI can help in healthcare by predicting diseases and reducing human error."
    }

    response = client.post("/api/v1/insights", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "insights" in data
    assert isinstance(data["insights"], list)
    assert len(data["insights"]) > 0
    assert any("predicting diseases" in i["text"].lower() for i in data["insights"])


def test_api_key_insights_empty_request():
    payload = {"text": ""}

    response = client.post("/api/v1/insights", json=payload)
    assert response.status_code == 400  # Bad request
    data = response.json()
    assert "detail" in data


def test_api_key_insights_garbage_input():
    payload = {"text": "asdfasdfasdfasdf"}  
    response = client.post("/api/v1/insights", json=payload)
    assert response.status_code in (200, 400)

    if response.status_code == 200:
        data = response.json()
        assert "insights" in data
        assert isinstance(data["insights"], list)

