import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    """Provide an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c



@pytest.mark.asyncio
async def test_summarization_valid(client):
    """
    Test summarization with valid text input.
    Should return a summary in the response.
    """
    response = await client.post("/api/v1/summarization", json={
        "text": "This is a long interview transcript where the candidate explains their background and skills..."
    })
    
    assert response.status_code == 200
    body = response.json()
    
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert len(body["summary"]) > 0


@pytest.mark.asyncio
async def test_summarization_empty_text(client):
    """
    Test summarization with empty text.
    Should return a 400 Bad Request error.
    """
    response = await client.post("/api/v1/summarization", json={"text": ""})
    
    assert response.status_code == 400
    body = response.json()
    
    assert body["detail"] == "Text cannot be empty"

