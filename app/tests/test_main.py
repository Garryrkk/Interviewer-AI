import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    
    body = response.json()
    assert "Interview AI Assistant" in body["message"]
    assert body["status"] == "healthy"
    assert "features" in body
    assert isinstance(body["features"], list)

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    
    body = response.json()
    assert "status" in body
    assert body["status"] in ["healthy", "degraded", "unhealthy"]
    assert "services" in body

@pytest.mark.asyncio
async def test_emergency_stop(client: AsyncClient):
    response = await client.post("/api/emergency-stop")
    assert response.status_code == 200
    
    body = response.json()
    assert "Emergency stop activated" in body["message"]

@pytest.mark.asyncio
async def test_root_response_structure(client: AsyncClient):
    response = await client.get("/")
    body = response.json()
    
    required_fields = ["message", "status", "features"]
    for field in required_fields:
        assert field in body, f"Missing required field: {field}"
    
    assert isinstance(body["message"], str)
    assert isinstance(body["status"], str)
    assert isinstance(body["features"], list)

@pytest.mark.asyncio
async def test_health_response_structure(client: AsyncClient):
    response = await client.get("/api/health")
    body = response.json()
    
    required_fields = ["status", "services"]
    for field in required_fields:
        assert field in body, f"Missing required field: {field}"