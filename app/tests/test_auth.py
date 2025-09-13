import pytest
from httpx import AsyncClient
from app.core.config import settings

class TestAuthentication:
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self, client: AsyncClient, invalid_auth_headers):
        response = await client.get(
            "/api/v1/summarization",
            headers=invalid_auth_headers
        )
        
        if settings.REQUIRE_AUTH:
            assert response.status_code == 401
            body = response.json()
            assert "Invalid" in body.get("message", "") or "Unauthorized" in body.get("detail", "")
    
    @pytest.mark.asyncio
    async def test_missing_api_key(self, client: AsyncClient):
        response = await client.get("/api/v1/summarization")
        
        if settings.REQUIRE_AUTH:
            assert response.status_code == 401
            body = response.json()
            error_message = body.get("message", body.get("detail", "")).lower()
            assert any(word in error_message for word in ["authentication", "unauthorized", "token"])
        else:
            assert response.status_code in [200, 404, 422]
    
    @pytest.mark.asyncio
    async def test_valid_api_key(self, client: AsyncClient, auth_headers):
        if not settings.REQUIRE_AUTH:
            pytest.skip("Authentication not required in this environment")
        
        if not auth_headers:
            pytest.skip("No valid API key configured for testing")
        
        response = await client.get(
            "/api/v1/summarization",
            headers=auth_headers
        )
        
        assert response.status_code in [200, 404, 422]
        if response.status_code == 401:
            pytest.fail("Valid API key was rejected")
    
    @pytest.mark.asyncio
    async def test_malformed_auth_header(self, client: AsyncClient):
        malformed_headers = {"Authorization": "InvalidFormat token123"}
        
        response = await client.get(
            "/api/v1/summarization",
            headers=malformed_headers
        )
        
        if settings.REQUIRE_AUTH:
            assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_empty_bearer_token(self, client: AsyncClient):
        empty_token_headers = {"Authorization": "Bearer "}
        
        response = await client.get(
            "/api/v1/summarization",
            headers=empty_token_headers
        )
        
        if settings.REQUIRE_AUTH:
            assert response.status_code == 401

@pytest.mark.skipif(not settings.REQUIRE_AUTH, reason="Authentication tests only run when auth is enabled")
class TestAuthenticationRequired:
    
    @pytest.mark.asyncio
    async def test_protected_endpoints_require_auth(self, client: AsyncClient):
        protected_endpoints = [
            "/api/v1/summarization",
            "/api/v1/analysis",
            "/api/v1/interview"
        ]
        
        for endpoint in protected_endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"