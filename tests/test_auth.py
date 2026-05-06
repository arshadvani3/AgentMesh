"""Tests for JWT authentication on protected registry endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mesh.models import AgentManifest, CapabilitySchema
from mesh.registry import _JWT_SECRET, _create_token, app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(client: TestClient, agent_id: str) -> dict:
    m = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint="ws://localhost:9999",
        capabilities=[CapabilitySchema(name="cap", description="test")],
    )
    resp = client.post("/agents/register", json=m.model_dump(mode="json"))
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuthToken:
    def test_get_token_wrong_secret_returns_401(self, client: TestClient):
        """POST /auth/token with wrong secret should return 401."""
        _register(client, "auth-tok-agent")
        resp = client.post(
            "/auth/token",
            params={"agent_id": "auth-tok-agent", "secret": "wrong-secret"},
        )
        assert resp.status_code == 401

    def test_get_token_unknown_agent_returns_404(self, client: TestClient):
        """POST /auth/token for an unregistered agent should return 404."""
        resp = client.post(
            "/auth/token",
            params={"agent_id": "ghost-agent", "secret": _JWT_SECRET},
        )
        assert resp.status_code == 404

    def test_get_token_valid_returns_bearer(self, client: TestClient):
        """POST /auth/token with correct secret returns an access_token."""
        _register(client, "auth-valid-agent")
        resp = client.post(
            "/auth/token",
            params={"agent_id": "auth-valid-agent", "secret": _JWT_SECRET},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 10


class TestProtectedEndpoints:
    def test_trust_update_no_token_succeeds_in_dev_mode(self, client: TestClient):
        """In dev mode (default secret), trust/update works without a token."""
        _register(client, "auth-dev-agent")
        resp = client.post(
            "/trust/update",
            params={"agent_id": "auth-dev-agent", "success": True, "quality": 0.8},
        )
        # Dev mode: _JWT_SECRET == default → _verify_token returns "anonymous"
        assert resp.status_code == 200

    def test_trust_update_valid_token_succeeds(self, client: TestClient):
        """trust/update with a valid Bearer token should succeed."""
        _register(client, "auth-bearer-agent")
        token = _create_token("auth-bearer-agent")
        resp = client.post(
            "/trust/update",
            params={"agent_id": "auth-bearer-agent", "success": True, "quality": 0.9},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_trust_update_invalid_token_dev_mode_still_passes(self, client: TestClient):
        """In dev mode, even an invalid token is accepted (auth is opt-in)."""
        _register(client, "auth-invalid-tok-agent")
        resp = client.post(
            "/trust/update",
            params={"agent_id": "auth-invalid-tok-agent", "success": True, "quality": 0.5},
            headers={"Authorization": "Bearer this-is-not-valid"},
        )
        # Dev mode allows anonymous — the bad token is ignored, not an error
        assert resp.status_code == 200

    def test_public_endpoints_require_no_auth(self, client: TestClient):
        """GET /agents and POST /discover are public — no token needed."""
        resp = client.get("/agents")
        assert resp.status_code == 200

        resp = client.post("/discover", json={"top_k": 5})
        assert resp.status_code == 200
