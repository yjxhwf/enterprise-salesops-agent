import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.main import create_app


def test_health_without_model_key():
    settings = Settings(_env_file=None, LLM_API_KEY=None)
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "enterprise-salesops-agent-backend",
        "environment": settings.APP_ENV,
    }


def test_health_and_settings_do_not_expose_secret():
    sentinel = "test-only-not-a-real-credential"
    settings = Settings(_env_file=None, LLM_API_KEY=sentinel)
    assert sentinel not in repr(settings)
    assert sentinel not in settings.model_dump_json()
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")
    assert sentinel not in response.text
    assert not any(word in response.text.lower() for word in ("key", "secret", "token", "password"))


def test_cors_allows_only_configured_origin():
    settings = Settings(_env_file=None, CORS_ORIGINS=["http://localhost:3000"])
    with TestClient(create_app(settings)) as client:
        allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
        denied = client.get("/health", headers={"Origin": "https://untrusted.example"})
        preflight = client.options("/health", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET",
        })
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers
    assert preflight.status_code == 200


def test_cors_rejects_wildcard():
    with pytest.raises(ValidationError, match="explicit http"):
        Settings(_env_file=None, CORS_ORIGINS=["*"])


def test_example_environment_loads():
    settings = Settings(_env_file=".env.example")
    assert settings.CORS_ORIGINS == ["http://localhost:3000"]
    assert not settings.LLM_API_KEY or not settings.LLM_API_KEY.get_secret_value()
