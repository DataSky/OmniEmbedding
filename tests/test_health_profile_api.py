"""HTTP controls for independent monitoring profiles; no live credentials."""
import json

import pytest
from fastapi.testclient import TestClient

from omniembed.service_health import ServiceHealthManager


@pytest.fixture
def profiles_api(tmp_path, monkeypatch):
    from omniembed import server

    manager = ServiceHealthManager(tmp_path)
    monkeypatch.setattr(server, "SERVICE_HEALTH", manager)
    monkeypatch.setattr(server.PIPE, "start", lambda: None)
    monkeypatch.setattr(server.PIPE, "stop", lambda: None)
    monkeypatch.setattr(server.CONFIG, "api_key", "")
    with TestClient(server.app) as client:
        yield client, manager, server


def test_reading_profiles_does_not_create_config_or_probe(profiles_api):
    client, manager, _ = profiles_api
    for name in (None, "local", "colab"):
        response = client.get("/api/service-health", params={} if name is None else {"profile": name})
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.json()
        assert body["profile"]["id"] == (name or "local")
        assert len(body["profiles"]) == 2
        assert body["entries"] == []
    assert not manager.config_file.exists()
    assert not any(service.monitor_running for service in manager.services.values())


def test_tab_selection_persists_without_enabling_colab(profiles_api):
    client, manager, _ = profiles_api
    body = client.patch("/api/service-health/profiles/colab", json={"selected": True}).json()
    assert body["selected_profile"] == "colab"
    assert body["profile"]["enabled"] is False and body["latest_status"] == "DISABLED"
    restored = ServiceHealthManager(manager.data_dir)
    assert restored.summary()["profile"]["id"] == "colab"
    assert restored.summary()["profile"]["enabled"] is False
    assert not any(service.monitor_running for service in restored.services.values())


def test_disabled_switch_wins_over_missing_or_stale_logs(profiles_api):
    client, manager, _ = profiles_api
    response = client.patch("/api/service-health/profiles/local", json={"enabled": False})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["latest_status"] == "DISABLED"
    assert client.get("/api/service-health?profile=local").json()["latest_status"] == "DISABLED"
    assert ServiceHealthManager(manager.data_dir).summary("local")["profile"]["enabled"] is False


@pytest.mark.parametrize("payload", [{}, {"enabled": "false"}, {"enabled": 0}, {"enabled": None},
                                     {"selected": False}, {"selected": None},
                                     {"url": "http://another-host"}, {"key": "do-not-save"}])
def test_controls_accept_only_explicit_boolean_fields(profiles_api, payload):
    client, manager, _ = profiles_api
    response = client.patch("/api/service-health/profiles/local", json=payload)
    assert response.status_code == 422
    assert not manager.config_file.exists()


def test_unconfigured_colab_cannot_be_started_by_toggle(profiles_api):
    client, manager, _ = profiles_api
    response = client.patch("/api/service-health/profiles/colab", json={"enabled": True})
    assert response.status_code == 400
    assert manager.summary("colab")["profile"]["enabled"] is False
    assert not manager.services["colab"].monitor_running


def test_control_authentication_and_cross_origin_rejection(profiles_api, monkeypatch):
    client, manager, server = profiles_api
    monkeypatch.setattr(server.CONFIG, "api_key", "synthetic-control-key")
    assert client.patch("/api/service-health/profiles/local", json={"enabled": False}).status_code == 401
    headers = {"X-API-Key": "synthetic-control-key", "Origin": "http://untrusted.example"}
    assert client.patch("/api/service-health/profiles/local", json={"enabled": False}, headers=headers).status_code == 403
    assert not manager.config_file.exists()
    headers["Origin"] = "http://testserver"
    response = client.patch("/api/service-health/profiles/local", json={"enabled": False}, headers=headers)
    assert response.status_code == 200
    assert "synthetic-control-key" not in json.dumps(response.json())
