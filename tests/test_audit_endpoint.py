"""Tests for the /audit/recent endpoint."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SHADOW_AUTH_ENABLED", "false")


@pytest.fixture(scope="module")
def client():
    from shadowd.shadowd import app, shadow
    from shikigami import get_registry
    if shadow.shikigami is None:
        shadow.shikigami = get_registry()
    return TestClient(app)


class TestAuditEndpoint:
    def test_returns_disabled_state_by_default(self, client):
        from shadowd import ability_audit
        ability_audit.reset_for_tests(enabled=False)
        r = client.get("/audit/recent")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["enabled"] is False
        assert body["count"] == 0
        assert body["entries"] == []

    def test_returns_entries_when_enabled(self, client, tmp_path):
        from shadowd import ability_audit
        ability_audit.reset_for_tests(audit_dir=tmp_path / "audit", enabled=True)
        # Invoke an ability to generate an audit entry.
        r = client.post("/shikigami/summon",
                        json={"question": "test", "shikigami": "oracle"})
        assert r.status_code == 200
        # Now fetch via the endpoint.
        r = client.get("/audit/recent")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["enabled"] is True
        assert body["count"] >= 1
        # Most recent entry should reference oracle.
        ability_names = {e["ability"] for e in body["entries"]}
        assert "oracle" in ability_names
        # Cleanup — disable for downstream tests.
        ability_audit.reset_for_tests(enabled=False)

    def test_query_hash_not_raw(self, client, tmp_path):
        from shadowd import ability_audit
        ability_audit.reset_for_tests(audit_dir=tmp_path / "audit2", enabled=True)
        secret = "MY_SECRET_TOKEN_8741"
        client.post("/shikigami/summon", json={"question": secret, "shikigami": "oracle"})
        r = client.get("/audit/recent")
        body = r.json()
        assert all(secret not in str(e) for e in body["entries"])
        # The hash should be present.
        assert all("query_hash" in e for e in body["entries"])
        ability_audit.reset_for_tests(enabled=False)

    def test_limit_clamped(self, client, tmp_path):
        from shadowd import ability_audit
        ability_audit.reset_for_tests(audit_dir=tmp_path / "audit3", enabled=True)
        for _ in range(5):
            client.post("/shikigami/summon",
                        json={"question": "x", "shikigami": "oracle"})
        # Asking for 999999 entries clamps to 1000 — no crash.
        r = client.get("/audit/recent?limit=999999")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] <= 1000
        ability_audit.reset_for_tests(enabled=False)
