"""Tests for the /abilities/health endpoint."""

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


class TestAbilitiesHealth:
    def test_lists_all_21_abilities(self, client):
        r = client.get("/abilities/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["registered"] == 21
        assert len(body["abilities"]) == 21
        names = {a["type"] for a in body["abilities"]}
        for needed in ("oracle", "tide", "echo", "lantern", "compass"):
            assert needed in names

    def test_each_ability_has_required_fields(self, client):
        r = client.get("/abilities/health")
        body = r.json()
        for a in body["abilities"]:
            for field in ("name", "type", "count", "successes", "failures",
                          "success_rate", "p50_latency_s", "p95_latency_s",
                          "avg_latency_s", "last_invoked_ts"):
                assert field in a, f"missing field: {field}"

    def test_stats_populate_after_invocation(self, client):
        # Reset metrics so this test starts clean.
        from shadowd import ability_metrics
        ability_metrics.reset()
        # Invoke an ability.
        client.post("/shikigami/summon",
                    json={"question": "test", "shikigami": "oracle"})
        r = client.get("/abilities/health")
        body = r.json()
        oracle_row = next(a for a in body["abilities"] if a["type"] == "oracle")
        assert oracle_row["count"] >= 1
        assert oracle_row["success_rate"] is not None
        assert oracle_row["p50_latency_s"] is not None
        # Cleanup.
        ability_metrics.reset()

    def test_uninvoked_abilities_show_null_stats(self, client):
        from shadowd import ability_metrics
        ability_metrics.reset()
        r = client.get("/abilities/health")
        body = r.json()
        # Every count should be zero.
        for a in body["abilities"]:
            assert a["count"] == 0
            assert a["success_rate"] is None
            assert a["p50_latency_s"] is None
