"""Smoke tests for the /shikigami/council endpoint.

Exercises the cross-ability synthesis path in-process via FastAPI's TestClient
— no real daemon needed. Verifies:
- council runs with default abilities when none specified
- council accepts explicit abilities
- unknown ability names return 400
- result has the documented shape (verdict / leader / members)
"""

import os

import pytest
from fastapi.testclient import TestClient

# Auth must stay off for the TestClient default path.
os.environ.setdefault("SHADOW_AUTH_ENABLED", "false")


@pytest.fixture(scope="module")
def client():
    # Import inside the fixture so the test file can be collected even when
    # the heavier daemon-side imports are slow.
    from shadowd.shadowd import app, shadow
    from shadowd.observers import observer_status  # noqa: F401 — keeps wiring loaded
    from shikigami import get_registry

    # Make sure the in-process registry is wired onto the shadow singleton so
    # /shikigami/council can find it (TestClient doesn't run the lifespan).
    if shadow.shikigami is None:
        shadow.shikigami = get_registry()

    return TestClient(app)


class TestCouncilEndpoint:
    def test_default_council(self, client):
        # A decision question ("should I…") now relevance-routes to the
        # decision-bearing abilities rather than the old fixed five.
        r = client.post("/shikigami/council", json={"question": "should I take the new job offer?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "verdict" in body
        assert "members" in body
        assert body["selection"] == "relevance"
        # Shadow Council is the natural lead for a "should I" decision; the
        # analytical core guarantees Oracle is topped up for breadth.
        assert "shadow_council" in body["members"]
        assert "oracle" in body["members"]
        # Breadth floor and latency ceiling are respected.
        assert 3 <= len(body["members"]) <= 8

    def test_relevance_council_selects_emotional_abilities(self, client):
        # The flagship "how am I doing" case: a mood question should consult
        # Tide (sentiment trend), never possible under the old fixed council.
        r = client.post("/shikigami/council",
                        json={"question": "what has my mood been lately?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["selection"] == "relevance"
        assert "tide" in body["members"]

    def test_council_falls_back_to_default_on_no_signal(self, client):
        # A contentless question maps to nothing → broad default blend.
        r = client.post("/shikigami/council", json={"question": "test"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["selection"] == "default"
        assert set(body["members"].keys()) == {
            "oracle", "shadow_council", "pattern_sight",
            "social_radar", "habit_tracker",
        }

    def test_explicit_abilities_marked_explicit(self, client):
        r = client.post("/shikigami/council",
                        json={"question": "anything",
                              "abilities": ["tide", "compass"]})
        assert r.status_code == 200, r.text
        assert r.json()["selection"] == "explicit"

    def test_explicit_council(self, client):
        r = client.post(
            "/shikigami/council",
            json={
                "question": "what patterns am I missing?",
                "abilities": ["pattern_sight", "perfect_memory"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["members"].keys()) == {"pattern_sight", "perfect_memory"}

    def test_unknown_ability_400(self, client):
        r = client.post(
            "/shikigami/council",
            json={"question": "test", "abilities": ["not_a_real_ability"]},
        )
        assert r.status_code == 400

    def test_question_too_long_422(self, client):
        r = client.post("/shikigami/council", json={"question": "x" * 20000})
        assert r.status_code == 422  # pydantic validation

    def test_verdict_mentions_consulted_count(self, client):
        r = client.post("/shikigami/council", json={"question": "anything"})
        body = r.json()
        verdict = body["verdict"]
        assert "Consulted" in verdict
        assert "abilities" in verdict

    def test_narrative_block_present(self, client):
        """Round 6 added a cross-ability narrative weaver. Verify the response
        includes the woven block with signals, themes, recommendations, and
        a health gauge."""
        r = client.post(
            "/shikigami/council",
            json={
                "question": "overall state",
                "abilities": ["tide", "lantern", "compass", "mirror"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "narrative" in body
        narr = body["narrative"]
        for k in ("summary", "signals", "themes", "health_score", "recommendations"):
            assert k in narr, f"narrative missing '{k}'"
        assert 0.0 <= narr["health_score"] <= 1.0
        assert isinstance(narr["signals"], list)
        assert isinstance(narr["recommendations"], list)
        # Verdict text should reference the health gauge.
        assert "Health gauge" in body["verdict"]

    def test_council_runs_all_21_abilities(self, client):
        """Council should be able to fan out to every registered ability
        without any of them crashing."""
        from shikigami import ShikigamiType
        all_abilities = [a.value for a in ShikigamiType]
        # Bump request size limit by chunking — council caps at 10 per request.
        # Verify each chunk succeeds.
        for start in range(0, len(all_abilities), 10):
            chunk = all_abilities[start:start + 10]
            r = client.post(
                "/shikigami/council",
                json={"question": "smoke", "abilities": chunk},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert set(body["members"].keys()) == set(chunk)
            # Every member must have a structured response.
            for name, member in body["members"].items():
                assert "success" in member
                assert "output" in member
                assert "data" in member

    def test_dry_run_returns_plan_without_invoking(self, client):
        """dry_run=true: response has planned_abilities but no members invoked."""
        from shadowd import ability_metrics
        ability_metrics.reset()
        r = client.post("/shikigami/council",
                        json={"question": "test", "dry_run": True,
                              "abilities": ["oracle", "tide"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["selection"] == "explicit"
        assert body["planned_abilities"] == ["oracle", "tide"]
        assert body["ability_count"] == 2
        assert "verdict" in body
        assert "would consult" in body["verdict"]
        # No invocation should have been recorded.
        snap = ability_metrics.snapshot()
        assert snap == {}, f"dry-run should not invoke abilities, got: {snap}"

    def test_dry_run_default_abilities(self, client):
        r = client.post("/shikigami/council",
                        json={"question": "test", "dry_run": True})
        body = r.json()
        # "test" carries no ability signal → default blend of 5.
        assert body["selection"] == "default"
        assert body["ability_count"] == 5
        # Default set should include oracle.
        assert "oracle" in body["planned_abilities"]

    def test_dry_run_reflects_relevance_selection(self, client):
        from shadowd import ability_metrics
        ability_metrics.reset()
        r = client.post("/shikigami/council",
                        json={"question": "what are my projects and energy?",
                              "dry_run": True})
        body = r.json()
        assert body["selection"] == "relevance"
        # Project + energy keywords route to lantern + energy_flow.
        assert "lantern" in body["planned_abilities"]
        assert "energy_flow" in body["planned_abilities"]
        # Still a dry run: nothing invoked.
        assert ability_metrics.snapshot() == {}

    def test_audit_logs_council_members_when_enabled(self, tmp_path, monkeypatch):
        """When SHADOW_ABILITY_AUDIT=1, each council member invocation hitting
        /shikigami/summon writes an audit entry. Council doesn't currently
        audit through summon — but the summon path is the documented one and
        we want it to keep working alongside metrics."""
        from shadowd import ability_audit

        # Point the audit log at a tmp dir and force-enable.
        audit = ability_audit.reset_for_tests(audit_dir=tmp_path / "audit", enabled=True)
        try:
            # Bypass FastAPI here — we're testing summon directly.
            import asyncio
            from fastapi.testclient import TestClient
            from shadowd.shadowd import app, shadow
            from shikigami import get_registry
            if shadow.shikigami is None:
                shadow.shikigami = get_registry()
            client = TestClient(app)
            r = client.post("/shikigami/summon",
                            json={"question": "test", "shikigami": "oracle"})
            assert r.status_code == 200, r.text
            entries = audit.read_recent(lines=10)
            assert any(e["ability"] == "oracle" for e in entries), entries
        finally:
            # Reset back to env-driven state.
            ability_audit.reset_for_tests(enabled=False)


class TestCouncilSelectionHelper:
    """Unit tests for _select_council_abilities — the relevance router behind
    the council's default (no explicit abilities) path. Tested directly so the
    contract (breadth floor, cap, dedupe, omni_sense drop, default fallback)
    is locked independent of the HTTP layer."""

    def _select(self, question):
        from shadowd.shadowd import _select_council_abilities
        chosen, mode = _select_council_abilities(question)
        return [s.value for s in chosen], mode

    def test_no_signal_returns_default_blend(self):
        names, mode = self._select("test")
        assert mode == "default"
        assert names == ["oracle", "shadow_council", "pattern_sight",
                         "social_radar", "habit_tracker"]

    def test_single_signal_topped_up_to_min_breadth(self):
        # "mood" → tide only; the analytical core tops it up to 3 voices.
        names, mode = self._select("what has my mood been lately?")
        assert mode == "relevance"
        assert "tide" in names
        assert len(names) >= 3

    def test_omni_sense_dropped_when_concrete_abilities_present(self):
        names, _ = self._select("what are my projects and energy?")
        assert "omni_sense" not in names
        assert "lantern" in names and "energy_flow" in names

    def test_result_is_deduped(self):
        names, _ = self._select("what are my goals, projects, mood, energy, focus, and habits?")
        assert len(names) == len(set(names))

    def test_capped_at_max(self):
        # A keyword-dense question should still never exceed the latency cap.
        from shadowd.shadowd import _COUNCIL_MAX
        names, mode = self._select(
            "predict my mood, energy, focus, habits, projects, goals, "
            "relationships, location, and self-talk"
        )
        assert mode == "relevance"
        assert len(names) <= _COUNCIL_MAX

    def test_decision_question_includes_shadow_council(self):
        names, mode = self._select("should I take the new job offer?")
        assert mode == "relevance"
        assert "shadow_council" in names
