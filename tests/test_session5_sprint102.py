#!/usr/bin/env python3
"""
Sprint 102 Tests — ShadowArise Universal Agent Creation Engine

Tests for:
1. ShadowArise core: arise, configure, activate, execute, evolve, complete, archive
2. ShadowArise templates: builtin, custom, recommend
3. ShadowArise limits, sanitization, thread safety
4. ShadowOS integration: arise_agent, execute_agent_step, evolve_agent, get_arise_stats
5. API endpoints: /arise, /arise/execute, /arise/evolve, /arise/complete, /arise/{id}, /arise/list, /arise/templates, /arise/stats
6. Data class to_dict immutability
"""

import copy
import re
import threading
import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.shadow_arise import (
    ShadowArise, AgentLifecycle, CapabilityType, MissionPriority,
    AgentStatus, AgentTemplate, AgentInstance, Capability,
    MissionContext, ExecutionStep, _sanitize, _sanitize_content,
)


@pytest.fixture
def engine():
    return ShadowArise()


class TestShadowAriseCore:
    def test_arise_creates_agent(self, engine):
        agent = engine.arise(name="TestAgent", objective="Analyze data patterns")
        assert agent.agent_id
        assert agent.name == "TestAgent"
        assert agent.mission.objective == "Analyze data patterns"
        assert agent.lifecycle == AgentLifecycle.ARISE
        assert agent.status == AgentStatus.PENDING
        assert len(agent.capabilities) > 0

    def test_arise_with_template(self, engine):
        agent = engine.arise(name="Analyst", objective="Research topic", template_id="analyst")
        assert agent.template_id == "analyst"
        assert any(c.capability_type == CapabilityType.ANALYZE for c in agent.capabilities)

    def test_arise_custom_capabilities(self, engine):
        caps = [CapabilityType.REASON, CapabilityType.CREATE]
        agent = engine.arise(name="Creator", objective="Make something", capabilities=caps)
        assert len(agent.capabilities) == 2
        assert any(c.capability_type == CapabilityType.REASON for c in agent.capabilities)
        assert any(c.capability_type == CapabilityType.CREATE for c in agent.capabilities)

    def test_arise_with_priority(self, engine):
        agent = engine.arise(name="Urgent", objective="Fix bug", priority=MissionPriority.CRITICAL)
        assert agent.mission.priority == MissionPriority.CRITICAL

    def test_arise_with_constraints_and_criteria(self, engine):
        agent = engine.arise(
            name="Bounded", objective="Task",
            constraints=["No cloud APIs"], success_criteria=["All tests pass"],
        )
        assert "No cloud APIs" in agent.mission.constraints
        assert "All tests pass" in agent.mission.success_criteria

    def test_arise_sanitizes_name(self, engine):
        agent = engine.arise(name="<script>alert(1)</script>Test", objective="Obj")
        assert "<script>" not in agent.name
        assert "Test" in agent.name

    def test_arise_sanitizes_objective(self, engine):
        agent = engine.arise(name="A", objective="Hello\x00world<script>bad</script>test")
        assert "\x00" not in agent.mission.objective

    def test_arise_default_capabilities(self, engine):
        agent = engine.arise(name="Default", objective="Do something")
        assert len(agent.capabilities) >= 2

    def test_arise_confidence_score(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        assert 0 < agent.confidence_score <= 1.0


class TestShadowAriseConfigure:
    def test_configure_adds_capabilities(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        result = engine.configure(agent.agent_id, capabilities=[CapabilityType.PLAN])
        assert result is not None
        assert any(c.capability_type == CapabilityType.PLAN for c in result.capabilities)

    def test_configure_overrides_proficiency(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        result = engine.configure(agent.agent_id, proficiency_overrides={"reason": 0.95})
        assert result is not None
        for cap in result.capabilities:
            if cap.name == "reason":
                assert cap.proficiency == 0.95

    def test_configure_nonexistent_agent(self, engine):
        result = engine.configure("nonexistent123", capabilities=[CapabilityType.PLAN])
        assert result is None

    def test_configure_wrong_lifecycle(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.configure(agent.agent_id, capabilities=[CapabilityType.PLAN])
        assert result is None


class TestShadowAriseActivate:
    def test_activate_transitions_state(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        result = engine.activate(agent.agent_id)
        assert result.lifecycle == AgentLifecycle.ACTIVATE
        assert result.status == AgentStatus.ACTIVE

    def test_activate_nonexistent(self, engine):
        result = engine.activate("nonexistent123")
        assert result is None


class TestShadowAriseExecute:
    def test_execute_step(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        step = engine.execute_step(agent.agent_id, "Analyze data", CapabilityType.ANALYZE)
        assert step is not None
        assert step.capability_used == CapabilityType.ANALYZE
        assert step.description == "Analyze data"

    def test_execute_updates_proficiency(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        original_prof = None
        for cap in engine.get_agent(agent.agent_id).capabilities:
            if cap.capability_type == CapabilityType.REASON:
                original_prof = cap.proficiency
                break
        engine.execute_step(agent.agent_id, "Think", CapabilityType.REASON)
        agent_after = engine.get_agent(agent.agent_id)
        for cap in agent_after.capabilities:
            if cap.capability_type == CapabilityType.REASON:
                assert cap.proficiency > original_prof
                assert cap.usage_count == 1

    def test_execute_nonexistent_agent(self, engine):
        step = engine.execute_step("nonexistent123", "Desc", CapabilityType.REASON)
        assert step is None

    def test_execute_inactive_agent(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        step = engine.execute_step(agent.agent_id, "Desc", CapabilityType.REASON)
        assert step is None


class TestShadowAriseEvolve:
    def test_evolve_adds_capabilities(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.evolve(agent.agent_id, "Need planning",
                              new_capabilities=[CapabilityType.PLAN])
        assert result is not None
        assert any(c.capability_type == CapabilityType.PLAN for c in result.capabilities)
        assert result.evolution_count == 1

    def test_evolve_adjusts_proficiency(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        original_prof = None
        for cap in engine.get_agent(agent.agent_id).capabilities:
            if cap.name == "reason":
                original_prof = cap.proficiency
                break
        result = engine.evolve(agent.agent_id, "Better reasoning",
                              capability_adjustments={"reason": 0.3})
        assert result is not None
        for cap in result.capabilities:
            if cap.name == "reason":
                assert cap.proficiency > original_prof

    def test_evolve_caps_adjustments(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.evolve(agent.agent_id, "boost", capability_adjustments={"reason": 10.0})
        assert result is not None
        for cap in result.capabilities:
            if cap.name == "reason":
                assert cap.proficiency <= 1.0


class TestShadowAriseComplete:
    def test_complete_agent(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.complete(agent.agent_id, result={"answer": "42"})
        assert result.lifecycle == AgentLifecycle.COMPLETE
        assert result.status == AgentStatus.COMPLETED
        assert result.result == {"answer": "42"}

    def test_complete_failed(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.complete(agent.agent_id, status=AgentStatus.FAILED)
        assert result.status == AgentStatus.FAILED

    def test_complete_nonexistent(self, engine):
        result = engine.complete("nonexistent123")
        assert result is None


class TestShadowAriseArchive:
    def test_archive_agent(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.complete(agent.agent_id)
        result = engine.archive(agent.agent_id)
        assert result.lifecycle == AgentLifecycle.ARCHIVE
        assert result.status == AgentStatus.ARCHIVED

    def test_archive_nonexistent(self, engine):
        result = engine.archive("nonexistent123")
        assert result is None


class TestShadowAriseTemplates:
    def test_builtin_templates_exist(self, engine):
        templates = engine.list_templates()
        assert len(templates) >= 8
        ids = [t.template_id for t in templates]
        assert "analyst" in ids
        assert "creator" in ids
        assert "planner" in ids
        assert "executor" in ids

    def test_get_template(self, engine):
        t = engine.get_template("analyst")
        assert t is not None
        assert t.name == "Analyst"

    def test_get_nonexistent_template(self, engine):
        t = engine.get_template("nonexistent_template")
        assert t is None

    def test_add_custom_template(self, engine):
        t = AgentTemplate(
            template_id="custom_agent",
            name="CustomAgent",
            description="Custom agent template",
            capabilities=[CapabilityType.REASON, CapabilityType.CREATE],
            default_proficiency=0.7,
            mission_types=["custom"],
            metadata={},
        )
        result = engine.add_template(t)
        assert result.template_id == "custom_agent"
        fetched = engine.get_template("custom_agent")
        assert fetched is not None

    def test_recommend_template(self, engine):
        template = engine.recommend_template("Analyze the data patterns")
        assert template is not None
        assert template.template_id == "analyst"

    def test_recommend_template_creator(self, engine):
        template = engine.recommend_template("Create a new design")
        assert template is not None
        assert template.template_id == "creator"

    def test_recommend_template_no_match(self, engine):
        template = engine.recommend_template("zzzzzzzz unmatchable query xyzzy")
        assert template is not None


class TestShadowAriseList:
    def test_list_agents(self, engine):
        engine.arise(name="A1", objective="Obj1")
        engine.arise(name="A2", objective="Obj2")
        agents = engine.list_agents()
        assert len(agents) == 2

    def test_list_agents_by_status(self, engine):
        engine.arise(name="A1", objective="Obj1")
        a2 = engine.arise(name="A2", objective="Obj2")
        engine.activate(a2.agent_id)
        pending = engine.list_agents(status=AgentStatus.PENDING)
        assert len(pending) == 1

    def test_list_agents_by_lifecycle(self, engine):
        a1 = engine.arise(name="A1", objective="Obj1")
        engine.activate(a1.agent_id)
        active = engine.list_agents(lifecycle=AgentLifecycle.ACTIVATE)
        assert len(active) >= 1

    def test_list_agents_limit(self, engine):
        for i in range(5):
            engine.arise(name=f"Agent{i}", objective="Obj")
        agents = engine.list_agents(limit=3)
        assert len(agents) == 3


class TestShadowAriseLimits:
    def test_max_agents_cap(self, engine):
        engine._MAX_AGENTS = 10
        for i in range(12):
            engine.arise(name=f"Agent{i}", objective="Obj")
        agents = engine.list_agents()
        assert len(agents) <= 10

    def test_sanitize_identifiers(self):
        result = _sanitize("<script>alert(1)</script>test\x00name", 50)
        assert "<script>" not in result
        assert "\x00" not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        result = _sanitize_content("latency < 100ms is good\x00bad", 2000)
        assert "<" in result
        assert "\x00" not in result

    def test_sanitize_content_truncation(self):
        result = _sanitize_content("x" * 5000, 100)
        assert len(result) <= 100

    def test_constraints_cap(self, engine):
        agent = engine.arise(
            name="Capped", objective="Obj",
            constraints=[f"constraint {i}" for i in range(20)],
        )
        assert len(agent.mission.constraints) <= 10

    def test_success_criteria_cap(self, engine):
        agent = engine.arise(
            name="Capped", objective="Obj",
            success_criteria=[f"criterion {i}" for i in range(20)],
        )
        assert len(agent.mission.success_criteria) <= 10


class TestShadowAriseThreadSafety:
    def test_concurrent_arise(self, engine):
        errors = []
        agents = []

        def create_agent(i):
            try:
                agent = engine.arise(name=f"Concurrent{i}", objective=f"Objective {i}")
                agents.append(agent.agent_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_agent, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(agents) == 50

    def test_concurrent_execute(self, engine):
        agent = engine.arise(name="Concurrent", objective="Obj")
        engine.activate(agent.agent_id)
        errors = []

        def execute_step(i):
            try:
                engine.execute_step(agent.agent_id, f"Step {i}", CapabilityType.ANALYZE)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=execute_step, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestShadowAriseStats:
    def test_stats_initial(self, engine):
        stats = engine.get_stats()
        assert stats["agents_created"] == 0
        assert stats["active_agents"] == 0

    def test_stats_after_arise(self, engine):
        engine.arise(name="Test", objective="Obj")
        stats = engine.get_stats()
        assert stats["agents_created"] == 1
        assert stats["active_agents"] == 1

    def test_stats_after_complete(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        engine.complete(agent.agent_id)
        stats = engine.get_stats()
        assert stats["agents_completed"] == 1

    def test_stats_after_evolve(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        engine.evolve(agent.agent_id, "feedback")
        stats = engine.get_stats()
        assert stats["agents_evolved"] == 1


class TestDataClassImmutability:
    def test_capability_to_dict_copy(self):
        cap = Capability(name="reason", capability_type=CapabilityType.REASON,
                         proficiency=0.8, description="Test", configuration={"a": 1}, usage_count=0)
        d = cap.to_dict()
        d["configuration"]["b"] = 2
        assert "b" not in cap.configuration

    def test_agent_template_to_dict_copy(self):
        t = AgentTemplate(template_id="test", name="Test", description="Desc",
                          capabilities=[CapabilityType.REASON], default_proficiency=0.7,
                          mission_types=["test"], metadata={"key": "val"})
        d = t.to_dict()
        d["metadata"]["key2"] = "val2"
        assert "key2" not in t.metadata

    def test_mission_context_to_dict_copy(self):
        m = MissionContext(objective="Obj", priority=MissionPriority.HIGH,
                          constraints=["c1"], success_criteria=["s1"],
                          deadline=None, context_data={"k": "v"})
        d = m.to_dict()
        d["constraints"].append("c2")
        assert "c2" not in m.constraints

    def test_agent_instance_to_dict_deep_copy(self):
        engine = ShadowArise()
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        fetched = engine.get_agent(agent.agent_id)
        d = fetched.to_dict()
        d["mission"]["constraints"].append("injected")
        d["capabilities"][0]["name"] = "tampered"
        refetched = engine.get_agent(agent.agent_id)
        assert "injected" not in refetched.mission.constraints
        assert refetched.capabilities[0].name == "reason" or refetched.capabilities[0].name != "tampered"

    def test_execution_step_to_dict_copy(self):
        step = ExecutionStep(
            step_id="step123", description="test", capability_used=CapabilityType.REASON,
            input_data={"k": "v"}, output_data={"r": "v2"},
            confidence=0.8, timestamp="2024-01-01", duration_ms=100,
        )
        d = step.to_dict()
        d["input_data"]["k2"] = "v3"
        assert "k2" not in step.input_data


class TestShadowOSIntegration:
    def test_shadowos_has_arise(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        assert hasattr(os_instance, 'arise')
        assert isinstance(os_instance.arise, ShadowArise)

    def test_arise_agent_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        result = os_instance.arise_agent(
            name="TestAgent", objective="Analyze patterns",
            template_id="analyst",
        )
        assert result["name"] == "TestAgent"
        assert result["mission"]["objective"] == "Analyze patterns"

    def test_execute_agent_step_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        agent = os_instance.arise.arise(name="TestAgent", objective="Obj")
        os_instance.arise.activate(agent.agent_id)
        result = os_instance.execute_agent_step(
            agent_id=agent.agent_id, description="Step 1",
            capability="analyze", confidence=0.8,
        )
        assert result["step_id"]
        assert result["description"] == "Step 1"

    def test_evolve_agent_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        agent = os_instance.arise.arise(name="TestAgent", objective="Obj")
        os_instance.arise.activate(agent.agent_id)
        result = os_instance.evolve_agent(
            agent_id=agent.agent_id, feedback="Need improvement",
            adjustments={"reason": 0.1},
        )
        assert result["evolution_count"] == 1

    def test_get_arise_stats_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        stats = os_instance.get_arise_stats()
        assert "agents_created" in stats
        assert "templates_available" in stats

    def test_arise_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        assert hasattr(os_instance, 'arise')

    def test_arise_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint102_os")
        status = os_instance.get_status()
        assert "arise_stats" in status


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from shadowd.cognition.api_router import router, _shadow_os, _shadow_os_lock
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_arise import ShadowArise as AriseEngine
        import shadowd.cognition.api_router as api_module

        new_os = ShadowOS.__new__(ShadowOS)
        new_os.arise = AriseEngine()

        with api_module._shadow_os_lock:
            api_module._shadow_os = new_os

        app = __import__("fastapi").FastAPI()
        app.include_router(router)
        tc = TestClient(app)
        yield tc

        with api_module._shadow_os_lock:
            api_module._shadow_os = None

    def test_arise_create_endpoint(self, client):
        resp = client.post("/os/arise", json={
            "name": "APIAgent", "objective": "Test via API",
            "template_id": "analyst",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "APIAgent"
        assert data["mission"]["objective"] == "Test via API"

    def test_arise_execute_endpoint(self, client):
        resp = client.post("/os/arise", json={"name": "ExecTest", "objective": "Obj"})
        agent_id = resp.json()["agent_id"]
        import shadowd.cognition.api_router as api_module
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        os_obj.arise.activate(agent_id)
        resp2 = client.post("/os/arise/execute", json={
            "agent_id": agent_id, "description": "Analyze",
            "capability": "analyze",
        })
        assert resp2.status_code == 200
        data = resp2.json()
        assert data.get("step_id") is not None

    def test_arise_evolve_endpoint(self, client):
        resp = client.post("/os/arise", json={"name": "EvolTest", "objective": "Obj"})
        agent_id = resp.json()["agent_id"]
        resp2 = client.post("/os/arise/evolve", json={
            "agent_id": agent_id, "feedback": "Improve reasoning",
        })
        assert resp2.status_code == 200

    def test_arise_get_agent_endpoint(self, client):
        resp = client.post("/os/arise", json={"name": "GetTest", "objective": "Obj"})
        agent_id = resp.json()["agent_id"]
        resp2 = client.get(f"/os/arise/agent/{agent_id}")
        assert resp2.status_code == 200
        assert resp2.json()["agent_id"] == agent_id

    def test_arise_get_nonexistent_agent(self, client):
        resp = client.get("/os/arise/agent/nonexistent123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_arise_list_agents_endpoint(self, client):
        client.post("/os/arise", json={"name": "ListTest", "objective": "Obj"})
        resp = client.get("/os/arise")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_arise_templates_endpoint(self, client):
        resp = client.get("/os/arise/templates")
        assert resp.status_code == 200
        templates = resp.json()
        assert len(templates) >= 8

    def test_arise_stats_endpoint(self, client):
        resp = client.get("/os/arise/stats")
        assert resp.status_code == 200
        assert "agents_created" in resp.json()

    def test_arise_invalid_priority(self, client):
        resp = client.post("/os/arise", json={
            "name": "BadPriority", "objective": "Obj", "priority": "invalid",
        })
        assert resp.status_code == 422

    def test_arise_invalid_capability(self, client):
        resp = client.post("/os/arise", json={
            "name": "BadCap", "objective": "Obj",
            "capabilities": ["nonexistent_capability"],
        })
        assert resp.status_code == 422

    def test_arise_complete_endpoint(self, client):
        resp = client.post("/os/arise", json={"name": "CompTest", "objective": "Obj"})
        agent_id = resp.json()["agent_id"]
        resp2 = client.post("/os/arise/complete", json={
            "agent_id": agent_id, "status": "completed",
            "result": {"answer": 42},
        })
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "completed"

    def test_arise_with_custom_capabilities(self, client):
        resp = client.post("/os/arise", json={
            "name": "CustomCap", "objective": "Obj",
            "capabilities": ["reason", "create", "analyze"],
        })
        assert resp.status_code == 200
        assert len(resp.json()["capabilities"]) == 3

    def test_arise_with_constraints(self, client):
        resp = client.post("/os/arise", json={
            "name": "Constrained", "objective": "Obj",
            "constraints": ["No cloud APIs"], "success_criteria": ["All tests pass"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["mission"]["constraints"]) == 1
        assert len(data["mission"]["success_criteria"]) == 1

    def test_arise_xss_sanitization(self, client):
        resp = client.post("/os/arise", json={
            "name": "<script>alert(1)</script>Evil",
            "objective": "Test\x00content with control chars",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "<script>" not in data["name"]
        assert "\x00" not in data["mission"]["objective"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])