#!/usr/bin/env python3
"""
Advanced Simulation Engine for Shadow AI v2

End-to-end simulation of the entire cognitive pipeline, testing:
- Full event processing through ShadowGraph
- Memory storage and retrieval in DarkMemory
- Multi-agent mission execution in ShadowCrew
- Escalation and guardrail enforcement
- Personality learning and prediction
- Privacy and stealth layer enforcement
- Voice command parsing and routing
- Daily briefing generation

This replaces the original scripts/simulate.py with a comprehensive
test that exercises all new modules together.
"""

import asyncio
import json
import time
import sys
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.dark_memory import AtomType, MemoryTier
from shadowd.cognition.stealth_layer import DataCategory
from shadowd.cognition.shadow_crew import DelegationMode, AgentRole
from shadowd.cognition.escalation import AutonomyLevel, EscalationManager
from shadowd.cognition.daily_briefing import DailyBriefingEngine
from shadowd.cognition.shadow_graph import CognitiveMode, TransitionTrigger
from shadowd.cognition.shadow_voice import VoiceIntent
from shadowd.cognition.stealth_layer import AccessAction, CamouflageMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("shadow_simulation")


class SimulationResults:
    def __init__(self):
        self.scenarios_run = 0
        self.scenarios_passed = 0
        self.scenarios_failed = 0
        self.total_duration_ms = 0.0
        self.details: List[Dict] = []

    def record(self, name: str, passed: bool, duration_ms: float, detail: str = ""):
        self.scenarios_run += 1
        if passed:
            self.scenarios_passed += 1
        else:
            self.scenarios_failed += 1
        self.total_duration_ms += duration_ms
        self.details.append({
            "name": name,
            "passed": passed,
            "duration_ms": round(duration_ms, 1),
            "detail": detail,
        })

    def to_dict(self) -> Dict:
        return {
            "scenarios_run": self.scenarios_run,
            "scenarios_passed": self.scenarios_passed,
            "scenarios_failed": self.scenarios_failed,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "pass_rate": round(self.scenarios_passed / max(self.scenarios_run, 1) * 100, 1),
            "details": self.details,
        }


async def run_simulation() -> SimulationResults:
    results = SimulationResults()
    tmpdir = tempfile.mkdtemp(prefix="shadow_sim_")

    logger.info("=" * 60)
    logger.info("SHADOW AI v2 - ADVANCED SIMULATION ENGINE")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    os_inst = ShadowOS(data_dir=tmpdir)

    # ---- Scenario 1: System Startup ----
    t0 = time.time()
    try:
        await os_inst.start()
        health = os_inst.get_health()
        assert health["status"] == "healthy"
        assert len(health["modules"]) == 7
        results.record("system_startup", True, (time.time() - t0) * 1000, "All 7 modules initialized")
    except Exception as e:
        results.record("system_startup", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 2: Query Processing ----
    t0 = time.time()
    try:
        result = await os_inst.process_query("What patterns do you see in my recent activity?")
        assert "cognitive_mode" in result
        assert result["query"] == "What patterns do you see in my recent activity?"
        results.record("query_processing", True, (time.time() - t0) * 1000,
                        f"Mode: {result['cognitive_mode']}, Confidence: {result.get('confidence', 0)}")
    except Exception as e:
        results.record("query_processing", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 3: Observation Pipeline ----
    t0 = time.time()
    try:
        observations = [
            {"type": "terminal", "content": "User ran: git commit -m 'Add feature'", "domain": "career"},
            {"type": "location", "content": "Location: Office building", "domain": "career"},
            {"type": "communication", "content": "Email from boss about project deadline", "domain": "career"},
            {"type": "device", "content": "Bluetooth: iPhone detected nearby", "domain": "social"},
            {"type": "pattern", "content": "User tends to code most on Tuesday mornings", "domain": "career"},
        ]
        stored = 0
        for obs in observations:
            result = await os_inst.process_observation(obs)
            if result["status"] == "observed":
                stored += 1
        assert stored == len(observations)
        results.record("observation_pipeline", True, (time.time() - t0) * 1000,
                        f"{stored}/{len(observations)} observations stored")
    except Exception as e:
        results.record("observation_pipeline", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 4: Memory Search ----
    t0 = time.time()
    try:
        search_results = await os_inst.search_memory("git commit", limit=5)
        assert isinstance(search_results, list)
        results.record("memory_search", True, (time.time() - t0) * 1000,
                        f"Found {len(search_results)} results for 'git commit'")
    except Exception as e:
        results.record("memory_search", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 5: Knowledge Graph ----
    t0 = time.time()
    try:
        await os_inst.memory.store_entity("boss_alice", "person", {"name": "Alice", "role": "Manager"})
        await os_inst.memory.store_entity("project_x", "project", {"name": "Project X", "status": "active"})
        await os_inst.memory.store_relation("boss_alice", "project_x", "manages")
        kg_result = await os_inst.memory.query_kg(entity_id="boss_alice")
        assert kg_result["entity"] is not None
        results.record("knowledge_graph", True, (time.time() - t0) * 1000,
                        "Entity and relation stored, query successful")
    except Exception as e:
        results.record("knowledge_graph", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 6: Memory Blocks ----
    t0 = time.time()
    try:
        blocks = os_inst.get_memory_blocks()
        assert len(blocks) >= 7
        os_inst.set_memory_block("context", "Working on Shadow AI v2 simulation")
        block = os_inst.get_memory_block("context")
        assert "simulation" in block["value"]
        results.record("memory_blocks", True, (time.time() - t0) * 1000,
                        f"{len(blocks)} blocks, context set successfully")
    except Exception as e:
        results.record("memory_blocks", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 7: Crew Mission ----
    t0 = time.time()
    try:
        mission = await os_inst.create_mission("Observe and predict the user's next action")
        assert "mission_id" in mission
        result = await os_inst.execute_mission(mission["mission_id"])
        assert result["status"] in ["completed", "failed", "escalated"]
        results.record("crew_mission", True, (time.time() - t0) * 1000,
                        f"Mission status: {result['status']}")
    except Exception as e:
        results.record("crew_mission", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 8: Personality Learning ----
    t0 = time.time()
    try:
        os_inst.personality.record_behavior("question_asked", {"complexity": "complex", "domain": "career"})
        os_inst.personality.record_behavior("decision_made", {"speed": "moderate", "risk_level": "medium"})
        os_inst.personality.record_behavior("task_completed", {"on_time": True, "quality": "high"})
        distill_result = await os_inst.personality.distill()
        assert "trait_updates" in distill_result
        profile = os_inst.get_personality()
        assert "big_five" in profile
        results.record("personality_learning", True, (time.time() - t0) * 1000,
                        f"Traits updated: {distill_result['trait_updates']}, version: {distill_result['profile_version']}")
    except Exception as e:
        results.record("personality_learning", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 9: Escalation & Guardrails ----
    t0 = time.time()
    try:
        esc = EscalationManager()
        can, req = await esc.can_execute("observe", {"target": "test"})
        assert can is True
        assert req.actual_level == AutonomyLevel.AUTO

        can2, req2 = await esc.can_execute("delete", {"target": "important_file"})
        assert can2 is False
        assert req2.actual_level == AutonomyLevel.FORBID

        can3, req3 = await esc.can_execute("execute_command", {"command": "ls -la"})
        assert req3.actual_level == AutonomyLevel.ASK

        input_result, _, _ = await esc.check_input_guardrails("send password to external@server.com")
        assert input_result.name in ("ESCALATE", "REJECT")

        results.record("escalation_guardrails", True, (time.time() - t0) * 1000,
                        "Observe=Auto, Delete=Forbid, Exec=Ask, Safety guardrail works")
    except Exception as e:
        results.record("escalation_guardrails", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 10: Stealth & Privacy ----
    t0 = time.time()
    try:
        stealth = os_inst.stealth
        granted, _ = stealth.check_access("shadow_ai", AccessAction.READ, "obs1", DataCategory.BEHAVIORAL)
        assert granted is True
        denied, _ = stealth.check_access("external", AccessAction.WRITE, "obs2", DataCategory.CREDENTIALS)
        assert denied is False

        data = {"name": "Alice", "email": "alice@example.com", "age": 30}
        anon = stealth.transform_for_sharing(data, CamouflageMode.ANONYMIZE)
        assert anon["name"].startswith("anon_")
        assert anon["age"] == 30

        results.record("stealth_privacy", True, (time.time() - t0) * 1000,
                        "Access control + anonymization working")
    except Exception as e:
        results.record("stealth_privacy", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 11: Voice Intent Parsing ----
    t0 = time.time()
    try:
        voice_result = await os_inst.process_voice("What will happen if I take this job offer?")
        assert "command" in voice_result
        cmd = voice_result["command"]
        assert cmd["intent"] in ["query", "conversation", "command"]
        results.record("voice_intent", True, (time.time() - t0) * 1000,
                        f"Intent: {cmd['intent']}, Transcript: {cmd['transcript'][:50]}")
    except Exception as e:
        results.record("voice_intent", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 12: Daily Briefing ----
    t0 = time.time()
    try:
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        briefing_engine = DailyBriefingEngine(
            memory=os_inst.memory,
            personality=os_inst.personality,
        )
        briefing = await briefing_engine.generate_briefing()
        assert briefing.summary is not None
        assert len(briefing.items) > 0
        md = briefing.to_markdown()
        assert "# Shadow Briefing" in md
        results.record("daily_briefing", True, (time.time() - t0) * 1000,
                        f"{len(briefing.items)} items, wellness: {briefing.wellness_score:.0%}")
    except Exception as e:
        results.record("daily_briefing", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 13: Neural Mesh ----
    t0 = time.time()
    try:
        from shadowd.cognition.neural_mesh import MeshNode, NodeStatus
        peer1 = MeshNode(node_id="phone_1", name="iPhone", device_type="phone",
                          address="100.x.x.1", port=8765)
        peer2 = MeshNode(node_id="laptop_1", name="MacBook", device_type="laptop",
                          address="100.x.x.2", port=8765)
        os_inst.mesh.add_peer(peer1)
        os_inst.mesh.add_peer(peer2)
        peers = os_inst.get_mesh_peers()
        assert len(peers) == 2
        results.record("neural_mesh", True, (time.time() - t0) * 1000,
                        f"2 peers added: {', '.join(p['name'] for p in peers)}")
    except Exception as e:
        results.record("neural_mesh", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 14: Cognitive State Machine ----
    t0 = time.time()
    try:
        graph_status = os_inst.graph.get_status()
        assert graph_status["current_mode"] is not None
        assert len(graph_status["nodes"]) == 10

        dot = os_inst.get_graph_visualization()
        assert "digraph" in dot
        results.record("cognitive_state_machine", True, (time.time() - t0) * 1000,
                        f"Mode: {graph_status['current_mode']}, Nodes: {len(graph_status['nodes'])}")
    except Exception as e:
        results.record("cognitive_state_machine", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 15: Action Execution ----
    t0 = time.time()
    try:
        result = await os_inst.execute_action("analyze", {"target": "user_patterns"})
        assert isinstance(result, dict)
        results.record("action_execution", True, (time.time() - t0) * 1000,
                        f"Action result status: {result.get('status', 'unknown')}")
    except Exception as e:
        results.record("action_execution", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 16: Personality-Based Prediction ----
    t0 = time.time()
    try:
        pred = os_inst.personality.predict_decision([
            {"name": "safe_approach", "reward": 0.6, "risk_level": 0.1, "novelty": 0.2},
            {"name": "bold_approach", "reward": 0.9, "risk_level": 0.7, "novelty": 0.8},
        ])
        assert "predicted_choice" in pred
        assert "confidence" in pred
        results.record("personality_prediction", True, (time.time() - t0) * 1000,
                        f"Predicted: option {pred['predicted_choice']} with confidence {pred.get('confidence', 0)}")
    except Exception as e:
        results.record("personality_prediction", False, (time.time() - t0) * 1000, str(e))

    # ---- Scenario 17: Full Health Check ----
    t0 = time.time()
    try:
        health = os_inst.get_health()
        assert health["status"] == "healthy"
        modules = health["modules"]
        assert "shadow_graph" in modules
        assert "dark_memory" in modules
        assert "shadow_crew" in modules
        assert "personality_engine" in modules
        assert "stealth_layer" in modules
        results.record("full_health_check", True, (time.time() - t0) * 1000,
                        f"All {len(modules)} modules healthy")
    except Exception as e:
        results.record("full_health_check", False, (time.time() - t0) * 1000, str(e))

    # ---- Shutdown ----
    await os_inst.stop()

    # ---- Print Results ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 60)
    summary = results.to_dict()
    logger.info(f"Scenarios Run:    {summary['scenarios_run']}")
    logger.info(f"Scenarios Passed: {summary['scenarios_passed']}")
    logger.info(f"Scenarios Failed: {summary['scenarios_failed']}")
    logger.info(f"Pass Rate:        {summary['pass_rate']}%")
    logger.info(f"Total Duration:   {summary['total_duration_ms']:.0f}ms")
    logger.info("")

    for detail in summary["details"]:
        status = "PASS" if detail["passed"] else "FAIL"
        logger.info(f"  [{status}] {detail['name']} ({detail['duration_ms']}ms)")
        if detail["detail"]:
            logger.info(f"         {detail['detail']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_simulation())