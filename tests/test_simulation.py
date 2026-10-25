"""
Simulation engine unit tests - verify the simulation harness components.
These test the simulate.py script's logic without requiring a running daemon.
"""

import pytest
import json
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestSimulationEngine:
    def test_simulation_script_exists(self):
        path = Path(_repo_file("scripts/simulate.py"))
        assert path.exists()
        content = path.read_text()
        assert "async def run_simulation" in content
        assert "test_health" in content
        assert "test_ask" in content

    def test_web_dashboard_exists(self):
        path = Path(_repo_file("web/index.html"))
        assert path.exists()
        content = path.read_text()
        assert "Shadow AI" in content
        assert "ask/stream" in content or "/ask" in content
        assert "memory" in content.lower()

    def test_dockerfile_exists(self):
        path = Path(_repo_file("Dockerfile"))
        assert path.exists()
        content = path.read_text()
        assert "python:3.12" in content
        assert "8765" in content

    def test_systemd_service_exists(self):
        path = Path(_repo_file("digital-shadow.service"))
        assert path.exists()
        content = path.read_text()
        assert "PYTHONUNBUFFERED" in content
        assert "SHADOW_DATA_DIR" in content
        assert "PrivateTmp=true" in content

    def test_requirements_complete(self):
        path = Path(_repo_file("requirements.txt"))
        content = path.read_text()
        required = ["fastapi", "uvicorn", "httpx", "pydantic", "websockets", "pyyaml"]
        for pkg in required:
            assert pkg in content, f"Missing package: {pkg}"

    def test_env_file_has_cors(self):
        path = Path(_repo_file(".env"))
        if not path.exists():
            pytest.skip(".env not present (secrets are never committed)")
        content = path.read_text()
        assert "SHADOW_CORS_ORIGINS" in content

    def test_gitignore_exists(self):
        path = Path(_repo_file(".gitignore"))
        assert path.exists()

    def test_gitignore_covers_secrets(self):
        path = Path(_repo_file(".gitignore"))
        content = path.read_text()
        assert ".env" in content


class TestDaemonStartup:
    """Test that the daemon can be imported and basic components initialize."""

    def test_import_shadowd(self):
        from shadowd import shadowd
        assert hasattr(shadowd, 'app')
        assert shadowd.DATA_DIR.exists()

    def test_import_shadow_ai(self):
        from shadowd.shadow_ai import ShadowAI
        assert hasattr(ShadowAI, '__init__')
        assert hasattr(ShadowAI, 'observe')
        assert hasattr(ShadowAI, 'ask')
        assert hasattr(ShadowAI, 'act')

    def test_import_mind(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        assert hasattr(engine, 'think')
        assert hasattr(engine, 'context_budget')

    def test_import_event_bus(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent, get_event_bus
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_import_stuck_detector(self):
        from shadowd.reasoning.stuck_detector import StuckDetector, LoopType, StuckAnalysis, ActionType
        sd = StuckDetector()
        assert hasattr(sd, 'analyze')
        assert hasattr(sd, 'record_action')
        assert hasattr(sd, 'record_thought')

    def test_import_memory_tools(self):
        from shadowd.memory_tools import MemoryTools, ContextWindowBudget, MemoryEditResult, render_memory_xml
        budget = ContextWindowBudget()
        assert budget.max_context_tokens == 8192

    def test_import_config(self):
        from shadowd.config import ShadowConfig, get_config
        config = ShadowConfig()
        assert config.llm.model == "glm-5"
        assert config.server.port == 8765

    def test_import_streaming(self):
        from shadowd.streaming import generate_sse_stream, stream_llm_response
        assert callable(generate_sse_stream)
        assert callable(stream_llm_response)

    def test_import_block_memory(self):
        from shadowd.block_memory import BlockBasedMemory, CoreMemory, WorkingMemory, ArchivalMemory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mem = BlockBasedMemory(data_dir=Path(tmp))
            assert mem.core_memory is not None
            assert hasattr(mem, 'core_set')
            assert hasattr(mem, 'core_get')
            assert hasattr(mem, 'recall')


class TestEnhancedAbilitiesInit:
    """Test that EnhancedShikigamiRegistry initializes all superhuman algorithms."""

    def test_registry_has_superhuman_attrs(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        reg = EnhancedShikigamiRegistry()
        assert hasattr(reg, 'bayesian')
        assert hasattr(reg, 'signal_triangulator')
        assert hasattr(reg, 'knowledge_graph')
        assert hasattr(reg, 'markov')
        assert hasattr(reg, 'rl_learner')
        assert hasattr(reg, 'env_model')

    def test_bayesian_has_hypotheses(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        reg = EnhancedShikigamiRegistry()
        assert len(reg.bayesian.hypotheses) >= 4
        assert "user_active" in reg.bayesian.hypotheses

    def test_env_model_observe(self):
        from shikigami.superhuman import EnvironmentModel
        model = EnvironmentModel()
        model.observe({"type": "terminal_command", "content": {"cmd": "ls"}})
        assert len(model.observations) == 1

    def test_env_model_snapshot(self):
        from shikigami.superhuman import EnvironmentModel
        model = EnvironmentModel()
        snap = model.get_environment_snapshot(
            devices=[{"mac": "AA:BB", "name": "phone", "known": False, "person": None}],
            networks=[{"ssid": "home-network", "signal": "-50"}],
            activities=["terminal_command"]
        )
        assert snap["time"]["context"] in ("morning", "afternoon", "evening", "night")
        assert snap["devices"]["count"] == 1

    def test_signal_triangulator(self):
        from shikigami.superhuman import SignalTriangulator
        triang = SignalTriangulator()
        triang.record_signal("AA:BB:CC:DD:EE:FF", rssi=-50)
        dist = triang.get_distance("AA:BB:CC:DD:EE:FF")
        assert dist is not None
        assert dist >= 0  # Distance can be 0 if RSSI >= tx_power

    def test_rl_learner_decision(self):
        from shikigami.superhuman import ReinforcementLearner
        rl = ReinforcementLearner()
        state = rl.encode_state({"person": "alice", "mood": "happy"})
        rl.update(state, "engage", 0.8, "outcome:engaged", ["engage", "observe", "wait"])
        rec = rl.get_recommendation(state, ["engage", "observe", "wait"])
        assert rec["recommended_action"] is not None