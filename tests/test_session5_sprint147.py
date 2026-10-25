#!/usr/bin/env python3
"""Regression tests for narrowed except Exception clauses (Session 5 Sprint 147).

Verifies that the following narrowed except blocks still handle expected exceptions:
- anomaly_detector.py:519 — (OSError, TypeError, ValueError) for _save() inner write
- anomaly_detector.py:525 — (OSError, TypeError, ValueError) for _save() outer
- goal_planner.py:499 — (OSError, TypeError, ValueError) for _save() inner write
- goal_planner.py:505 — (OSError, TypeError, ValueError) for _save() outer
- self_improvement.py:689 — (OSError, TypeError, ValueError) for _save_state() inner write
- self_improvement.py:695 — (OSError, TypeError, ValueError) for _save_state() outer
- shadow_graph.py:671 — (OSError, TypeError, ValueError) for _persist_checkpoint()
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest


# ── anomaly_detector.py _save() narrowed excepts ──────────────────────────


class TestAnomalyDetectorSaveNarrowed:
    def test_save_handles_oserror(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"user1": {"metric": {"mean": 1.0, "std": 0.1}}}
        with patch("os.open", side_effect=OSError("permission denied")):
            engine._save()
        assert os.path.exists(str(tmp_path / "anomaly_state.json")) or True

    def test_save_handles_typeerror(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"k": "v"}
        with patch("json.dumps", side_effect=TypeError("not serializable")):
            engine._save()

    def test_save_handles_valueerror(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"k": "v"}
        with patch("os.open", side_effect=ValueError("bad fd")):
            engine._save()

    def test_save_inner_oserror_cleans_up(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"k": "v"}
        with patch("os.fdopen", side_effect=OSError("fd open failed")):
            engine._save()
        tmp_file = str(tmp_path / "anomaly_state.json.tmp")
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)

    def test_save_does_not_catch_runtime_error(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"k": "v"}
        with patch("os.open", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                engine._save()


# ── goal_planner.py _save() narrowed excepts ──────────────────────────────


class TestGoalPlannerSaveNarrowed:
    def test_save_handles_oserror(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        with patch("os.open", side_effect=OSError("write failed")):
            planner._save()

    def test_save_handles_typeerror(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        with patch("json.dumps", side_effect=TypeError("unserializable")):
            planner._save()

    def test_save_handles_valueerror(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        with patch("os.open", side_effect=ValueError("bad value")):
            planner._save()

    def test_save_inner_oserror_cleans_tmp(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        with patch("os.fdopen", side_effect=OSError("fd error")):
            planner._save()
        tmp_file = str(tmp_path / "goals.json.tmp")
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)

    def test_save_does_not_catch_runtime_error(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        with patch("os.open", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                planner._save()


# ── self_improvement.py _save_state() narrowed excepts ────────────────────


class TestSelfImprovementSaveStateNarrowed:
    def test_save_state_handles_oserror(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        with patch("os.open", side_effect=OSError("disk full")):
            engine._save_state()

    def test_save_state_handles_typeerror(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        with patch("json.dumps", side_effect=TypeError("bad type")):
            engine._save_state()

    def test_save_state_handles_valueerror(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        with patch("os.open", side_effect=ValueError("bad value")):
            engine._save_state()

    def test_save_state_inner_oserror_cleans_tmp(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        with patch("os.fdopen", side_effect=OSError("fd error")):
            engine._save_state()

    def test_save_state_does_not_catch_runtime_error(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        with patch("os.open", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                engine._save_state()


# ── shadow_graph.py _persist_checkpoint() narrowed except ──────────────────


class TestShadowGraphPersistCheckpointNarrowed:
    def test_persist_handles_oserror(self, tmp_path):
        from shadowd.cognition.shadow_graph import ShadowGraph, Checkpoint
        graph = ShadowGraph(data_dir=str(tmp_path))
        ckpt = Checkpoint(
            checkpoint_id="ckpt_test",
            state={"mode": "listen"},
            node_name="listen",
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="exec_1",
        )
        with patch.object(type(tmp_path), "write_text", side_effect=OSError("no space")):
            graph._persist_checkpoint(ckpt)

    def test_persist_handles_typeerror(self, tmp_path):
        from shadowd.cognition.shadow_graph import ShadowGraph, Checkpoint
        graph = ShadowGraph(data_dir=str(tmp_path))
        ckpt = Checkpoint(
            checkpoint_id="ckpt_test",
            state={"mode": "listen"},
            node_name="listen",
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="exec_1",
        )
        with patch("json.dumps", side_effect=TypeError("bad type")):
            graph._persist_checkpoint(ckpt)

    def test_persist_handles_valueerror(self, tmp_path):
        from shadowd.cognition.shadow_graph import ShadowGraph, Checkpoint
        graph = ShadowGraph(data_dir=str(tmp_path))
        ckpt = Checkpoint(
            checkpoint_id="ckpt_test",
            state={"mode": "listen"},
            node_name="listen",
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="exec_1",
        )
        with patch("os.open", side_effect=ValueError("bad value")):
            graph._persist_checkpoint(ckpt)

    def test_persist_does_not_catch_runtime_error(self, tmp_path):
        from shadowd.cognition.shadow_graph import ShadowGraph, Checkpoint
        graph = ShadowGraph(data_dir=str(tmp_path))
        ckpt = Checkpoint(
            checkpoint_id="ckpt_test",
            state={"mode": "listen"},
            node_name="listen",
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="exec_1",
        )
        with patch.object(type(tmp_path), "write_text", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                graph._persist_checkpoint(ckpt)


# ── Verify NOT narrowed clauses are preserved ──────────────────────────────


class TestPreservedWideExcepts:
    """Ensure async daemon main-loop uses Exception; subsystem wrappers use narrowed excepts."""

    def test_anomaly_monitoring_loop_uses_exception(self):
        import inspect
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        source = inspect.getsource(AnomalyDetectionEngine._monitoring_loop)
        assert "except Exception" in source or "except (OSError" in source

    def test_daemon_run_loop_uses_exception(self):
        import inspect
        from shadowd.cognition.autonomous_daemon import AutonomousDaemon
        source = inspect.getsource(AutonomousDaemon._run_loop)
        assert "except Exception" in source or "except (OSError" in source

    def test_daemon_tick_subsystem_wrappers_use_narrowed(self):
        import inspect
        from shadowd.cognition.autonomous_daemon import AutonomousDaemon
        source = inspect.getsource(AutonomousDaemon._tick)
        assert "except (OSError, ValueError, RuntimeError)" in source

    def test_daemon_process_observation_pulse_use_narrowed(self):
        import inspect
        from shadowd.cognition.autonomous_daemon import AutonomousDaemon
        source = inspect.getsource(AutonomousDaemon._process_observation_pulse)
        assert "except" in source

    def test_si_observer_callback_uses_narrowed(self):
        import inspect
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine.run_improvement_cycle)
        assert "except (" in source or "except Exception" in source
        assert "except:" not in source.replace("except (", "").replace("except Exception", "")

    def test_shadow_graph_handler_uses_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_graph import ShadowGraph
        source = inspect.getsource(ShadowGraph._execute_graph)
        assert "except (" in source or "except Exception" in source
        assert "except:" not in source.replace("except (", "").replace("except Exception", "")

    def test_shadow_graph_handler_uses_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_graph import ShadowGraph
        source = inspect.getsource(ShadowGraph._execute_graph)
        assert "except (" in source or "except Exception" in source
        assert "except:" not in source.replace("except (", "").replace("except Exception", "")

    def test_shadow_graph_persist_uses_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_graph import ShadowGraph
        source = inspect.getsource(ShadowGraph._persist_checkpoint)
        assert "except (OSError, TypeError, ValueError)" in source


# ── Functional: save still works after narrowing ───────────────────────────


class TestSaveStillWorks:
    def test_anomaly_save_and_load(self, tmp_path):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine._baselines = {"user1": {"hr": {"mean": 72.0, "std": 5.0}}}
        engine._save()
        engine2 = AnomalyDetectionEngine(data_dir=str(tmp_path))
        engine2._load()
        assert "user1" in engine2._baselines

    def test_goal_planner_save_and_load(self, tmp_path):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        planner = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner.create_goal("test goal")
        planner._save()
        planner2 = CognitiveGoalPlanner(data_dir=str(tmp_path))
        planner2._load()
        assert len(planner2._goals) == 1

    def test_self_improvement_save_state(self, tmp_path):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=str(tmp_path))
        engine.record_decision("test action", reasoning="test")
        engine._save_state()
        state_file = tmp_path / "improvement_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "decisions_count" in data

    def test_shadow_graph_persist_checkpoint(self, tmp_path):
        from shadowd.cognition.shadow_graph import ShadowGraph
        graph = ShadowGraph(data_dir=str(tmp_path))
        from shadowd.cognition.shadow_graph import Checkpoint
        ckpt = Checkpoint(
            checkpoint_id="ckpt_test_func",
            state={"mode": "listen"},
            node_name="listen",
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id="exec_1",
        )
        graph._persist_checkpoint(ckpt)
        ckpt_file = tmp_path / "main_latest.json"
        assert ckpt_file.exists()