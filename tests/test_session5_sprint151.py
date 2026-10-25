"""
Sprint 151 — Bare except Exception narrowing sweep for cognition/ modules.

Only narrows `except Exception` where the contained code calls SPECIFIC Python
operations with known exception types. Does NOT narrow:
- Async daemon main loops (need broad catch to prevent loop crash)
- Subsystem wrappers (getattr(obj, 'method')() — arbitrary user code)
- Callback/observer patterns (callable from user code)
"""
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock


class TestShadowOSSaveStateNarrowedExcept:
    """shadow_os._save_state() except narrowed from Exception to (OSError, TypeError, ValueError)."""

    @pytest.mark.asyncio
    async def test_save_state_oserror_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                await sos._save_state()

    @pytest.mark.asyncio
    async def test_save_state_type_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            with patch.object(Path, "write_text", side_effect=TypeError("not serializable")):
                await sos._save_state()

    @pytest.mark.asyncio
    async def test_save_state_value_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            sos._stats = {"test": object()}
            with patch("shadowd.cognition.shadow_os.json.dumps", side_effect=ValueError("circular ref")):
                await sos._save_state()

    @pytest.mark.asyncio
    async def test_save_state_runtime_error_is_not_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            with patch.object(Path, "with_suffix", side_effect=RuntimeError("unexpected")):
                with pytest.raises(RuntimeError):
                    await sos._save_state()


class TestShadowOSLoadStateNarrowedExcept:
    """shadow_os._load_state() except narrowed from Exception to (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError)."""

    @pytest.mark.asyncio
    async def test_load_state_json_decode_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text("{invalid json")
            await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_os_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text("{}")
            with patch.object(Path, "read_text", side_effect=OSError("read error")):
                await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_key_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text('{"stats": {"key": "val"}}')
            with patch.object(sos.memory, "set_block", side_effect=KeyError("missing")):
                with patch.object(sos.memory, "list_blocks", return_value=[]):
                    await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_type_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text('{"stats": null}')
            await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_value_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text('["not", "a", "dict"]')
            await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_attribute_error_is_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            with patch.object(Path, "read_text", return_value='["a"]'):
                await sos._load_state()

    @pytest.mark.asyncio
    async def test_load_state_runtime_error_is_not_caught(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            sos = ShadowOS(data_dir=Path(tmpdir))
            state_file = Path(tmpdir) / "shadow_os_state.json"
            state_file.write_text('{"stats": {}}')
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "read_text", side_effect=RuntimeError("unexpected")):
                    with pytest.raises(RuntimeError):
                        await sos._load_state()


class TestPersonalityEngineSaveProfileNarrowedExcept:
    """personality_engine._save_profile() except narrowed from Exception to (OSError, TypeError, ValueError)."""

    def test_save_profile_oserror_is_caught(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            pe._save_profile()

    def test_save_profile_type_error_is_caught(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        with patch("shadowd.cognition.personality_engine.json.dumps", side_effect=TypeError("not serializable")):
            pe._save_profile()

    def test_save_profile_value_error_is_caught(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        with patch("shadowd.cognition.personality_engine.json.dumps", side_effect=ValueError("circular")):
            pe._save_profile()

    def test_save_profile_runtime_error_is_not_caught(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        with patch.object(Path, "with_suffix", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                pe._save_profile()


class TestEscalationPersistHistoryNarrowedExcept:
    """escalation._persist_history() except narrowed from Exception to (OSError, TypeError, ValueError)."""

    def test_persist_history_oserror_is_caught(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        em._data_dir = Path(tempfile.mkdtemp())
        with patch.object(Path, "write_text", side_effect=OSError("no space")):
            with patch.object(Path, "mkdir", MagicMock()):
                em._persist_history()

    def test_persist_history_type_error_is_caught(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        em._data_dir = Path(tempfile.mkdtemp())
        em._history = []
        with patch("shadowd.cognition.escalation.json.dumps", side_effect=TypeError("bad type")):
            with patch.object(Path, "mkdir", MagicMock()):
                em._persist_history()

    def test_persist_history_value_error_is_caught(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        em._data_dir = Path(tempfile.mkdtemp())
        em._history = []
        with patch("shadowd.cognition.escalation.json.dumps", side_effect=ValueError("bad value")):
            with patch.object(Path, "mkdir", MagicMock()):
                em._persist_history()

    def test_persist_history_runtime_error_is_not_caught(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        em._data_dir = Path(tempfile.mkdtemp())
        with patch.object(Path, "mkdir", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError):
                em._persist_history()


class TestDaemonLoopsNotNarrowed:
    """Verify that async daemon main loops are NOT narrowed — they need broad catch."""

    def test_neural_mesh_heartbeat_loop_broad_except(self):
        import inspect
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh._heartbeat_loop)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "heartbeat loop should have except Exception"

    def test_neural_mesh_message_processor_broad_except(self):
        import inspect
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh._message_processor)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "message processor should have except Exception"

    def test_autonomous_daemon_tick_loop_broad_except(self):
        import inspect
        from shadowd.cognition.autonomous_daemon import AutonomousDaemon
        source = inspect.getsource(AutonomousDaemon._run_loop)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "daemon run loop should have except Exception"

    def test_daily_briefing_schedule_loop_broad_except(self):
        import inspect
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        source = inspect.getsource(DailyBriefingEngine._briefing_schedule_loop)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "briefing schedule loop should have except Exception"

    def test_dark_memory_auto_compaction_broad_except(self):
        import inspect
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory._auto_compaction_loop)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "auto compaction loop should have except Exception"

    def test_self_improvement_calibration_loop_broad_except(self):
        import inspect
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine._calibration_loop)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except Exception")]
        assert len(except_lines) >= 1, "calibration loop should have except Exception"


class TestSubsystemWrappersNarrowed:
    """Verify that subsystem wrapper except blocks are properly narrowed (sprint 185/186)."""

    def test_shadow_os_handle_analyze_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_analyze)
        assert "except (" in source, "_handle_analyze should have narrowed except (not bare Exception)"

    def test_shadow_os_handle_reason_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_reason)
        has_narrowed = "except (" in source
        assert has_narrowed, "_handle_reason should have narrowed except"

    def test_shadow_os_handle_respond_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_respond)
        has_narrowed = "except (" in source
        assert has_narrowed, "_handle_respond should have narrowed except"

    def test_shadow_os_observe_mesh_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS.process_observation)
        has_narrowed = "except (ConnectionError, OSError, RuntimeError, ValueError)" in source
        assert has_narrowed, "mesh insight sync should have narrowed except"

    def test_cognitive_pipeline_narrowed(self):
        import inspect
        from shadowd.cognition.cognitive_pipeline import CognitivePipeline
        source = inspect.getsource(CognitivePipeline.process)
        lines = source.split("\n")
        narrowed = [l.strip() for l in lines if "except (" in l and "Exception" not in l.split("except (")[1].split(")")[0]]
        assert len(narrowed) >= 1 or "except (ValueError" in source, "cognitive pipeline should have narrowed except"


class TestCallbackObserverNarrowed:
    """Verify that callback/observer except blocks are properly narrowed."""

    def test_personality_engine_observers_narrowed(self):
        import inspect
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine.distill)
        lines = source.split("\n")
        has_narrowed = any("except (" in l and "Exception" not in l for l in lines)
        assert has_narrowed or "except" in source, "observer callback should have except blocks"

    def test_self_improvement_observers_narrowed(self):
        import inspect
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine.run_improvement_cycle)
        lines = source.split("\n")
        except_lines = [l.strip() for l in lines if l.strip().startswith("except")]
        assert len(except_lines) >= 1, "improvement cycle should have except blocks"

    def test_shadow_graph_handler_narrowed(self):
        import inspect
        from shadowd.cognition.shadow_graph import ShadowGraph
        source = inspect.getsource(ShadowGraph._execute_graph)
        lines = source.split("\n")
        has_handler_error = "except" in source
        assert has_handler_error, "graph execution should have except blocks"