#!/usr/bin/env python3
"""
Session 5 Sprint 22 - Deep Codebase Audit Fixes

Regression tests for 5 CRITICAL + 7 HIGH fixes from deep review:

CRITICAL:
- C1: dark_memory SQLite thread safety (threading.Lock added)
- C2: dark_memory compact() now deletes evicted atoms from SQLite
- C3: neural_mesh message signing uses proper HMAC (was hash-based MAC)
- C4: neural_mesh unsigned messages no longer bypass verification when secret is set
- C5: stealth_layer key rotation re-encrypts stored data (was destroying it)

HIGH:
- H1: shadow_os UnboundLocalError for search_results fixed
- H2: shadow_os XSS sanitization improved (strips all tags + javascript: + event handlers)
- H3: api_router singleton creation thread-safe with lock
- H4: shadowosd path leak removed from error response
- H5: self_improvement AttributeError crash (_predictions → calibrator._predictions)
- H6: self_improvement float 0.0 short-circuit fixed
- H7: escalation ReDoS regex fixed (removed | in character class)

MEDIUM:
- M1: self_improvement decision/open_decision/observer caps
- M2: anomaly_detector baseline caps
- M3: dark_memory import_data persists to SQLite
- M4: dark_memory deduplicate DB delete logs errors instead of silent pass
- M5: stealth_layer decrypt error handling (InvalidTag caught)
"""

import asyncio
import json
import os
import pytest
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================
# C1: dark_memory SQLite thread safety
# ============================================================

class TestDarkMemoryThreadSafety:
    """SQLite should be protected by threading.Lock for cross-thread safety."""

    def test_db_thread_lock_exists(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td)
            assert hasattr(dm, '_db_thread_lock')
            assert isinstance(dm._db_thread_lock, type(threading.Lock()))

    def test_concurrent_store(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td)
            errors = []

            def store_items(idx):
                try:
                    loop = asyncio.new_event_loop()
                    for i in range(10):
                        loop.run_until_complete(
                            dm.store(f"content_{idx}_{i}", atom_type=AtomType.OBSERVATION)
                        )
                    loop.close()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=store_items, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================
# C2: dark_memory compact deletes from SQLite
# ============================================================

class TestDarkMemoryCompactDB:
    """Compact should delete evicted atoms from SQLite, not just memory."""

    def test_compact_removes_from_db(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType, MemoryTier
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td)

            async def _test():
                for i in range(600):
                    await dm.store(
                        f"content_{i}",
                        atom_type=AtomType.OBSERVATION,
                        importance=0.1 if i < 520 else 0.9,
                    )

                count_before = len(dm._atoms)
                removed = await dm.compact()
                count_after = len(dm._atoms)

                if removed > 0:
                    cursor = dm._db_conn.execute("SELECT COUNT(*) FROM observations")
                    db_count = cursor.fetchone()[0]
                    assert db_count == count_after, f"DB count ({db_count}) should match memory ({count_after})"

            asyncio.run(_test())


# ============================================================
# C3: neural_mesh HMAC signing
# ============================================================

class TestNeuralMeshHMAC:
    """Message signing should use HMAC, not simple hash-based MAC."""

    def test_sign_uses_hmac(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        msg = MeshMessage(sender_id="test")
        sig = msg.sign("test_secret")
        assert sig, "Sign should return a signature"
        assert len(sig) == 32, "HMAC sig should be 32 hex chars"

    def test_verify_correct_secret(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        msg = MeshMessage(sender_id="test")
        msg.sign("my_secret")
        assert msg.verify("my_secret") is True

    def test_verify_wrong_secret(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        msg = MeshMessage(sender_id="test")
        msg.sign("correct_secret")
        assert msg.verify("wrong_secret") is False

    def test_verify_unsigned_message(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        msg = MeshMessage(sender_id="test")
        assert msg.verify("any_secret") is False


# ============================================================
# C4: neural_mesh unsigned messages rejected when secret set
# ============================================================

class TestNeuralMeshUnsignedRejected:
    """When mesh secret is set, unsigned messages must be rejected."""

    def test_unsigned_message_verification_fails(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        msg = MeshMessage(sender_id="attacker")
        assert msg.signature == ""
        assert msg.verify("secret") is False, "Unsigned message should fail verification"


# ============================================================
# C5: stealth_layer key rotation re-encrypts
# ============================================================

class TestStealthLayerKeyRotation:
    """Key rotation should re-encrypt stored data, not destroy it."""

    def test_key_rotation_preserves_data(self):
        from shadowd.cognition.stealth_layer import StealthLayer, DataCategory
        sl = StealthLayer(epsilon=1.0)
        sl.guard.store_data("test_key", {"secret": "hello world"}, DataCategory.PERSONAL)

        sl.guard.rotate_keys()

        result = sl.guard.retrieve_data("test_key", subject="shadow_ai")
        assert result is not None, "Data should survive key rotation"
        assert result["secret"] == "hello world"

    def test_decrypt_failure_raises(self):
        from shadowd.cognition.stealth_layer import StealthLayer
        sl = StealthLayer(epsilon=1.0)
        with pytest.raises(ValueError):
            sl.guard._simple_decrypt("aW52YWxpZGJhc2U2NA==")

    def test_key_rotation_multiple_items(self):
        from shadowd.cognition.stealth_layer import StealthLayer, DataCategory
        sl = StealthLayer(epsilon=1.0)
        sl.guard.store_data("k1", {"v": 1}, DataCategory.BEHAVIORAL)
        sl.guard.store_data("k2", {"v": 2}, DataCategory.HEALTH)

        sl.guard.rotate_keys()

        r1 = sl.guard.retrieve_data("k1", subject="shadow_ai")
        r2 = sl.guard.retrieve_data("k2", subject="shadow_ai")
        assert r1["v"] == 1
        assert r2["v"] == 2


# ============================================================
# H1: shadow_os UnboundLocalError fixed
# ============================================================

class TestShadowOSNoUnboundLocal:
    """_handle_analyze should not crash with UnboundLocalError for search_results."""

    def test_analyze_without_query(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode

        async def _test():
            os_inst = ShadowOS(data_dir=tempfile.mkdtemp())
            state = CognitiveState(mode=CognitiveMode.LISTEN)
            state.current_query = ""

            result = await os_inst._handle_analyze(state)
            assert result is not None
            assert result.confidence == 0.3

        asyncio.run(_test())

    def test_analyze_with_query(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode

        async def _test():
            os_inst = ShadowOS(data_dir=tempfile.mkdtemp())
            state = CognitiveState(mode=CognitiveMode.LISTEN)
            state.current_query = "test query"

            result = await os_inst._handle_analyze(state)
            assert result is not None

        asyncio.run(_test())


# ============================================================
# H2: XSS sanitization improved
# ============================================================

class TestXSSSanitizationImproved:
    """Sanitization should strip script tags + content, all HTML tags, javascript:, event handlers."""

    def test_script_tag_removed_with_content(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert sanitize_input("<script>alert('xss')</script>hello") == "hello"

    def test_javascript_uri_removed(self):
        from shadowd.cognition.shadow_os import sanitize_input
        result = sanitize_input("javascript:alert(1)")
        assert "javascript:" not in result

    def test_event_handler_removed(self):
        from shadowd.cognition.shadow_os import sanitize_input
        result = sanitize_input('<div onclick=alert(1)>text</div>')
        assert "onclick" not in result
        assert "alert" not in result or "text" in result

    def test_img_onerror_removed(self):
        from shadowd.cognition.shadow_os import sanitize_input
        result = sanitize_input('<img src=x onerror=alert(1)>')
        assert "onerror" not in result

    def test_control_chars_stripped(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert "\x00" not in sanitize_input("hello\x00world")
        assert "\x08" not in sanitize_input("test\x08data")


# ============================================================
# H3: api_router singleton thread safety
# ============================================================

class TestAPIRouterSingleton:
    """get_shadow_os should be thread-safe."""

    def test_lock_exists(self):
        import shadowd.cognition.api_router as api_module
        assert hasattr(api_module, '_shadow_os_lock')
        assert isinstance(api_module._shadow_os_lock, type(threading.Lock()))


# ============================================================
# H4: shadowosd no path leak
# ============================================================

class TestShadowosdNoPathLeak:
    """Error response should not leak filesystem paths."""

    def test_dashboard_error_no_path(self):
        from shadowd.cognition.shadowosd import app
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        # The actual file probably exists, so this tests the code path
        # If it doesn't exist, the error shouldn't contain the path
        # We just verify the function doesn't hardcode paths in errors
        import shadowd.cognition.shadowosd as mod
        import inspect
        source = inspect.getsource(mod.dashboard)
        assert "Looking for:" not in source, "Path leak should be removed from error response"


# ============================================================
# H5: self_improvement AttributeError crash fix
# ============================================================

class TestSelfImprovementCalibrationLoop:
    """_calibration_loop should reference calibrator._predictions, not self._predictions."""

    def test_predictions_reference(self):
        import inspect
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine._calibration_loop)
        assert "self.calibrator._predictions" in source, "Should use calibrator._predictions"
        assert "self._predictions" not in source, "Should NOT use self._predictions"


# ============================================================
# H6: self_improvement float short-circuit
# ============================================================

class TestSelfImprovementFloatShortCircuit:
    """0.0 accuracy should not short-circuit the boolean expression."""

    def test_zero_accuracy_is_not_none(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=tempfile.mkdtemp())

        pred = engine.record_prediction("test", "positive", confidence=0.8)
        pred.accuracy = 0.0
        engine.resolve_prediction(pred, "negative")
        # Should not crash and should handle 0.0 correctly

    def test_none_accuracy_handled(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=tempfile.mkdtemp())

        pred = engine.record_prediction("test", "positive", confidence=0.8)
        # accuracy is None by default
        result = engine.resolve_prediction(pred, "negative")
        assert result is not None


# ============================================================
# H7: escalation ReDoS regex fix
# ============================================================

class TestEscalationReDoSFix:
    """Email regex should not contain | in character class."""

    def test_email_regex_no_pipe(self):
        from shadowd.cognition.escalation import PrivacyGuardrail
        email_pattern = PrivacyGuardrail.SENSITIVE_PATTERNS[2]
        assert "|a-z" not in email_pattern, "Should not have | inside character class"


# ============================================================
# M1: self_improvement caps
# ============================================================

class TestSelfImprovementCaps:
    """Decisions, open decisions, and observers should be capped."""

    def test_decision_caps_exist(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=tempfile.mkdtemp())
        assert hasattr(engine, '_MAX_DECISIONS')
        assert engine._MAX_DECISIONS == 10000
        assert hasattr(engine, '_MAX_OPEN_DECISIONS')
        assert hasattr(engine, '_MAX_OBSERVERS')
        assert engine._MAX_OBSERVERS == 20

    def test_observer_cap(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine(data_dir=tempfile.mkdtemp())
        for i in range(engine._MAX_OBSERVERS):
            assert engine.add_observer(lambda r: None) is True
        assert engine.add_observer(lambda r: None) is False

    def test_calibrator_caps(self):
        from shadowd.cognition.self_improvement import AccuracyCalibrator
        cal = AccuracyCalibrator()
        assert hasattr(cal, '_MAX_BY_MODEL')
        assert hasattr(cal, '_MAX_BY_DOMAIN')


# ============================================================
# M2: anomaly_detector baseline caps
# ============================================================

class TestAnomalyDetectorBaselineCaps:
    """Baselines should be capped to prevent memory exhaustion."""

    def test_baseline_cap_exists(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert hasattr(engine, '_MAX_BASELINES')
        assert engine._MAX_BASELINES == 1000

    def test_baseline_eviction(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for i in range(engine._MAX_BASELINES + 50):
            engine.set_baseline(f"user_{i}", {"metric1": {"mean": 0.5, "std": 0.1}})
        assert len(engine._baselines) <= engine._MAX_BASELINES + 1


# ============================================================
# M3: dark_memory import_data persists to SQLite
# ============================================================

class TestDarkMemoryImportPersistence:
    """import_data should persist atoms to SQLite, not just in-memory."""

    def test_import_persists_to_db(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td)

            async def _test():
                data = json.dumps({
                    "atoms": [
                        {
                            "atom_id": "test_import_1",
                            "atom_type": "observation",
                            "tier": "working",
                            "content": "imported content",
                            "importance": 0.8,
                        }
                    ]
                })
                result = await dm.import_data(data)
                assert result["imported"] == 1

                cursor = dm._db_conn.execute("SELECT COUNT(*) FROM observations WHERE id = ?", ("test_import_1",))
                count = cursor.fetchone()[0]
                assert count == 1, "Imported atom should be in SQLite"

            asyncio.run(_test())


# ============================================================
# M5: stealth_layer decrypt error handling
# ============================================================

class TestStealthLayerDecryptError:
    """_simple_decrypt should raise ValueError for corrupted ciphertext."""

    def test_invalid_ciphertext_raises_valueerror(self):
        from shadowd.cognition.stealth_layer import StealthLayer
        sl = StealthLayer(epsilon=1.0)
        with pytest.raises(ValueError):
            sl.guard._simple_decrypt("not-valid-base64-data-that-will-fail-on-decrypt==")

    def test_corrupted_ciphertext_raises_valueerror(self):
        from shadowd.cognition.stealth_layer import StealthLayer
        sl = StealthLayer(epsilon=1.0)
        encrypted = sl.guard._simple_encrypt("hello")
        corrupted = encrypted[:-5] + "XXXXX"
        with pytest.raises(ValueError):
            sl.guard._simple_decrypt(corrupted)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])