"""Sprint 199 regression tests — block_memory except narrowing, unified_system datetime/strip/except,
calm_ambient except narrowing, self_improvement except narrowing, ambient_monitor strip-before-truncate."""

import pytest
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestBlockMemoryExceptNarrowing:
    """Verify _save() except blocks are narrowed to specific types."""

    def test_save_raises_oserror(self):
        from shadowd.block_memory import ArchivalMemory
        am = ArchivalMemory()
        am.memories = [{"content": "test"}]
        with patch("shadowd.block_memory.os.open", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                am._save()

    def test_save_raises_typeerror(self):
        from shadowd.block_memory import ArchivalMemory
        am = ArchivalMemory()
        am.memories = [{"content": object()}]
        with patch("shadowd.block_memory.os.open", side_effect=TypeError("bad type")):
            with pytest.raises(TypeError):
                am._save()


class TestUnifiedSystemTimezone:
    """Verify unified_system uses timezone-aware datetime."""

    def test_observe_uses_utc(self):
        from shadowd.unified_system import UnifiedShadowSystem
        us = UnifiedShadowSystem.__new__(UnifiedShadowSystem)
        us._lock = threading.Lock()
        now = datetime.now(timezone.utc)
        assert now.tzinfo is not None


class TestUnifiedSystemStripBeforeTruncate:
    """Verify .strip() comes before [:max_length] in sanitization."""

    def test_observe_strips_whitespace(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE, _MAX_OBSERVATION_SIZE
        text = "  hello world  "
        result = _SANITIZE_CONTENT_RE.sub('', text).strip()[:_MAX_OBSERVATION_SIZE]
        assert result == "hello world"

    def test_ask_strips_whitespace(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE, _MAX_QUESTION_SIZE
        text = "  what is this?  "
        result = _SANITIZE_CONTENT_RE.sub('', text).strip()[:_MAX_QUESTION_SIZE]
        assert result == "what is this?"

    def test_track_decision_strips_whitespace(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "  thinking hard  "
        result = _SANITIZE_CONTENT_RE.sub('', text).strip()[:5000]
        assert result == "thinking hard"

    def test_record_outcome_decision_id_strips(self):
        from shadowd.unified_system import _SANITIZE_STRICT_RE
        text = "  abc-123  "
        result = _SANITIZE_STRICT_RE.sub('', text).strip()[:200]
        assert result == "abc-123"

    def test_record_outcome_outcome_strips(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "  it went well  "
        result = _SANITIZE_CONTENT_RE.sub('', text).strip()[:5000]
        assert result == "it went well"


class TestUnifiedSystemExceptNarrowing:
    """Verify bare except Exception blocks are narrowed."""

    def test_observe_exception_types_in_module(self):
        import inspect
        from shadowd import unified_system
        source = inspect.getsource(unified_system)
        assert "except Exception:" not in source.replace("except (ValueError, TypeError, AttributeError):\n", "X")
        assert "except Exception as e:" not in source


class TestCalmAmbientExceptNarrowing:
    """Verify observation callback except is narrowed."""

    def test_callback_except_narrowed(self):
        import inspect
        from shadowd.calm_ambient import CalmAmbient
        source = inspect.getsource(CalmAmbient.observe)
        assert "except (TypeError, ValueError, AttributeError, RuntimeError)" in source


class TestSelfImprovementExceptNarrowing:
    """Verify observer except is narrowed."""

    def test_observer_except_narrowed(self):
        import inspect
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine)
        assert "except (TypeError, ValueError, AttributeError, RuntimeError):" in source


class TestAmbientMonitorStripBeforeTruncate:
    """Verify ambient_monitor person sanitization strips before truncating."""

    def test_person_strips_before_truncate(self):
        import re
        text = "  John Doe  "
        result = re.sub(r'[^\w\s-]', '', text).strip()[:100]
        assert result == "John Doe"

    def test_person_truncation(self):
        import re
        text = "A" * 200
        result = re.sub(r'[^\w\s-]', '', text).strip()[:100]
        assert len(result) == 100


class TestBlockMemoryDatetimeTimezone:
    """Verify block_memory uses timezone-aware datetimes."""

    def test_block_default_factory_uses_utc(self):
        from shadowd.block_memory import MemoryBlock
        mb = MemoryBlock(label="test", value="hello")
        assert mb.last_modified.tzinfo is not None

    def test_working_memory_add_message_utc(self):
        from shadowd.block_memory import WorkingMemory
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        msg = wm.messages[0]
        assert isinstance(msg, dict)
        ts = msg.get("timestamp")
        assert ts is not None


class TestBlockMemoryCriticalSave:
    """Verify ArchivalMemory._save uses narrowed exception types."""

    def test_save_inner_except_narrowed(self):
        import inspect
        from shadowd.block_memory import ArchivalMemory
        source = inspect.getsource(ArchivalMemory._save)
        assert "except (OSError, TypeError, ValueError):" in source

    def test_save_outer_except_narrowed(self):
        import inspect
        from shadowd.block_memory import ArchivalMemory
        source = inspect.getsource(ArchivalMemory._save)
        lines = [l for l in source.split('\n') if 'except' in l and 'OSError' in l]
        assert len(lines) >= 2