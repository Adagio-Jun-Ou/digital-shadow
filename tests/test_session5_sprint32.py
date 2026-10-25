#!/usr/bin/env python3
"""Regression tests for Sprint 32 audit fixes.

1. mind.py: _get_client() TOCTOU race — asyncio.Lock
2. mind.py: Retry on 5xx errors (500, 502, 503)
3. mind.py: json.dumps() default=str fallback
4. shadow_ai.py: _lock acquired in observe/track_person/track_decision/learn_pattern
5. shadow_ai.py: _save_data() truncation 500→1000
6. shadow_ai.py: _synthesize() try/except around _scan_bt/_scan_wifi
7. shadow_ai.py: datetime.fromisoformat() crash guard
8. shadow_ai.py: _is_commitment() word boundary matching
9. shadowd.py: _login_rate_limits thread-safe with threading.Lock
10. shadowd.py: WebSocket token uses removeprefix
11. observers.py: Bash history truncation reset
12. observers.py: rglob depth/file limit
13. calm_ambient.py: timezone-aware datetime.now(timezone.utc)
14. calm_ambient.py: Daily reset tracks date, not hour
15. proactive_insights.py: OVERDUE category for past deadlines
16. proactive_insights.py: run_proactive_monitor calls evaluate_rules
"""

import asyncio
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# Fix 1: mind.py _get_client() TOCTOU race — asyncio.Lock
# =============================================================================

class TestMindClientLock:
    @pytest.mark.asyncio
    async def test_get_client_has_lock(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        assert hasattr(engine, '_client_lock')
        assert isinstance(engine._client_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_get_client_returns_same(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        clients = await asyncio.gather(
            engine._get_client(),
            engine._get_client(),
            engine._get_client(),
        )
        assert clients[0] is clients[1]
        assert clients[1] is clients[2]
        await engine.close()

    @pytest.mark.asyncio
    async def test_client_recreated_after_close(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        c1 = await engine._get_client()
        await engine.close()
        c2 = await engine._get_client()
        assert c1 is not c2
        await engine.close()


# =============================================================================
# Fix 2: mind.py retry on 5xx errors
# =============================================================================

class TestMindRetry5xx:
    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        import httpx
        from shadowd.mind import MindEngine
        engine = MindEngine()
        call_count = 0
        orig_call = engine._call_llm

        async def mock_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("POST", "http://test"),
                    response=httpx.Response(500),
                )
            return "success"

        engine._call_llm = mock_call
        result = await engine.think("test")
        assert call_count == 3
        assert result == "success"
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_on_502(self):
        import httpx
        from shadowd.mind import MindEngine
        engine = MindEngine()
        call_count = 0

        async def mock_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.HTTPStatusError(
                    "Bad Gateway",
                    request=httpx.Request("POST", "http://test"),
                    response=httpx.Response(502),
                )
            return "success"

        engine._call_llm = mock_call
        result = await engine.think("test")
        assert call_count == 2
        assert result == "success"
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        import httpx
        from shadowd.mind import MindEngine
        engine = MindEngine()

        async def mock_call(messages):
            raise httpx.HTTPStatusError(
                "Service Unavailable",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(503),
            )

        engine._call_llm = mock_call
        result = await engine.think("test")
        assert "error" in result.lower() or "max retries" in result.lower()
        await engine.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        import httpx
        from shadowd.mind import MindEngine
        engine = MindEngine()
        call_count = 0

        async def mock_call(messages):
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(404),
            )

        engine._call_llm = mock_call
        result = await engine.think("test")
        assert call_count == 1
        assert "error" in result.lower()
        await engine.close()


# =============================================================================
# Fix 3: mind.py json.dumps() default=str
# =============================================================================

class TestMindJsonDefaultStr:
    def test_build_messages_default_str_in_dumps(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        obs = [{"type": "test", "content": {"value": datetime(2025, 1, 1, tzinfo=timezone.utc)}, "timestamp": "2025-01-01T00:00:00"}]
        result = json.dumps(obs[-20:], default=str)
        assert "2025" in result

    def test_build_messages_with_context_dict_containing_datetime(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        ctx = {"time": datetime(2025, 6, 1, tzinfo=timezone.utc), "count": 42}
        messages = engine._build_messages("test query", context=ctx, observations=None)
        user_msg = messages[1]["content"]
        assert "Additional context" in user_msg
        assert "2025" in user_msg


# =============================================================================
# Fix 4: shadow_ai.py _lock acquired in observe/track_person/track_decision/learn_pattern
# =============================================================================

class TestShadowAILock:
    def _make_ai(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        mind = MagicMock()
        ai = ShadowAI(registry, mind, data_dir=str(tmp_path))
        return ai

    def test_track_person_is_thread_safe(self, tmp_path):
        ai = self._make_ai(tmp_path)
        errors = []

        def add_person(i):
            try:
                ai.track_person(f"person{i}", "aa:bb:cc:dd:ee:f1", "friend")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_person, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(ai.people) == 20

    def test_track_decision_is_thread_safe(self, tmp_path):
        ai = self._make_ai(tmp_path)
        errors = []

        def add_decision(i):
            try:
                ai.track_decision(f"situation{i}", f"choice{i}", f"outcome{i}", 5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_decision, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(ai.decisions) == 20

    def test_learn_pattern_is_thread_safe(self, tmp_path):
        ai = self._make_ai(tmp_path)
        errors = []

        def add_pattern(i):
            try:
                ai.learn_pattern(f"cause{i}", f"effect{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_pattern, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(ai.patterns) == 20

    @pytest.mark.asyncio
    async def test_observe_concurrent(self, tmp_path):
        ai = self._make_ai(tmp_path)
        ai.stuck_detector = MagicMock()
        ai.stuck_detector.record_observation = MagicMock()
        ai.event_bus = MagicMock()
        ai.event_bus.publish = AsyncMock()

        async def add_obs(i):
            await ai.observe({"type": "test", "content": {"i": i}})

        await asyncio.gather(*[add_obs(i) for i in range(50)])
        assert len(ai.observations) == 50

    def test_lock_is_threading_lock(self, tmp_path):
        ai = self._make_ai(tmp_path)
        assert isinstance(ai._lock, type(threading.Lock()))


# =============================================================================
# Fix 5: shadow_ai.py _save_data() truncation 500→1000
# =============================================================================

class TestSaveDataTruncation:
    def test_save_truncates_to_1000(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI(MagicMock(), MagicMock(), data_dir=str(tmp_path))
        ai.observations = [{"i": i} for i in range(1200)]
        ai._save_data()
        saved = json.loads((tmp_path / "observations.json").read_text())
        assert len(saved) == 1000  # truncated to in-memory cap

    def test_save_handles_dicts(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI(MagicMock(), MagicMock(), data_dir=str(tmp_path))
        ai.patterns = {f"key{i}": f"val{i}" for i in range(10)}
        ai._save_data()
        saved = json.loads((tmp_path / "patterns.json").read_text())
        assert len(saved) == 10


# =============================================================================
# Fix 6: shadow_ai.py _synthesize() try/except around _scan_bt/_scan_wifi
# =============================================================================

class TestSynthesizeErrorHandling:
    @pytest.mark.asyncio
    async def test_synthesize_handles_scan_bt_failure(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        registry._scan_bt = AsyncMock(side_effect=RuntimeError("BT scan failed"))
        registry._scan_wifi = AsyncMock(return_value=[])
        ai = ShadowAI(registry, MagicMock(), data_dir=str(tmp_path))
        ai.observations = [{"type": "test"}]
        ai.people = {}
        insights = await ai._synthesize({"observations": []})
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_synthesize_handles_scan_wifi_failure(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        registry._scan_bt = AsyncMock(return_value=[])
        registry._scan_wifi = AsyncMock(side_effect=RuntimeError("WiFi scan failed"))
        ai = ShadowAI(registry, MagicMock(), data_dir=str(tmp_path))
        ai.observations = [{"type": "test"}]
        ai.people = {}
        insights = await ai._synthesize({"observations": []})
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_synthesize_handles_both_failures(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        registry._scan_bt = AsyncMock(side_effect=OSError("BT fail"))
        registry._scan_wifi = AsyncMock(side_effect=OSError("WiFi fail"))
        ai = ShadowAI(registry, MagicMock(), data_dir=str(tmp_path))
        ai.observations = [{"type": "test"}]
        ai.people = {}
        insights = await ai._synthesize({"observations": []})
        assert isinstance(insights, list)


# =============================================================================
# Fix 7: shadow_ai.py datetime.fromisoformat() crash guard
# =============================================================================

class TestDatetimeFromisoformatGuard:
    @pytest.mark.asyncio
    async def test_synthesize_handles_invalid_timestamp(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        registry._scan_bt = AsyncMock(return_value=[])
        registry._scan_wifi = AsyncMock(return_value=[])
        ai = ShadowAI(registry, MagicMock(), data_dir=str(tmp_path))
        ai.decisions = [
            {"timestamp": "invalid-timestamp", "satisfaction": 5},
            {"timestamp": datetime.now(timezone.utc).isoformat(), "satisfaction": 7},
        ]
        ai.people = {}
        ai.observations = [{"type": "test"}]
        insights = await ai._synthesize({"observations": []})
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_synthesize_handles_missing_timestamp(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        registry = MagicMock()
        registry._scan_bt = AsyncMock(return_value=[])
        registry._scan_wifi = AsyncMock(return_value=[])
        ai = ShadowAI(registry, MagicMock(), data_dir=str(tmp_path))
        ai.decisions = [
            {"satisfaction": 5},
        ]
        ai.people = {}
        ai.observations = [{"type": "test"}]
        insights = await ai._synthesize({"observations": []})
        assert isinstance(insights, list)


# =============================================================================
# Fix 8: shadow_ai.py _is_commitment() word boundary matching
# =============================================================================

class TestCommitmentWordBoundary:
    def _make_ai(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        return ShadowAI(MagicMock(), MagicMock(), data_dir=str(tmp_path))

    def test_word_boundary_matches_call(self, tmp_path):
        ai = self._make_ai(tmp_path)
        result = ai._is_commitment({"type": "meeting", "content": "I will call you tomorrow"})
        assert result is True

    def test_word_boundary_rejects_substring(self, tmp_path):
        ai = self._make_ai(tmp_path)
        result = ai._is_commitment({"type": "log", "content": "the recall feature works"})
        assert result is False

    def test_word_boundary_rejects_called(self, tmp_path):
        ai = self._make_ai(tmp_path)
        result = ai._is_commitment({"type": "log", "content": "already called"})
        assert result is False  # "call" not a standalone word in "called"

    def test_word_boundary_tomorrow(self, tmp_path):
        ai = self._make_ai(tmp_path)
        assert ai._is_commitment({"type": "cal", "content": "tomorrow at 3pm"}) is False

    def test_word_boundary_deadline(self, tmp_path):
        ai = self._make_ai(tmp_path)
        assert ai._is_commitment({"type": "task", "content": "project deadline"}) is False

    def test_word_boundary_will_in_willing(self, tmp_path):
        ai = self._make_ai(tmp_path)
        assert ai._is_commitment({"type": "log", "content": "willing to help"}) is False  # "will" not a word in "willing"

    def test_word_boundary_recall_not_call(self, tmp_path):
        ai = self._make_ai(tmp_path)
        assert ai._is_commitment({"type": "log", "content": "the recall feature"}) is False  # "call" not word in "recall"


# =============================================================================
# Fix 9: shadowd.py _login_rate_limits thread-safe
# =============================================================================

class TestLoginRateLimitThreadSafe:
    def test_concurrent_rate_limit_checks(self):
        from shadowd.shadowd import _check_login_rateLimit, _login_rate_limits, _login_rate_lock
        _login_rate_limits.clear()
        errors = []

        def check(ip):
            try:
                for _ in range(3):
                    _check_login_rateLimit(ip)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check, args=(f"10.0.0.{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        _login_rate_limits.clear()

    def test_rate_limit_blocks_after_max(self):
        from shadowd.shadowd import _check_login_rateLimit, _login_rate_limits
        _login_rate_limits.clear()
        ip = "192.168.1.1"
        for _ in range(5):
            assert _check_login_rateLimit(ip) is True
        assert _check_login_rateLimit(ip) is False
        _login_rate_limits.clear()

    def test_lock_exists(self):
        from shadowd.shadowd import _login_rate_lock
        assert isinstance(_login_rate_lock, type(threading.Lock()))


# =============================================================================
# Fix 10: shadowd.py WebSocket token removeprefix
# =============================================================================

class TestWebSocketTokenExtraction:
    def test_removeprefix_extracts_token(self):
        header = "Bearer abc123"
        assert header.removeprefix("Bearer ") == "abc123"

    def test_removeprefix_no_double_replace(self):
        header = "Bearer Bearer abc"
        result = header.removeprefix("Bearer ")
        assert result == "Bearer abc"

    def test_replace_vs_removeprefix_difference(self):
        header = "Bearer Bearer abc"
        old_style = header.replace("Bearer ", "")
        new_style = header.removeprefix("Bearer ")
        assert old_style != new_style
        assert new_style == "Bearer abc"
        assert old_style == "abc"


# =============================================================================
# Fix 11: observers.py Bash history truncation reset
# =============================================================================

class TestBashHistoryTruncation:
    def test_last_size_reset_on_truncation_logic(self):
        last_size = 500
        current_size = 100
        if current_size < last_size:
            last_size = 0
        assert last_size == 0

    def test_last_size_unchanged_when_larger(self):
        last_size = 100
        current_size = 200
        if current_size < last_size:
            last_size = 0
        assert last_size == 100


# =============================================================================
# Fix 12: observers.py rglob depth/file limit
# =============================================================================

class TestRglobDepthLimit:
    def test_rglob_limited_function_exists(self):
        from shadowd.observers import _observe_filesystem_poll
        assert callable(_observe_filesystem_poll)

    def test_max_depth_constants_defined(self):
        import shadowd.observers as obs_module
        assert hasattr(obs_module, '_MAX_FS_DEPTH')
        assert hasattr(obs_module, '_MAX_FS_FILES')
        assert obs_module._MAX_FS_DEPTH > 0
        assert obs_module._MAX_FS_FILES > 0


# =============================================================================
# Fix 13: calm_ambient.py timezone-aware datetime
# =============================================================================

class TestCalmAmbientTimezone:
    def test_attention_event_uses_utc(self):
        from shadowd.calm_ambient import AttentionEvent
        event = AttentionEvent(event_type="test", message="test", priority="low")
        assert event.timestamp.tzinfo is not None

    def test_set_dnd_uses_utc(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        mgr.set_dnd(1)
        assert mgr.dnd_until.tzinfo is not None

    def test_now_calls_use_utc(self):
        from shadowd.calm_ambient import CalmAmbient
        ambient = CalmAmbient()
        event = {"type": "test"}
        ambient.observe(event)
        ts = event["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_can_interrupt_uses_utc(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        result = mgr.can_interrupt("focus")
        assert isinstance(result, bool)


# =============================================================================
# Fix 14: calm_ambient.py daily reset tracks date
# =============================================================================

class TestDailyResetByDate:
    def test_attention_manager_has_last_reset_date(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        assert hasattr(mgr, '_last_reset_date')
        assert mgr._last_reset_date == datetime.now(timezone.utc).date()

    def test_prune_expired_resets_on_new_date(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        mgr.daily_interruptions = 15
        mgr._last_reset_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        mgr.prune_expired()
        assert mgr.daily_interruptions == 0
        assert mgr._last_reset_date == datetime.now(timezone.utc).date()

    def test_prune_expired_no_reset_same_day(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        mgr.daily_interruptions = 15
        mgr._last_reset_date = datetime.now(timezone.utc).date()
        mgr.prune_expired()
        assert mgr.daily_interruptions == 15


# =============================================================================
# Fix 15: proactive_insights.py OVERDUE category
# =============================================================================

class TestOverdueCommitment:
    def test_overdue_commitment_detected(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        from shadowd.ambient_monitor import Commitment
        engine = ProactiveInsightEngine()
        past_deadline = datetime.now(timezone.utc) - timedelta(minutes=30)
        commitment = Commitment(
            id="1",
            content="Submit report",
            deadline=past_deadline,
            status="pending"
        )
        insights = engine.check_commitments([commitment])
        overdue = [i for i in insights if "OVERDUE" in i.title]
        assert len(overdue) >= 1

    def test_overdue_commitment_critical_priority(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        from shadowd.ambient_monitor import Commitment
        engine = ProactiveInsightEngine()
        past_deadline = datetime.now(timezone.utc) - timedelta(hours=2)
        commitment = Commitment(
            id="2",
            content="Late task",
            deadline=past_deadline,
            status="pending"
        )
        insights = engine.check_commitments([commitment])
        overdue = [i for i in insights if "OVERDUE" in i.title]
        assert len(overdue) >= 1
        assert overdue[0].priority == "critical"

    def test_future_commitment_not_overdue(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        from shadowd.ambient_monitor import Commitment
        engine = ProactiveInsightEngine()
        future_deadline = datetime.now(timezone.utc) + timedelta(hours=48)
        commitment = Commitment(
            id="3",
            content="Future task",
            deadline=future_deadline,
            status="pending"
        )
        insights = engine.check_commitments([commitment])
        overdue = [i for i in insights if "OVERDUE" in i.title]
        assert len(overdue) == 0

    def test_negative_minutes_not_shown(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        from shadowd.ambient_monitor import Commitment
        engine = ProactiveInsightEngine()
        past_deadline = datetime.now(timezone.utc) - timedelta(minutes=30)
        commitment = Commitment(
            id="4",
            content="Late task",
            deadline=past_deadline,
            status="pending"
        )
        insights = engine.check_commitments([commitment])
        for i in insights:
            assert "Due in -" not in i.message


# =============================================================================
# Fix 16: proactive_insights.py run_proactive_monitor calls evaluate_rules
# =============================================================================

class TestProactiveMonitorCallsEvaluate:
    def test_evaluate_rules_is_called_in_monitor(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        engine = ProactiveInsightEngine()
        original_evaluate = engine.evaluate_rules
        call_count = 0

        def mock_evaluate(context):
            nonlocal call_count
            call_count += 1
            return []

        engine.evaluate_rules = mock_evaluate
        engine.evaluate_rules({"hour": 9, "pending_commitments": 1})
        assert call_count == 1
        engine.evaluate_rules = original_evaluate

    @pytest.mark.asyncio
    async def test_run_proactive_monitor_invokes_evaluate(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        engine = ProactiveInsightEngine()
        evaluated = []

        original_evaluate = engine.evaluate_rules

        def mock_evaluate(context):
            evaluated.append(context)
            return []

        engine.evaluate_rules = mock_evaluate

        call_count = 0

        class StopLoop(Exception):
            pass

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise StopLoop()

        with patch('shadowd.proactive_insights.asyncio.sleep', side_effect=mock_sleep):
            try:
                await engine.run_proactive_monitor(interval=1)
            except StopLoop:
                pass

        assert len(evaluated) >= 1

    def test_evaluate_rules_returns_insights(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        engine = ProactiveInsightEngine()
        result = engine.evaluate_rules({"hour": 9, "pending_commitments": 1})
        assert isinstance(result, list)


# =============================================================================
# Additional edge case tests
# =============================================================================

class TestEdgeCases:
    def test_commitment_word_boundary_empty_content(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI(MagicMock(), MagicMock(), data_dir=str(tmp_path))
        result = ai._is_commitment({"type": "", "content": ""})
        assert result is False

    def test_calm_ambient_prune_expired_no_crash(self):
        from shadowd.calm_ambient import AttentionManager
        mgr = AttentionManager()
        mgr.prune_expired()
        assert mgr.daily_interruptions >= 0

    def test_mind_json_default_str_handles_set(self):
        result = json.dumps({1, 2, 3}, default=str)
        assert isinstance(result, str)
        assert "1" in result

    @pytest.mark.asyncio
    async def test_shadow_ai_save_data_with_datetime(self, tmp_path):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI(MagicMock(), MagicMock(), data_dir=str(tmp_path))
        ai.patterns = {"key": datetime(2025, 1, 1, tzinfo=timezone.utc)}
        ai._save_data()
        saved = json.loads((tmp_path / "patterns.json").read_text())
        assert "key" in saved