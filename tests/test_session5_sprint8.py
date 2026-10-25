#!/usr/bin/env python3
"""
Session 5 Sprint 8 Regression Tests

Covers:
- V-259: Cap decisions/patterns in decision_learning_v2
- V-265/V-266/V-267: Cap insights/rule_cooldowns, fix str(e) in proactive_insights
- V-268/V-269/V-270: Cap critical/detected_patterns in calm_ambient
- V-264: Cap git_activity dict in observers
- V-250: Auth on /act endpoint in shadowd
- V-252: WebSocket auth in shadowd
- V-251: Rate limiting on /auth/login and /auth/refresh
- V-253: Auth on /shikigami/summon
- V-254: Broadcast message size validation
- V-256/V-257/V-258: Auth on data-exposing GET endpoints
- V-260: Confidence/satisfaction range validation
- V-263: ObserveRequest content size validation
"""

import pytest
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDecisionLearningCaps:
    """V-259: Cap decisions/patterns in decision_learning_v2."""

    def test_has_max_decisions_constant(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        assert hasattr(DecisionLearningV2, '_MAX_DECISIONS')
        assert DecisionLearningV2._MAX_DECISIONS > 0

    def test_has_max_patterns_constant(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        assert hasattr(DecisionLearningV2, '_MAX_PATTERNS')
        assert DecisionLearningV2._MAX_PATTERNS > 0

    def test_has_max_history_constant(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        assert hasattr(DecisionLearningV2, '_MAX_HISTORY')
        assert DecisionLearningV2._MAX_HISTORY > 0

    def test_decisions_capped(self, tmp_path):
        from shadowd.decision_learning_v2 import DecisionLearningV2, DecisionV2
        dl = DecisionLearningV2(data_dir=tmp_path)
        dl._MAX_DECISIONS = 10
        for i in range(20):
            dl.decisions[f"dec_{i}"] = DecisionV2(
                id=f"dec_{i}", context="", situation=f"sit_{i}",
                options=["a"], chosen="a", reasoning="r"
            )
            dl.decision_history.append(f"dec_{i}")
            dl.categories["general"].append(f"dec_{i}")
        result, _ = dl.create_decision("test", ["opt1"])
        assert len(dl.decisions) <= 11

    def test_patterns_capped(self, tmp_path):
        from shadowd.decision_learning_v2 import DecisionLearningV2, DecisionV2
        dl = DecisionLearningV2(data_dir=tmp_path)
        dl._MAX_PATTERNS = 5
        for i in range(10):
            decision = DecisionV2(
                id=f"dec_{i}", context="", situation=f"sit_{i}",
                options=["a", "b"], chosen="a", reasoning="test",
                tags=[f"tag_{i}"], satisfaction=8,
            )
            decision.status = "resolved"
            decision.outcome = "good"
            dl.decisions[f"dec_{i}"] = decision
            dl._detect_pattern(decision)
        assert len(dl.patterns) <= 5


class TestProactiveInsightCaps:
    """V-265/V-266/V-267: Cap insights/rule_cooldowns, fix str(e) in proactive_insights."""

    def test_has_max_insights_constant(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        assert hasattr(ProactiveInsightEngine, '_MAX_INSIGHTS')
        assert ProactiveInsightEngine._MAX_INSIGHTS > 0

    def test_has_max_rule_cooldowns_constant(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        assert hasattr(ProactiveInsightEngine, '_MAX_RULE_COOLDOWNS')
        assert ProactiveInsightEngine._MAX_RULE_COOLDOWNS > 0

    def test_insights_queue_capped(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        from shadowd.ambient_monitor import Insight
        from datetime import datetime, timedelta
        engine = ProactiveInsightEngine()
        engine._MAX_INSIGHTS = 10
        for i in range(20):
            engine.insights_queue.append(Insight(
                insight_type="test",
                priority="low",
                title=f"Test {i}",
                message=f"Test message {i}",
                confidence=0.5,
                expires_at=datetime.now() + timedelta(hours=1)
            ))
        engine.evaluate_rules({"hour": 9, "pending_commitments": 5})
        assert len(engine.insights_queue) <= 10

    def test_str_e_replaced_with_type(self):
        from shadowd import proactive_insights as pi
        source = inspect.getsource(pi)
        assert 'type(e).__name__' in source, "Should use type(e).__name__ not str(e)"


class TestCalmAmbientCaps:
    """V-268/V-269/V-270: Cap critical/detected_patterns in calm_ambient."""

    def test_attention_manager_has_max_critical(self):
        from shadowd.calm_ambient import AttentionManager
        assert hasattr(AttentionManager, '_MAX_CRITICAL')
        assert AttentionManager._MAX_CRITICAL > 0

    def test_attention_manager_has_max_focus(self):
        from shadowd.calm_ambient import AttentionManager
        assert hasattr(AttentionManager, '_MAX_FOCUS')
        assert AttentionManager._MAX_FOCUS > 0

    def test_calm_ambient_has_max_patterns(self):
        from shadowd.calm_ambient import CalmAmbient
        assert hasattr(CalmAmbient, '_MAX_DETECTED_PATTERNS')
        assert CalmAmbient._MAX_DETECTED_PATTERNS > 0

    def test_critical_list_capped(self):
        from shadowd.calm_ambient import AttentionManager, AttentionEvent
        mgr = AttentionManager()
        mgr._MAX_CRITICAL = 5
        mgr.dnd_mode = False
        mgr.daily_interruptions = 0
        mgr.max_daily_interruptions = 100
        for i in range(20):
            mgr.add_event(AttentionEvent(
                event_type="test",
                message=f"Critical {i}",
                priority="critical",
            ))
        assert len(mgr.critical) <= 5

    def test_detected_patterns_capped(self):
        from shadowd.calm_ambient import CalmAmbient
        ambient = CalmAmbient()
        ambient._MAX_DETECTED_PATTERNS = 10
        for i in range(50):
            ambient._detect_pattern({"type": f"type_{i}"})
        assert len(ambient.detected_patterns) <= 10

    def test_str_e_replaced_with_type(self):
        from shadowd import calm_ambient as ca
        source = inspect.getsource(ca)
        assert 'type(e).__name__' in source


class TestObserversCaps:
    """V-264: Cap git_activity dict in observers."""

    def test_git_max_repos_constant(self):
        from shadowd import observers
        source = inspect.getsource(observers.observe_git_activity)
        assert '_MAX_GIT_REPOS' in source

    def test_git_last_check_cap_logic(self):
        from shadowd import observers
        source = inspect.getsource(observers.observe_git_activity)
        assert '_MAX_GIT_REPOS' in source


class TestShadowdActAuth:
    """V-250: Auth on /act endpoint."""

    def test_act_has_auth_dependency(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        act_idx = None
        for i, line in enumerate(lines):
            if '@app.post("/act")' in line:
                act_idx = i
                break
        assert act_idx is not None
        func_line = lines[act_idx + 1]
        assert 'auth_dependency' in func_line, "/act should have auth_dependency"


class TestWebSocketAuth:
    """V-252: WebSocket auth in shadowd."""

    def test_ws_endpoint_has_auth_check(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        assert 'SHADOW_AUTH_ENABLED' in source.split('websocket_endpoint')[1].split('def ')[0]

    def test_ws_with_id_has_auth_check(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        ws_section = source.split('websocket_endpoint_with_id')[1].split('async def')[0]
        assert 'SHADOW_AUTH_ENABLED' in ws_section


class TestLoginRateLimiting:
    """V-251: Rate limiting on /auth/login and /auth/refresh."""

    def test_check_login_rate_limit_exists(self):
        from shadowd.shadowd import _check_login_rateLimit
        assert callable(_check_login_rateLimit)

    def test_rate_limit_allows_initial_attempts(self):
        from shadowd.shadowd import _check_login_rateLimit, _login_rate_limits
        _login_rate_limits.clear()
        assert _check_login_rateLimit("test_client") is True

    def test_rate_limit_blocks_after_max(self):
        from shadowd.shadowd import _check_login_rateLimit, _login_rate_limits
        _login_rate_limits.clear()
        for _ in range(5):
            _check_login_rateLimit("test_client_block")
        assert _check_login_rateLimit("test_client_block") is False

    def test_login_has_rate_limit(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        login_section = source[source.index('async def login('):source.index('async def login(') + 500]
        assert 'rate' in login_section.lower() or '429' in login_section

    def test_refresh_has_rate_limit(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        refresh_section = source[source.index('async def refresh_token('):source.index('async def refresh_token(') + 500]
        assert 'rate' in refresh_section.lower() or '429' in refresh_section


class TestShikigamiSummonAuth:
    """V-253: Auth on /shikigami/summon."""

    def test_summon_has_auth_dependency(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        summon_idx = None
        for i, line in enumerate(lines):
            if '@app.post("/shikigami/summon")' in line:
                summon_idx = i
                break
        assert summon_idx is not None
        func_line = lines[summon_idx + 1]
        assert 'auth_dependency' in func_line


class TestBroadcastValidation:
    """V-254: Broadcast endpoint message validation."""

    def test_broadcast_has_size_check(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        broadcast_section = source[source.index('async def broadcast_message('):]
        broadcast_section = broadcast_section[:broadcast_section.index('return {')]
        assert '65536' in broadcast_section or 'too large' in broadcast_section.lower()


class TestGetDataEndpointsAuth:
    """V-256/V-257/V-258: Auth on data-exposing GET endpoints."""

    def test_status_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def status(' in line:
                assert 'auth_dependency' in line
                break

    def test_coordinator_status_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def coordinator_status(' in line:
                assert 'auth_dependency' in line
                break

    def test_snapshot_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def unified_snapshot(' in line:
                assert 'auth_dependency' in line
                break

    def test_insights_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def get_insights(' in line:
                assert 'auth_dependency' in line
                break

    def test_observations_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def get_observations(' in line:
                assert 'auth_dependency' in line
                break

    def test_ws_clients_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def get_ws_clients(' in line:
                assert 'auth_dependency' in line
                break

    def test_orchestrator_process_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def orchestrator_process(' in line:
                assert 'auth_dependency' in line
                break

    def test_orchestrator_learn_has_auth(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'async def orchestrator_learn(' in line:
                assert 'auth_dependency' in line
                break


class TestConfidenceSatisfactionValidation:
    """V-260: Confidence/satisfaction range validation."""

    def test_satisfaction_clamped(self, tmp_path):
        from shadowd.decision_learning_v2 import DecisionLearningV2, DecisionV2, DecisionStatus
        dl = DecisionLearningV2(data_dir=tmp_path)
        decision = DecisionV2(
            id="test_dec",
            context="",
            situation="test",
            options=["a", "b"],
            chosen="a",
            reasoning="test",
        )
        dl.decisions["test_dec"] = decision
        dl.decide("test_dec", "a", "because", 5.0)
        result = dl.record_outcome("test_dec", "good", satisfaction=15)
        assert result.satisfaction <= 10

    def test_confidence_clamped(self, tmp_path):
        from shadowd.decision_learning_v2 import DecisionLearningV2, DecisionV2
        dl = DecisionLearningV2(data_dir=tmp_path)
        decision = DecisionV2(
            id="test_dec2",
            context="",
            situation="test",
            options=["a", "b"],
            chosen="",
            reasoning="",
        )
        dl.decisions["test_dec2"] = decision
        result = dl.decide("test_dec2", "a", "because", confidence=5.0)
        assert result.confidence <= 1.0


class TestObserveRequestValidation:
    """V-263: ObserveRequest content size validation."""

    def test_has_content_validator(self):
        from shadowd.shadowd import ObserveRequest
        assert hasattr(ObserveRequest, 'validate_content_size') or any(
            'content' in f for f in dir(ObserveRequest) if 'valid' in f.lower()
        )

    def test_content_field_validator(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        assert 'validate_content_size' in source or '64000' in source or '64KB' in source


class TestLoginFieldLengthValidation:
    """Ensure login/refresh requests have max_length on fields."""

    def test_login_email_max_length(self):
        from shadowd.shadowd import LoginRequest
        field_info = LoginRequest.model_fields['email']
        assert 'max_length=200' in str(field_info) or '200' in str(field_info)

    def test_login_password_max_length(self):
        from shadowd.shadowd import LoginRequest
        field_info = LoginRequest.model_fields['password']
        assert 'max_length=200' in str(field_info) or '200' in str(field_info)

    def test_refresh_token_max_length(self):
        from shadowd.shadowd import TokenRequest
        field_info = TokenRequest.model_fields['refresh_token']
        assert 'max_length=500' in str(field_info) or '500' in str(field_info)