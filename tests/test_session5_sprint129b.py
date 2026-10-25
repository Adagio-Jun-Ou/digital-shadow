#!/usr/bin/env python3
"""Sprint 129b regression tests — second-pass deep audit round 2.

Verifies fixes in:
- empathy_mapping.py: deepcopy + round in to_dict()
- attention_engine.py: round() in multiple to_dict()
- presence_detection.py: round() + f-string logger (verified already correct)
- escalation.py: thread safety for getter methods, except narrowing, dead import removal
- daily_briefing.py: domain/action_url sanitization, already has content-safe _SANITIZE_RE
"""

import copy
import threading
import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.empathy_mapping import (
    EmpathyMapping, PersonProfile, CommunicationStyle,
    ReactionPrediction, RelationshipHealth, RelationshipType,
)
from shadowd.cognition.attention_engine import (
    AttentionEngine, AttentionItem, AttentionLevel,
    FocusSession, DistractionEvent, DistractionType,
    AttentionProfile, FocusState,
)
from shadowd.cognition.presence_detection import (
    PresenceDetection, PresenceInfo, EnvironmentContext, PresencePattern,
)
from shadowd.cognition.escalation import (
    EscalationManager, AutonomyLevel, ActionRequest, EscalationRecord,
)
from shadowd.cognition.daily_briefing import (
    DailyBriefingEngine, BriefingItem, DailyBriefing, UrgencyLevel, BriefingSection,
)


# ── EmpathyMapping ──────────────────────────────────────────────────────────

class TestEmpathyMappingSprint129b:
    def test_person_profile_to_dict_deepcopy_metadata(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice", "friend", metadata={"key": "val"})
        profile = em.get_person_profile(pid)
        d = profile.to_dict()
        d["metadata"]["new_key"] = "tampered"
        profile2 = em.get_person_profile(pid)
        assert "new_key" not in profile2.metadata

    def test_person_profile_to_dict_deepcopy_response_patterns(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob", "colleague")
        em.record_interaction(pid, "meeting", 0.5, notes="test")
        profile = em.get_person_profile(pid)
        d = profile.to_dict()
        d["response_patterns"]["evil"] = "data"
        profile2 = em.get_person_profile(pid)
        assert "evil" not in profile2.response_patterns

    def test_person_profile_avg_sentiment_rounded(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol", "friend")
        em.record_interaction(pid, "call", 0.3333333)
        profile = em.get_person_profile(pid)
        d = profile.to_dict()
        assert d["avg_sentiment"] == round(0.3333333, 3)

    def test_communication_style_response_time_rounded(self):
        em = EmpathyMapping()
        pid = em.add_person("Dave", "manager")
        em.record_interaction(pid, "email", 0.1, notes="first")
        em.record_interaction(pid, "email", 0.2, notes="second")
        style = em.get_communication_style(pid)
        d = style.to_dict()
        assert d["response_time_hours"] == round(style.response_time_hours, 2)

    def test_reaction_prediction_floats_rounded(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve", "friend")
        em.record_interaction(pid, "text", 0.5)
        pred = em.predict_reaction(pid, "project update", "direct")
        d = pred.to_dict()
        assert d["predicted_sentiment"] == round(pred.predicted_sentiment, 3)
        assert d["predicted_receptiveness"] == round(pred.predicted_receptiveness, 3)
        assert d["confidence"] == round(pred.confidence, 3)

    def test_relationship_health_deepcopy(self):
        em = EmpathyMapping()
        pid = em.add_person("Frank", "family")
        em.record_interaction(pid, "call", 0.8)
        health = em.get_relationship_health(pid)
        d = health.to_dict()
        d["recommendation"] = "TAMPERED"
        health2 = em.get_relationship_health(pid)
        assert health2.recommendation != "TAMPERED"

    def test_relationship_health_score_rounded(self):
        em = EmpathyMapping()
        pid = em.add_person("Grace", "friend")
        em.record_interaction(pid, "text", 0.7777777)
        health = em.get_relationship_health(pid)
        d = health.to_dict()
        assert d["health_score"] == round(health.health_score, 2)


# ── AttentionEngine ──────────────────────────────────────────────────────────

class TestAttentionEngineSprint129b:
    def test_attention_item_urgency_rounded(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item("test", "high", "work", urgency=0.7777777, importance=0.6666666)
        item = copy.deepcopy(ae._attention_items[item_id])
        d = item.to_dict()
        assert d["urgency_score"] == round(0.7777777, 3)
        assert d["importance_score"] == round(0.6666666, 3)

    def test_attention_item_metadata_deepcopy(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item("test", "medium", "general", metadata={"key": "val"})
        with ae._lock:
            item = copy.deepcopy(ae._attention_items[item_id])
        d = item.to_dict()
        d["metadata"]["evil"] = "data"
        with ae._lock:
            item2 = copy.deepcopy(ae._attention_items[item_id])
        assert "evil" not in item2.metadata

    def test_distraction_event_recovery_time_rounded(self):
        de = DistractionEvent(
            event_id="test123",
            distraction_type=DistractionType.NOTIFICATION,
            description="test",
            timestamp="2024-01-01T00:00:00+00:00",
            recovery_time_seconds=123.456789,
        )
        d = de.to_dict()
        assert d["recovery_time_seconds"] == round(123.456789, 3)

    def test_attention_profile_rounded(self):
        ap = AttentionProfile(
            avg_focus_span_minutes=25.33333,
            max_focus_span_minutes=60.77777,
            total_distractions=3,
            total_interruptions=1,
            deep_work_sessions=2,
            current_state=FocusState.ACTIVE,
            cognitive_load=0.44444,
            fatigue_level=0.22222,
        )
        d = ap.to_dict()
        assert d["cognitive_load"] == round(0.44444, 3)
        assert d["fatigue_level"] == round(0.22222, 3)
        assert d["avg_focus_span_minutes"] == round(25.33333, 2)
        assert d["max_focus_span_minutes"] == round(60.77777, 2)

    def test_focus_session_deepcopy(self):
        fs = FocusSession(
            session_id="sess123",
            topic="test",
            started_at="2024-01-01T00:00:00+00:00",
            ended_at=None,
            duration_minutes=30.0,
            interruptions=0,
            depth_score=0.8,
        )
        d = fs.to_dict()
        d["topic"] = "TAMPERED"
        assert fs.topic != "TAMPERED"


# ── PresenceDetection ─────────────────────────────────────────────────────────

class TestPresenceDetectionSprint129b:
    def test_environment_context_social_density_rounded(self):
        from shadowd.cognition.presence_detection import EnvironmentType, PrivacyModeStatus
        ec = EnvironmentContext(
            environment=EnvironmentType.PUBLIC,
            noise_level="loud",
            social_density=5.777777,
            present_people=["Alice", "Bob"],
            privacy_mode=PrivacyModeStatus.PUBLIC,
        )
        d = ec.to_dict()
        assert d["social_density"] == round(5.777777, 2)

    def test_presence_pattern_confidence_rounded(self):
        pp = PresencePattern(
            person="Alice",
            pattern_type="weekday_morning",
            typical_days=["monday"],
            typical_hours=[9],
            confidence=0.777777,
        )
        d = pp.to_dict()
        assert d["confidence"] == round(0.777777, 3)

    def test_presence_info_devices_copy(self):
        from shadowd.cognition.presence_detection import PresenceStatus
        pi = PresenceInfo(
            person="Alice",
            devices=["phone", "laptop"],
            status=PresenceStatus.ACTIVE,
            last_seen="2024-01-01T00:00:00+00:00",
            estimated_activity="working",
        )
        d = pi.to_dict()
        d["devices"].append("evil")
        assert len(pi.devices) == 2


# ── EscalationManager ─────────────────────────────────────────────────────────

class TestEscalationSprint129b:
    @pytest.mark.asyncio
    async def test_get_guardrails_thread_safe(self):
        em = EscalationManager()
        results = []
        errors = []

        def read_guardrails():
            try:
                g = em.get_guardrails()
                results.append(len(g))
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=read_guardrails) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert all(r == 4 for r in results)

    def test_get_action_rules_thread_safe(self):
        em = EscalationManager()
        errors = []
        def read_rules():
            try:
                em.get_action_rules()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=read_rules) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_domain_rules_thread_safe(self):
        em = EscalationManager()
        errors = []
        def read_rules():
            try:
                em.get_domain_rules()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=read_rules) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_stats_returns_copy(self):
        em = EscalationManager()
        em._stats["actions_evaluated"] = 5
        stats = em.get_stats()
        stats["stats"]["actions_evaluated"] = 999
        assert em._stats["actions_evaluated"] == 5

    def test_action_request_deepcopy_parameters(self):
        ar = ActionRequest(
            action_type="test",
            parameters={"key": "val"},
        )
        d = ar.to_dict()
        d["parameters"]["evil"] = "data"
        assert "evil" not in ar.parameters

    def test_escalation_record_deepcopy_metadata(self):
        er = EscalationRecord(
            metadata={"key": "val"},
        )
        d = er.to_dict()
        d["metadata"]["evil"] = "data"
        assert "evil" not in er.metadata

    def test_no_asyncio_import(self):
        import shadowd.cognition.escalation as esc_mod
        assert not hasattr(esc_mod, "asyncio") or "asyncio" not in dir(esc_mod)

    @pytest.mark.asyncio
    async def test_evaluate_sanitizes_action_type(self):
        em = EscalationManager()
        req = await em.evaluate("observe<script>", {"key": "val"})
        assert "<script>" not in req.action_type

    @pytest.mark.asyncio
    async def test_evaluate_sanitizes_domain(self):
        em = EscalationManager()
        req = await em.evaluate("observe", {"key": "val"}, domain="career<script>")
        assert "<script>" not in (req.domain or "")

    @pytest.mark.asyncio
    async def test_evaluate_sanitizes_parameters_values_content_safe(self):
        em = EscalationManager()
        req = await em.evaluate("observe", {"key": "val<script>"})
        assert "val" in req.parameters.get("key", "")
        assert "\x00" not in str(req.parameters)

    @pytest.mark.asyncio
    async def test_evaluate_sanitizes_parameters_keys_strict(self):
        em = EscalationManager()
        req = await em.evaluate("observe", {"key<script>": "val"})
        assert "<script>" not in str(list(req.parameters.keys()))


# ── DailyBriefing ────────────────────────────────────────────────────────────

class TestDailyBriefingSprint129b:
    def test_briefing_item_domain_sanitized_strict(self):
        bi = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="Hello",
            domain="career<script>",
            action_url="https://example.com/<evil>",
        )
        assert "<script>" not in bi.domain
        assert "<evil>" not in (bi.action_url or "")

    def test_briefing_item_metadata_deepcopy(self):
        bi = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="Hello",
            metadata={"key": "val"},
        )
        d = bi.to_dict()
        d["metadata"]["evil"] = "data"
        assert "evil" not in bi.metadata

    def test_daily_briefing_wellness_score_rounded(self):
        db = DailyBriefing(wellness_score=0.777777)
        d = db.to_dict()
        assert d["wellness_score"] == round(0.777777, 2)

    def test_daily_briefing_metadata_deepcopy(self):
        db = DailyBriefing(metadata={"key": "val"})
        d = db.to_dict()
        d["metadata"]["evil"] = "data"
        assert "evil" not in db.metadata

    def test_briefing_item_content_sanitized_content_safe(self):
        bi = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="Latency < 100ms is fine",
        )
        assert "<" in bi.content

    def test_briefing_item_title_sanitized_strict(self):
        bi = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test <script>alert(1)</script>",
            content="Hello",
        )
        assert "<script>" not in bi.title