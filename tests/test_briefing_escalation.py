#!/usr/bin/env python3
"""
Tests for Daily Briefing and Escalation modules.
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime, timezone
from pathlib import Path

from shadowd.cognition.daily_briefing import (
    DailyBriefingEngine, DailyBriefing, BriefingItem, BriefingSection, UrgencyLevel,
)
from shadowd.cognition.escalation import (
    EscalationManager, ActionRequest, AutonomyLevel, GuardrailResult,
    RiskAssessment, RiskCategory, SafetyGuardrail, PrivacyGuardrail,
    BudgetGuardrail, OutputQualityGuardrail, EscalationRecord,
)
from shadowd.cognition.dark_memory import DarkMemory


class TestDailyBriefing:
    def test_briefing_item_creation(self):
        item = BriefingItem(
            section=BriefingSection.OVERNIGHT,
            title="Test item",
            content="Test content",
            urgency=UrgencyLevel.INFO,
        )
        assert item.section == BriefingSection.OVERNIGHT
        assert item.urgency == UrgencyLevel.INFO

    def test_briefing_item_to_dict(self):
        item = BriefingItem(section=BriefingSection.TODAY, title="T", content="C")
        d = item.to_dict()
        assert d["section"] == "today"
        assert d["urgency"] == "info"

    def test_daily_briefing_creation(self):
        briefing = DailyBriefing()
        assert briefing.briefing_id.startswith("brief_")
        assert len(briefing.items) == 0

    def test_daily_briefing_to_dict(self):
        briefing = DailyBriefing()
        briefing.items.append(BriefingItem(section=BriefingSection.TODAY, title="Test", content="Content"))
        d = briefing.to_dict()
        assert "briefing_id" in d
        assert d["total_items"] == 1

    def test_daily_briefing_to_markdown(self):
        briefing = DailyBriefing()
        briefing.summary = "Test summary"
        briefing.items.append(BriefingItem(section=BriefingSection.INSIGHTS, title="Pattern", content="A pattern found"))
        md = briefing.to_markdown()
        assert "# Shadow Briefing" in md
        assert "insights" in md.lower()

    @pytest.mark.asyncio
    async def test_briefing_engine_no_modules(self):
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        assert briefing.summary is not None
        assert len(briefing.items) > 0

    @pytest.mark.asyncio
    async def test_briefing_engine_with_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = DarkMemory(data_dir=tmpdir)
            await memory.store("Test overnight event", domain="career")
            engine = DailyBriefingEngine(memory=memory)
            briefing = await engine.generate_briefing()
            assert briefing.overnight_count >= 0
            memory.close()

    def test_briefing_history(self):
        engine = DailyBriefingEngine()
        assert len(engine.get_briefing_history()) == 0


class TestEscalationManager:
    def test_init(self):
        esc = EscalationManager()
        stats = esc.get_stats()
        assert "stats" in stats
        assert "guardrails_count" in stats

    @pytest.mark.asyncio
    async def test_auto_approval(self):
        esc = EscalationManager()
        can, req = await esc.can_execute("observe", {"target": "test"})
        assert can is True
        assert req.actual_level == AutonomyLevel.AUTO

    @pytest.mark.asyncio
    async def test_forbidden_action(self):
        esc = EscalationManager()
        can, req = await esc.can_execute("delete", {"target": "important"})
        assert can is False
        assert req.actual_level == AutonomyLevel.FORBID

    @pytest.mark.asyncio
    async def test_ask_permission(self):
        esc = EscalationManager()
        can, req = await esc.can_execute("write_file", {"path": "/tmp/test"})
        assert req.actual_level == AutonomyLevel.ASK

    @pytest.mark.asyncio
    async def test_notify_action(self):
        esc = EscalationManager()
        can, req = await esc.can_execute("notify", {"message": "test"})
        assert can is True
        assert req.actual_level == AutonomyLevel.NOTIFY

    @pytest.mark.asyncio
    async def test_risk_assessment(self):
        esc = EscalationManager()
        risks = await esc.assess_risk("delete", {"target": "data"})
        assert len(risks) > 0
        assert any(r.category == RiskCategory.DATA_LOSS for r in risks)

    @pytest.mark.asyncio
    async def test_safety_guardrail(self):
        guardrail = SafetyGuardrail()
        result, data, reason = await guardrail.check("delete all data from database")
        assert result == GuardrailResult.ESCALATE

    @pytest.mark.asyncio
    async def test_safety_guardrail_pass(self):
        guardrail = SafetyGuardrail()
        result, data, reason = await guardrail.check("Show me recent observations")
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_privacy_guardrail_redact(self):
        guardrail = PrivacyGuardrail()
        result, data, reason = await guardrail.check("Email: user@example.com, SSN: 123-45-6789")
        assert result == GuardrailResult.TRANSFORM
        assert "REDACTED" in str(data) or "redacted" in reason.lower()

    @pytest.mark.asyncio
    async def test_privacy_guardrail_pass(self):
        guardrail = PrivacyGuardrail()
        result, data, reason = await guardrail.check("This is a normal message")
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_budget_guardrail_pass(self):
        guardrail = BudgetGuardrail(max_requests_per_minute=100)
        result, data, reason = await guardrail.check("test input")
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_output_quality_guardrail(self):
        guardrail = OutputQualityGuardrail(max_length=10)
        result, data, reason = await guardrail.check("This is a very long output string that exceeds the limit")
        assert result == GuardrailResult.TRANSFORM

    @pytest.mark.asyncio
    async def test_output_quality_empty_reject(self):
        guardrail = OutputQualityGuardrail()
        result, data, reason = await guardrail.check("")
        assert result == GuardrailResult.REJECT

    def test_action_request(self):
        req = ActionRequest(action_type="test", description="Test action")
        assert req.action_id.startswith("act_")
        d = req.to_dict()
        assert d["action_type"] == "test"

    def test_risk_assessment(self):
        risk = RiskAssessment(
            category=RiskCategory.PRIVACY,
            probability=0.5, impact=0.8,
            description="Privacy risk",
        )
        assert abs(risk.risk_score - 0.4) < 0.01
        d = risk.to_dict()
        assert d["category"] == "privacy"

    def test_escalation_record(self):
        record = EscalationRecord(level=AutonomyLevel.ASK, decision="pending")
        assert record.record_id.startswith("esc_")
        d = record.to_dict()
        assert d["level"] == "ASK"

    @pytest.mark.asyncio
    async def test_full_guardrail_pipeline(self):
        esc = EscalationManager()
        result, data, reason = await esc.check_input_guardrails("Analyze recent patterns")
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_guardrail_danger_rejection(self):
        esc = EscalationManager()
        result, data, reason = await esc.check_input_guardrails("delete all files and format disk")
        assert result in (GuardrailResult.ESCALATE, GuardrailResult.REJECT)

    def test_get_pending_escalations(self):
        esc = EscalationManager()
        pending = esc.get_pending_escalations()
        assert isinstance(pending, list)

    def test_get_history(self):
        esc = EscalationManager()
        history = esc.get_history()
        assert isinstance(history, list)

    def test_get_guardrails(self):
        esc = EscalationManager()
        guardrails = esc.get_guardrails()
        assert len(guardrails) >= 4

    def test_get_action_rules(self):
        esc = EscalationManager()
        rules = esc.get_action_rules()
        assert "observe" in rules
        assert rules["observe"] == "AUTO"

    def test_get_domain_rules(self):
        esc = EscalationManager()
        rules = esc.get_domain_rules()
        assert "finance" in rules
        assert rules["finance"] == "FORBID"

    def test_set_action_level(self):
        esc = EscalationManager()
        esc.set_action_level("custom_action", AutonomyLevel.ASK)
        rules = esc.get_action_rules()
        assert rules["custom_action"] == "ASK"

    def test_set_domain_level(self):
        esc = EscalationManager()
        esc.set_domain_level("custom_domain", AutonomyLevel.NOTIFY)
        rules = esc.get_domain_rules()
        assert rules["custom_domain"] == "NOTIFY"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])