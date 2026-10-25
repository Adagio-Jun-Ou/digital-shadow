import pytest
import threading
import copy


class TestDomainExpertiseDeepAudit:
    def test_get_domain_summary_returns_deepcopy(self):
        from shadowd.domain_expertise import DomainExpertiseManager, DomainKnowledge
        m = DomainExpertiseManager.__new__(DomainExpertiseManager)
        m._lock = threading.Lock()
        m.experts = {}
        m.domain_observations = {"career": [{"content": "test_obs", "timestamp": "2024-01-01"}]}
        m.experts["career"] = type("E", (), {"knowledge": DomainKnowledge(domain="career", keywords=["python", "git"])})()
        result = m.get_domain_summary("career")
        assert result["observation_count"] == 1
        result["recent"].append({"mutated": True})
        assert len(m.domain_observations["career"]) == 1

    def test_get_domain_summary_unknown_domain(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        m = DomainExpertiseManager.__new__(DomainExpertiseManager)
        m._lock = threading.Lock()
        m.experts = {}
        result = m.get_domain_summary("nonexistent")
        assert result["observation_count"] == 0


class TestLearningSystemDeepAudit:
    def test_get_status_under_lock(self):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem()
        status = ls.get_status()
        assert "decisions" in status
        assert "patterns" in status


class TestShadowAIDeepAudit:
    def test_get_status_under_lock(self):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI.__new__(ShadowAI)
        ai._lock = threading.Lock()
        ai.observations = []
        ai.people = {}
        ai.decisions = []
        ai.patterns = {}
        ai.insights = []
        status = ai.get_status()
        assert "observations" in status
        assert status["observations"] == 0


class TestProactiveInsightsDeepAudit:
    def test_lock_exists(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        e = ProactiveInsightEngine()
        assert hasattr(e, '_lock')

    def test_get_summary_under_lock(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        e = ProactiveInsightEngine()
        summary = e.get_summary()
        assert "total_insights" in summary
        assert "active_insights" in summary


class TestCalmAmbientDeepAudit:
    def test_get_glanceable_context_under_lock(self):
        from shadowd.calm_ambient import CalmAmbient
        c = CalmAmbient()
        result = c.get_glanceable_context()
        assert "attention" in result
        assert "context" in result

    def test_get_detailed_context_returns_copies(self):
        from shadowd.calm_ambient import CalmAmbient
        c = CalmAmbient()
        result = c.get_detailed_context()
        assert "current_context" in result
        assert "observations" in result
        assert isinstance(result["suggestions"], list)


class TestShadowEnhancedLockAudit:
    def test_get_domain_summary_deepcopy(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI, LifeDomain
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.domains = {
            "test": LifeDomain(
                name="Test", description="test domain",
                records=[{"content": "rec1"}],
                insights=["ins1"],
                goals=[{"goal": "g1"}],
            ),
        }
        result = ai.get_domain_summary("test")
        assert result["records_count"] == 1
        result["recent_records"].append({"mutated": True})
        assert len(ai.domains["test"].records) == 1

    def test_get_status_context_copy(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.current_context = {"domain": "career"}
        ai.insights_queue = []
        ai.people = {}
        ai.decisions = {}
        ai.patterns = {}
        ai.domains = {}
        ai.memory = type("M", (), {"get_stats": lambda self: {}})()
        status = ai.get_status()
        assert status["current_context"]["domain"] == "career"
        status["current_context"]["domain"] = "MUTATED"
        assert ai.current_context["domain"] == "career"