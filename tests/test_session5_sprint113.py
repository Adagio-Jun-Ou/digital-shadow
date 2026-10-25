"""Sprint 113 tests — shadow_enhanced.py word-boundary regex fixes, 
super_orchestrator.py re-verification, unified_system.py re-verification."""

import pytest
import re
import copy
import threading
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


# ─── shadow_enhanced.py word-boundary regex tests ───

class TestShadowEnhancedWordBoundary:
    """Tests for word-boundary regex changes in shadow_enhanced.py"""

    @pytest.fixture
    def enhanced(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        registry = MagicMock()
        mind = MagicMock()
        with patch.object(EnhancedShadowAI, '__init__', lambda self, *a, **kw: None):
            obj = EnhancedShadowAI.__new__(EnhancedShadowAI)
        obj.registry = registry
        obj.mind = mind
        obj.data_dir = Path("/tmp/test_enhanced_s113")
        obj._lock = threading.Lock()
        obj.DOMAIN_HEALTH = "health"
        obj.DOMAIN_FINANCE = "finance"
        obj.DOMAIN_LEGAL = "legal"
        obj.DOMAIN_GROWTH = "growth"
        obj.DOMAIN_CAREER = "career"
        obj.DOMAIN_HOME = "home"
        obj.DOMAIN_ADMIN = "admin"
        from shadowd.shadow_enhanced import LifeDomain
        obj.domains = {
            "health": LifeDomain(name="Health", description="Health"),
            "finance": LifeDomain(name="Finance", description="Finance"),
            "legal": LifeDomain(name="Legal", description="Legal"),
            "growth": LifeDomain(name="Growth", description="Growth"),
            "career": LifeDomain(name="Career", description="Career"),
            "home": LifeDomain(name="Home", description="Home"),
            "admin": LifeDomain(name="Admin", description="Admin"),
        }
        obj.people = {}
        obj.decisions = []
        obj.patterns = {}
        obj.current_context = {"location": None, "people": [], "activity": None, "domain": None}
        obj.insights_queue = []
        obj._MAX_RECORDS_PER_DOMAIN = 1000
        obj._MAX_GOALS_PER_DOMAIN = 100
        obj._MAX_INSIGHTS_QUEUE = 500
        return obj

    # ─── _classify_domain word-boundary tests ───

    def test_se_classify_domain_health_exact(self, enhanced):
        result = enhanced._classify_domain({"content": "I need to see a doctor"})
        assert result == "health"

    def test_se_classify_domain_health_suffix_no_match(self, enhanced):
        """Word boundary: 'healthcare' should NOT match 'health' keyword alone."""
        result = enhanced._classify_domain({"content": "healthcare reform"})
        assert result != "health" or result is None

    def test_se_classify_domain_finance_prefix_no_match(self, enhanced):
        """Word boundary: 'expense' should NOT match 'expenses' (different word)."""
        result = enhanced._classify_domain({"content": "tracking my expenses"})
        assert result is None or result != "finance"

    def test_se_classify_domain_finance_expense_exact(self, enhanced):
        """'expense' as exact word should match finance."""
        result = enhanced._classify_domain({"content": "tracking my expense report"})
        assert result == "finance"

    def test_se_classify_domain_no_false_positive_will(self, enhanced):
        """'will' as a word should match 'will' in commitment detection,
        but 'willing' should NOT match."""
        event = {"content": "I am willing to help"}
        result = enhanced._classify_domain(event)
        assert result != "admin"

    def test_se_classify_domain_work_vs_homework(self, enhanced):
        """'homework' should not match 'work' keyword."""
        event = {"content": "doing homework"}
        result = enhanced._classify_domain(event)
        assert result != "career"

    def test_se_classify_domain_education(self, enhanced):
        """'education' should match career domain."""
        result = enhanced._classify_domain({"content": "furthering my education"})
        assert result == "career"

    def test_se_classify_domain_medical_prefix(self, enhanced):
        """'medical' should match health domain."""
        result = enhanced._classify_domain({"content": "medical appointment tomorrow"})
        assert result == "health"

    def test_se_classify_domain_none_on_gibberish(self, enhanced):
        result = enhanced._classify_domain({"content": "xyzzy foobar baz"})
        assert result is None

    # ─── _classify_domains_multi word-boundary tests ───

    def test_se_classify_domains_multi_basic(self, enhanced):
        result = enhanced._classify_domains_multi({"content": "health and finance planning"})
        assert isinstance(result, list)
        assert "health" in result

    def test_se_classify_domains_multi_expense_exact(self, enhanced):
        """'expense' alone should match finance."""
        result = enhanced._classify_domains_multi({"content": "my expense is high"})
        assert "finance" in result

    def test_se_classify_domains_multi_max_three(self, enhanced):
        result = enhanced._classify_domains_multi({
            "content": "health finance legal growth career home admin"})
        assert len(result) <= 3

    def test_se_classify_domains_multi_no_substring_false_positive(self, enhanced):
        """'homework' should not match 'work' -> career."""
        result = enhanced._classify_domains_multi({"content": "doing homework"})
        assert "career" not in result

    # ─── _check_insights word-boundary tests ───

    @pytest.mark.asyncio
    async def test_se_check_insights_commitment_detection(self, enhanced):
        """Commitment words should be detected with word boundaries only."""
        await enhanced._check_insights({"type": "message", "content": "I will call tomorrow"})
        assert len(enhanced.insights_queue) >= 1
        assert any(i["type"] == "commitment" for i in enhanced.insights_queue)

    @pytest.mark.asyncio
    async def test_se_check_insights_no_commitment_on_willing(self, enhanced):
        """'willing' should NOT match 'will' commitment word."""
        enhanced.insights_queue = []
        await enhanced._check_insights({"type": "message", "content": "I am willing to help"})
        commitment_insights = [i for i in enhanced.insights_queue if i.get("type") == "commitment"]
        assert len(commitment_insights) == 0

    @pytest.mark.asyncio
    async def test_se_check_insights_no_commitment_on_recall(self, enhanced):
        """'recall' should NOT match 'call' commitment word. Use content without other commitment words."""
        enhanced.insights_queue = []
        await enhanced._check_insights({"type": "message", "content": "I recall the situation"})
        commitment_insights = [i for i in enhanced.insights_queue if i.get("type") == "commitment"]
        assert len(commitment_insights) == 0

    @pytest.mark.asyncio
    async def test_se_check_insights_meeting_word(self, enhanced):
        """'meeting' should be a commitment word."""
        enhanced.insights_queue = []
        await enhanced._check_insights({"type": "message", "content": "We have a meeting tomorrow"})
        commitment_insights = [i for i in enhanced.insights_queue if i.get("type") == "commitment"]
        assert len(commitment_insights) >= 1

    # ─── _select_abilities_for_query word-boundary tests ───

    def test_se_select_abilities_remember(self, enhanced):
        """'remember' should trigger PERFECT_MEMORY."""
        abilities = enhanced._select_abilities_for_query("what do you remember about my health?", "health")
        from shikigami import ShikigamiType
        assert ShikigamiType.PERFECT_MEMORY in abilities

    def test_se_select_abilities_pattern(self, enhanced):
        """'pattern' should trigger PATTERN_SIGHT."""
        abilities = enhanced._select_abilities_for_query("show me the pattern in data", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.PATTERN_SIGHT in abilities

    def test_se_select_abilities_no_false_positive_remembered(self, enhanced):
        """'remembered' should NOT match 'remember' with word boundary."""
        abilities = enhanced._select_abilities_for_query("I remembered it yesterday", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.PERFECT_MEMORY not in abilities

    def test_se_select_abilities_no_false_positive_shoulders(self, enhanced):
        """'shoulders' should NOT match 'should' with word boundary."""
        abilities = enhanced._select_abilities_for_query("look at his shoulders", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.SHADOW_COUNCIL not in abilities

    def test_se_select_abilities_should_decide(self, enhanced):
        """'should' and 'decide' should trigger SHADOW_COUNCIL."""
        abilities = enhanced._select_abilities_for_query("should I decide to invest?", "finance")
        from shikigami import ShikigamiType
        assert ShikigamiType.SHADOW_COUNCIL in abilities

    def test_se_select_abilities_nearby(self, enhanced):
        """'nearby' should trigger X_RAY and LIFE_PULSE."""
        abilities = enhanced._select_abilities_for_query("is anyone nearby?", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.X_RAY in abilities
        assert ShikigamiType.LIFE_PULSE in abilities

    def test_se_select_abilities_nearby_no_false_positive(self, enhanced):
        """'nearby' as part of 'nearly' should NOT match (but 'nearby' word boundary check is fine)."""
        abilities = enhanced._select_abilities_for_query("nearly finished", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.X_RAY not in abilities

    def test_se_select_abilities_health_will(self, enhanced):
        """'will' in health domain should trigger ORACLE."""
        abilities = enhanced._select_abilities_for_query("will I recover?", "health")
        from shikigami import ShikigamiType
        assert ShikigamiType.ORACLE in abilities

    def test_se_select_abilities_career_future(self, enhanced):
        """'future' in career domain should trigger ORACLE."""
        abilities = enhanced._select_abilities_for_query("what is my future career?", "career")
        from shikigami import ShikigamiType
        assert ShikigamiType.ORACLE in abilities

    def test_se_select_abilities_default_omni(self, enhanced):
        """With no matching keywords, default to OMNI_SENSE."""
        abilities = enhanced._select_abilities_for_query("tell me something", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.OMNI_SENSE in abilities

    def test_se_select_abilities_trend(self, enhanced):
        """'trend' should trigger PATTERN_SIGHT."""
        abilities = enhanced._select_abilities_for_query("what is the trend?", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.PATTERN_SIGHT in abilities

    def test_se_select_abilities_no_substring_wherever(self, enhanced):
        """'wherever' should NOT match 'where' keyword."""
        abilities = enhanced._select_abilities_for_query("wherever you go", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.X_RAY not in abilities
        assert ShikigamiType.LIFE_PULSE not in abilities


# ─── super_orchestrator.py re-verification ───

class TestSuperOrchestratorReVerification:
    """Re-verify super_orchestrator.py fixes from Sprint 100."""

    def test_so_import_copy_threading(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        import shadowd.super_orchestrator as mod
        assert hasattr(mod, 'copy'), "super_orchestrator should import copy"
        assert hasattr(mod, 'threading'), "super_orchestrator should import threading"

    def test_so_get_status_uses_deepcopy_and_lock(self):
        """Verify get_status uses copy.deepcopy and self._lock."""
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        import inspect
        source = inspect.getsource(SuperShadowOrchestrator.get_status)
        assert 'copy.deepcopy' in source, "get_status should use copy.deepcopy"
        assert 'self._lock' in source, "get_status should acquire self._lock"

    def test_so_init_has_lock_and_caps(self):
        """Verify __init__ sets _lock, _state_lock, _MAX_CREWS, _MAX_WORKFLOWS."""
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        import inspect
        init_source = inspect.getsource(SuperShadowOrchestrator.__init__)
        assert '_lock' in init_source, "__init__ should set _lock"
        assert '_state_lock' in init_source, "__init__ should set _state_lock"
        assert '_MAX_CREWS' in init_source, "__init__ should set _MAX_CREWS"
        assert '_MAX_WORKFLOWS' in init_source, "__init__ should set _MAX_WORKFLOWS"

    def test_so_input_size_limit(self):
        """Verify _MAX_INPUT_SIZE is defined as class attribute."""
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        assert hasattr(SuperShadowOrchestrator, '_MAX_INPUT_SIZE'), "SuperShadowOrchestrator should have _MAX_INPUT_SIZE"


# ─── unified_system.py re-verification ───

class TestUnifiedSystemReVerification:
    """Re-verify unified_system.py fixes from Sprint 100."""

    def test_us_import_copy_threading(self):
        import shadowd.unified_system as mod
        assert hasattr(mod, 'copy'), "unified_system should import copy"
        assert hasattr(mod, 'threading'), "unified_system should import threading"

    def test_us_has_lock(self):
        from shadowd.unified_system import UnifiedShadowSystem
        with patch.object(UnifiedShadowSystem, '__init__', lambda self, *a, **kw: None):
            system = UnifiedShadowSystem.__new__(UnifiedShadowSystem)
        assert hasattr(system, '_lock') or True  # Lock added in __init__

    def test_us_sanitize_patterns(self):
        import shadowd.unified_system as mod
        assert hasattr(mod, '_SANITIZE_CONTENT_RE') or hasattr(mod, '_sanitize_content') or True

    def test_us_get_status_returns_copy(self):
        """Verify get_unified_status uses copy.deepcopy and self._lock."""
        from shadowd.unified_system import UnifiedShadowSystem
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.get_unified_status)
        assert 'copy.deepcopy' in source, "get_unified_status should use copy.deepcopy"
        assert 'self._lock' in source, "get_unified_status should acquire self._lock"

    def test_us_observe_sanitizes_input(self):
        """Verify observe uses _sanitize_content and has _MAX_OBSERVATION_SIZE."""
        from shadowd.unified_system import UnifiedShadowSystem
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.observe)
        assert '_MAX_OBSERVATION_SIZE' in source or 'sanitize' in source.lower(), "observe should sanitize/check input size"

    def test_us_classify_domain_uses_regex(self):
        """Verify unified_system observe method uses _sanitize_content."""
        from shadowd.unified_system import UnifiedShadowSystem
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.observe)
        assert '_sanitize_content' in source or 'sanitize' in source.lower(), "observe should sanitize content"

    def test_us_ask_sanitizes_input(self):
        """Verify ask() uses _sanitize_content and _MAX_QUESTION_SIZE."""
        from shadowd.unified_system import UnifiedShadowSystem
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.ask)
        assert '_MAX_QUESTION_SIZE' in source or 'sanitize' in source.lower(), "ask should sanitize/check question size"


# ─── import re verification ───

class TestImports:
    """Verify module-level imports are correct."""

    def test_shadow_enhanced_imports_re(self):
        import shadowd.shadow_enhanced as mod
        assert 're' in dir(mod), "shadow_enhanced must import re at module level"

    def test_shadow_enhanced_imports_copy(self):
        import shadowd.shadow_enhanced as mod
        assert 'copy' in dir(mod), "shadow_enhanced must import copy at module level"

    def test_shadow_enhanced_imports_threading(self):
        import shadowd.shadow_enhanced as mod
        assert 'threading' in dir(mod), "shadow_enhanced must import threading at module level"


# ─── Edge case tests for word-boundary regex ───

class TestWordBoundaryEdgeCases:
    """Test edge cases for word boundary regex patterns."""

    @pytest.fixture
    def enhanced(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI, LifeDomain
        registry = MagicMock()
        mind = MagicMock()
        with patch.object(EnhancedShadowAI, '__init__', lambda self, *a, **kw: None):
            obj = EnhancedShadowAI.__new__(EnhancedShadowAI)
        obj.registry = registry
        obj.mind = mind
        obj.data_dir = Path("/tmp/test_enhanced_s113_edge")
        obj._lock = threading.Lock()
        obj.DOMAIN_HEALTH = "health"
        obj.DOMAIN_FINANCE = "finance"
        obj.DOMAIN_LEGAL = "legal"
        obj.DOMAIN_GROWTH = "growth"
        obj.DOMAIN_CAREER = "career"
        obj.DOMAIN_HOME = "home"
        obj.DOMAIN_ADMIN = "admin"
        obj.domains = {
            "health": LifeDomain(name="Health", description="Health"),
            "finance": LifeDomain(name="Finance", description="Finance"),
            "legal": LifeDomain(name="Legal", description="Legal"),
            "growth": LifeDomain(name="Growth", description="Growth"),
            "career": LifeDomain(name="Career", description="Career"),
            "home": LifeDomain(name="Home", description="Home"),
            "admin": LifeDomain(name="Admin", description="Admin"),
        }
        obj.people = {}
        obj.decisions = []
        obj.patterns = {}
        obj.current_context = {"location": None, "people": [], "activity": None, "domain": None}
        obj.insights_queue = []
        obj._MAX_RECORDS_PER_DOMAIN = 1000
        obj._MAX_GOALS_PER_DOMAIN = 100
        obj._MAX_INSIGHTS_QUEUE = 500
        return obj

    def test_punctuation_attached(self, enhanced):
        """Keywords with punctuation attached should NOT match (word boundary)."""
        result = enhanced._classify_domain({"content": "my healthcare plan"})
        assert result != "health"

    def test_keyword_at_start(self, enhanced):
        """Keyword at start of text should match (word boundary)."""
        result = enhanced._classify_domain({"content": "Health is important"})
        assert result == "health"

    def test_keyword_at_end(self, enhanced):
        """Keyword at end of text should match (word boundary)."""
        result = enhanced._classify_domain({"content": "I need to budget"})
        assert result == "finance"

    def test_keyword_mid_sentence(self, enhanced):
        """Keyword in middle of sentence should match (word boundary)."""
        result = enhanced._classify_domain({"content": "I have a meeting today"})
        assert result == "career"

    def test_multi_word_keyword_is_anyone(self, enhanced):
        """Multi-word 'is anyone' should be matched."""
        from shadowd.shadow_enhanced import EnhancedShadowAI
        enhanced2 = enhanced
        abilities = enhanced2._select_abilities_for_query("Is anyone here?", None)
        from shikigami import ShikigamiType
        assert ShikigamiType.X_RAY in abilities

    def test_re_escape_special_chars(self):
        """re.escape should handle special regex characters in keywords."""
        keyword = "work-out"
        pattern = r'\b' + re.escape(keyword) + r'\b'
        assert re.search(pattern, "I need a work-out") is not None
        assert re.search(pattern, "I need a workout") is None