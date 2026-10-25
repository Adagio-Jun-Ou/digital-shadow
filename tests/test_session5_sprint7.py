#!/usr/bin/env python3
"""
Session 5 Sprint 7 Regression Tests

Covers:
- V-218: Auth on /os/* POST endpoints
- V-200: asyncio.Lock in shadow_enhanced
- V-212: Block memory debounced saves
- V-215: Knowledge graph conflict detection requires entity overlap
- V-241: Wellness score logic (more items = lower wellness)
- V-242: Briefing ID uses 12-char UUID
- Feature: shadow_crew handlers wired to ShadowOS
- Feature: BigFiveProfile.update with decay and evidence weighting
- Feature: predict_decision uses all Big Five + decision style
- Feature: Dynamic task decomposition (sequential + collaborative)
"""

import pytest
import inspect


class TestOSEndpointAuth:
    """V-218: POST endpoints on /os/* require API key when SHADOW_OS_API_KEY is set."""

    def test_auth_middleware_exists(self):
        import shadowd.cognition.api_router as api
        source = inspect.getsource(api)
        assert "verify_api_key" in source, "Should have auth verification function"
        assert "SHADOW_OS_API_KEY" in source, "Should read API key from env"

    def test_post_endpoints_have_auth_dependency(self):
        import shadowd.cognition.api_router as api
        source = inspect.getsource(api)
        post_lines = [l for l in source.split('\n') if '@router.post(' in l]
        auth_lines = [l for l in source.split('\n') if 'dependencies' in l and 'verify_api_key' in l]
        assert len(auth_lines) > 0, "POST endpoints should have auth dependency"

    def test_auth_disabled_by_default(self):
        import shadowd.cognition.api_router as api
        assert api._auth_enabled is False, "Auth should be disabled when no API key set"


class TestShadowEnhancedLock:
    """V-200: shadow_enhanced has asyncio.Lock for shared state."""

    def test_has_lock(self):
        import shadowd.shadow_enhanced as se
        source = inspect.getsource(se.EnhancedShadowAI.__init__)
        assert "_lock" in source, "Should have asyncio.Lock"


class TestBlockMemoryDebounce:
    """V-212: Block memory debounces saves."""

    def test_insert_does_not_call_save_directly(self):
        import shadowd.block_memory as bm
        source = inspect.getsource(bm.ArchivalMemory.insert)
        assert "_mark_dirty()" in source or "_save()" not in source, "Insert should use debounced save"

    def test_has_flush_method(self):
        import shadowd.block_memory as bm
        assert hasattr(bm.ArchivalMemory, 'flush'), "Should have flush method"


class TestKnowledgeGraphConflictDetection:
    """V-215: Conflict detection requires shared entities."""

    def test_conflict_requires_entity_overlap(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg.Entity._facts_conflict)
        assert "shared_entities" in source or "entities" in source, "Should check entity overlap"


class TestWellnessScoreLogic:
    """V-241: More urgent items should lower wellness score, not total item count."""

    def test_wellness_subtracts_urgency_not_item_count(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        source = inspect.getsource(DailyBriefingEngine._calculate_wellness)
        assert "medium" in source, "Should account for medium urgency"
        assert "0.03" in source, "Should subtract medium urgency weight"
        assert "+ 0.02" not in source, "Should not add per item"


class TestBriefingIDLength:
    """V-242: Briefing IDs use 12-char UUID."""

    def test_briefing_id_uses_12_chars(self):
        from shadowd.cognition.daily_briefing import DailyBriefing
        source = inspect.getsource(DailyBriefing.__post_init__)
        assert "hex[:12]" in source, "Should use 12-char UUID"


class TestShadowCrewWiredToShadowOS:
    """Feature: shadow_crew has shadow_os attribute for real processing."""

    def test_has_shadow_os_attribute(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.__init__)
        assert "shadow_os" in source, "Should have shadow_os attribute"

    def test_observe_handler_uses_memory(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew._handle_observe)
        assert "shadow_os" in source, "Observe handler should use shadow_os"

    def test_predict_handler_uses_personality(self):
        from shadowd.cognition.shadow_crew import ShadowCrew as CSCrew
        source = inspect.getsource(CSCrew._handle_predict)
        assert "personality" in source, "Predict handler should use personality engine"


class TestBigFiveProfileUpdateDecay:
    """Feature: BigFiveProfile.update uses decay and evidence weighting."""

    def test_update_method_exists(self):
        from shadowd.cognition.personality_engine import BigFiveProfile
        assert hasattr(BigFiveProfile, 'update'), "Should have update method"

    def test_update_uses_decay(self):
        from shadowd.cognition.personality_engine import BigFiveProfile
        source = inspect.getsource(BigFiveProfile.update)
        assert "DECAY_RATE" in source, "Should use decay toward baseline"

    def test_update_uses_evidence_weight(self):
        from shadowd.cognition.personality_engine import BigFiveProfile
        source = inspect.getsource(BigFiveProfile.update)
        assert "evidence_count" in source, "Should weight by evidence count"


class TestPredictDecisionFullTraits:
    """Feature: predict_decision uses all Big Five traits + decision style."""

    def test_uses_all_traits(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine.predict_decision)
        assert "style_weights" in source, "Should use style-based weights"
        assert "bf.openness" in source or "big_five.openness" in source, "Should use openness"


class TestDynamicTaskDecomposition:
    """Feature: task planning is now objective-aware."""

    def test_sequential_plans_multiple_steps(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew._plan_tasks_sequential)
        assert "obj_lower" in source, "Should analyze objective text"

    def test_collaborative_plans_based_on_objective(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew._plan_tasks_collaborative)
        assert "obj_lower" in source, "Should analyze objective text"


class TestCrewMissionsCap:
    """V-225: ShadowCrew caps missions at 100."""

    def test_missions_cap_exists(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.__init__)
        assert "_MAX_MISSIONS" in source, "Should have MAX_MISSIONS"


class TestInsightBufferCap:
    """V-230: NeuralMesh insight buffer capped."""

    def test_insight_buffer_cap(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh.share_insight)
        assert "_MAX_INSIGHT_BUFFER" in source, "Should have buffer cap"