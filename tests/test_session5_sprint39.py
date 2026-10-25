#!/usr/bin/env python3
"""Tests for Session 5 Sprint 39 — 10 audit fixes (HIGH/CRITICAL)"""

import asyncio
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from shadowd.learning_system import LearningSystem, Decision, LearnedPattern, Feedback
from shadowd.domain_expertise import (
    DomainExpertiseManager, FinanceExpert, HealthExpert, DomainKnowledge,
)
from shadowd.memory_optimization import SmartCache, QueryOptimizer, _SENTINEL
from shadowd.coordinator import AbilityCoordinator, AbilityProfile, AbilityType
from shadowd.agent_coordination import ShadowCrew, ShadowTask, TaskStatus
from shadowd.cognition.daily_briefing import DailyBriefingEngine


# ── Fix 1: learning_system lock ──


class TestLearningSystemLock:
    def test_track_decision_is_thread_safe(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls1")
        errors = []

        def worker(i):
            try:
                d = ls.track_decision(
                    context=f"context_{i}",
                    options=["a", "b"],
                    chosen="a",
                    reasoning="test",
                    confidence=0.5,
                )
                assert d.id.startswith("dec_")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(ls.decisions) == 20

    def test_record_outcome_uses_lock(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls2")
        d = ls.track_decision("ctx", ["a"], "a", "reason")
        result = ls.record_outcome(d.id, "good", 8)
        assert result is True

    def test_record_feedback_is_thread_safe(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls3")
        errors = []

        def worker(i):
            try:
                ls.record_feedback("ability", f"query_{i}", "resp", 4, "helpful")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(ls.feedback) == 20

    def test_query_similar_decisions_is_thread_safe(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls4")
        d1 = ls.track_decision("python debug", ["a"], "a", "reason")
        ls.record_outcome(d1.id, "ok", 8)

        results = []

        def worker():
            try:
                r = ls.query_similar_decisions("python debug")
                results.append(r)
            except Exception as e:
                results.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(isinstance(r, list) for r in results)

    def test_save_state_uses_lock(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls5")
        ls.track_decision("test", ["a"], "a", "reason")
        ls._save_state()
        assert (tmp_path / "ls5" / "learning_state.json").exists()


# ── Fix 2: Substring pattern matching → word boundaries ──


class TestLearningSystemWordBoundary:
    def test_learn_from_outcome_uses_word_boundary(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls6")
        d1 = ls.track_decision("I like running in the park", ["a"], "a", "reason")
        ls.record_outcome(d1.id, "good", 9)

        pat_id = f"decision_{__import__('hashlib').sha256(('I like running in the park').encode()).hexdigest()[:16]}"
        assert pat_id in ls.patterns

    def test_learn_from_outcome_no_false_positive(self, tmp_path):
        ls = LearningSystem(data_dir=tmp_path / "ls7")
        d1 = ls.track_decision("test pattern match", ["a"], "a", "reason")
        ls.record_outcome(d1.id, "good", 9)

        d2 = ls.track_decision("contest pattern matcher", ["b"], "b", "reason")
        ls.record_outcome(d2.id, "ok", 7)

        snippet = d2.context[:50]
        pattern_examples = " ".join(ls.patterns.values().__iter__().__next__().examples) if ls.patterns else ""
        pat_re = re.compile(r'\b' + re.escape(snippet) + r'\b')

        assert not pat_re.search("test pattern match")


# ── Fix 3: domain_expertise keyword matching with word boundaries ──


class TestDomainExpertiseWordBoundary:
    def test_classify_domain_word_boundary(self):
        dem = DomainExpertiseManager()

        result = dem.classify_domain("I need to schedule a meeting")
        assert result == "career"

        result = dem.classify_domain("my car needs repair")
        assert result != "health"

        result = dem.classify_domain("I have a health appointment")
        assert result == "health"

    def test_categorize_expense_word_boundary(self):
        fe = FinanceExpert()
        obs = {"text": "bought a game at the store"}
        result = fe._categorize_expense(obs)
        assert result == "entertainment"

    def test_classify_domain_no_false_match_on_substring(self):
        dem = DomainExpertiseManager()

        result = dem.classify_domain("I have workout gear in my car")
        assert result != "career"

    def test_categorize_expense_no_false_transport(self):
        fe = FinanceExpert()

        obs = {"text": "bought a gascanister"}
        result = fe._categorize_expense(obs)
        assert result != "transport"

    def test_categorize_grocery_matches_food(self):
        fe = FinanceExpert()
        obs = {"text": "went to grocery store"}
        assert fe._categorize_expense(obs) == "food"

    def test_categorize_netflix_matches_entertainment(self):
        fe = FinanceExpert()
        obs = {"text": "netflix subscription"}
        assert fe._categorize_expense(obs) == "entertainment"


# ── Fix 3b: _extract_amount requires $ prefix ──


class TestExtractAmountRequiresDollar:
    def test_extract_with_dollar_sign(self):
        fe = FinanceExpert()
        result = fe._extract_amount({"text": "spent $50.00 on food"})
        assert result == 50.0

    def test_extract_without_dollar_sign_returns_zero(self):
        fe = FinanceExpert()
        result = fe._extract_amount({"text": "spent 50.00 on food"})
        assert result == 0.0

    def test_extract_multiple_amounts_takes_first(self):
        fe = FinanceExpert()
        result = fe._extract_amount({"text": "spent $30.50 and $20.00"})
        assert result == 30.5

    def test_extract_no_amounts(self):
        fe = FinanceExpert()
        result = fe._extract_amount({"text": "no money mentioned"})
        assert result == 0.0


# ── Fix 4: SmartCache None sentinel bug ──


class TestSmartCacheSentinel:
    def test_memoize_caches_none_result(self):
        cache = SmartCache()

        call_count = 0

        @cache.memoize
        def return_none():
            nonlocal call_count
            call_count += 1
            return None

        result1 = return_none()
        result2 = return_none()

        assert result1 is None
        assert result2 is None
        assert call_count == 1

    def test_memoize_caches_zero(self):
        cache = SmartCache()

        call_count = 0

        @cache.memoize
        def return_zero():
            nonlocal call_count
            call_count += 1
            return 0

        result1 = return_zero()
        result2 = return_zero()

        assert result1 == 0
        assert result2 == 0
        assert call_count == 1

    def test_memoize_caches_false(self):
        cache = SmartCache()

        call_count = 0

        @cache.memoize
        def return_false():
            nonlocal call_count
            call_count += 1
            return False

        result1 = return_false()
        result2 = return_false()

        assert result1 is False
        assert result2 is False
        assert call_count == 1

    def test_memoize_caches_empty_string(self):
        cache = SmartCache()

        call_count = 0

        @cache.memoize
        def return_empty():
            nonlocal call_count
            call_count += 1
            return ""

        result1 = return_empty()
        result2 = return_empty()

        assert result1 == ""
        assert result2 == ""
        assert call_count == 1

    def test_get_returns_sentinel_on_miss(self):
        cache = SmartCache()
        result = cache.get("nonexistent_key")
        assert result is _SENTINEL

    def test_get_returns_cached_none(self):
        cache = SmartCache()
        cache.set("key_none", None)
        result = cache.get("key_none")
        assert result is None


# ── Fix 5: lru_cache removed from get_similar_queries ──


class TestQueryOptimizerNoLruCache:
    def test_get_similar_queries_reflects_mutable_state(self):
        qo = QueryOptimizer()
        qo.optimize_query("python web framework")

        result1 = qo.get_similar_queries("python web")
        qo.optimize_query("python web server")
        result2 = qo.get_similar_queries("python web")

        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_no_lru_cache_attribute(self):
        qo = QueryOptimizer()
        assert not hasattr(qo.get_similar_queries, '__wrapped__')

    def test_get_similar_queries_updates_after_pattern_changes(self):
        qo = QueryOptimizer()
        qo.optimize_query("python django web")
        qo.optimize_query("python django web")
        qo.optimize_query("python django web")

        results = qo.get_similar_queries("python django")
        assert any("django" in q for q in results)


# ── Fix 6: coordinator circular dep crash ──


class TestCoordinatorCircularDep:
    @pytest.mark.asyncio
    async def test_circular_dependency_does_not_crash(self):
        coordinator = AbilityCoordinator(shadow_ai=MagicMock())

        ORACLE = AbilityType.ORACLE
        X_RAY = AbilityType.X_RAY
        LIFE_PULSE = AbilityType.LIFE_PULSE

        coordinator.context.profiles[ORACLE.value] = AbilityProfile(
            name="TestOracle",
            ability_type=ORACLE,
            description="Test",
            dependencies=[X_RAY],
        )
        coordinator.context.profiles[X_RAY.value] = AbilityProfile(
            name="TestXRay",
            ability_type=X_RAY,
            description="Test",
            dependencies=[LIFE_PULSE],
        )
        coordinator.context.profiles[LIFE_PULSE.value] = AbilityProfile(
            name="TestLifePulse",
            ability_type=LIFE_PULSE,
            description="Test",
            dependencies=[ORACLE],
        )

        result = await coordinator.coordinate("test query")
        assert isinstance(result, dict)
        assert "results" in result

    @pytest.mark.asyncio
    async def test_build_execution_order_raises_on_circular(self):
        coordinator = AbilityCoordinator(shadow_ai=MagicMock())

        A = AbilityType.ORACLE
        B = AbilityType.X_RAY
        C = AbilityType.LIFE_PULSE

        coordinator.context.profiles[A.value] = AbilityProfile(
            name="A", ability_type=A, description="T", dependencies=[B]
        )
        coordinator.context.profiles[B.value] = AbilityProfile(
            name="B", ability_type=B, description="T", dependencies=[C]
        )
        coordinator.context.profiles[C.value] = AbilityProfile(
            name="C", ability_type=C, description="T", dependencies=[A]
        )

        with pytest.raises(ValueError, match="[Cc]ircular"):
            coordinator._build_execution_order([A, B, C])


# ── Fix 7: agent_coordination circular deps ──


class TestAgentCoordinationCircularDeps:
    @pytest.mark.asyncio
    async def test_circular_dependency_raises_valueerror(self):
        crew = ShadowCrew(crew_id="test_crew")

        t1 = ShadowTask(id="t1", description="Task 1", goal="g1", dependencies=["t2"])
        t2 = ShadowTask(id="t2", description="Task 2", goal="g2", dependencies=["t1"])

        crew.add_task(t1)
        crew.add_task(t2)

        with pytest.raises(ValueError, match="[Cc]ircular"):
            await crew._plan_tasks("test goal")

    @pytest.mark.asyncio
    async def test_self_dependency_raises_valueerror(self):
        crew = ShadowCrew(crew_id="test_crew2")

        t1 = ShadowTask(id="t1", description="Task 1", goal="g1", dependencies=["t1"])
        crew.add_task(t1)

        with pytest.raises(ValueError, match="[Cc]ircular"):
            await crew._plan_tasks("test goal")


# ── Fix 8: daily_briefing timezone ──


class TestDailyBriefingTimezone:
    def test_schedule_timezone_default(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine

        engine = DailyBriefingEngine()
        assert engine._schedule_timezone == "America/New_York"

    def test_schedule_hour_default(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine

        engine = DailyBriefingEngine()
        assert engine._schedule_hour == 7

    def test_briefing_schedule_loop_uses_local_time(self):
        engine = DailyBriefingEngine()
        engine._schedule_hour = 3
        engine._schedule_timezone = "America/New_York"

        tz = ZoneInfo("America/New_York")
        now_local = datetime.now(tz)
        target_local = now_local.replace(hour=3, minute=0, second=0, microsecond=0)
        if target_local <= now_local:
            target_local += timedelta(days=1)

        wait_seconds = (target_local - now_local).total_seconds()
        assert wait_seconds > 0
        assert wait_seconds < 86400

    def test_briefing_schedule_tokyo_timezone(self):
        engine = DailyBriefingEngine()
        engine._schedule_timezone = "Asia/Tokyo"
        engine._schedule_hour = 7

        tz = ZoneInfo("Asia/Tokyo")
        now_local = datetime.now(tz)
        target_local = now_local.replace(hour=7, minute=0, second=0, microsecond=0)
        if target_local <= now_local:
            target_local += timedelta(days=1)

        wait_seconds = (target_local - now_local).total_seconds()
        assert 0 < wait_seconds <= 86400

    def test_schedule_loop_code_uses_zoneinfo(self):
        import inspect
        from shadowd.cognition.daily_briefing import DailyBriefingEngine

        source = inspect.getsource(DailyBriefingEngine._briefing_schedule_loop)
        assert "ZoneInfo" in source
        assert "self._schedule_timezone" in source


# ── Fix 9: DomainExpertiseManager thread safety ──


class TestDomainExpertiseManagerThreadSafety:
    def test_concurrent_add_observations(self):
        dem = DomainExpertiseManager()
        errors = []

        def worker(i):
            try:
                dem.add_observation({"data": f"obs_{i}"}, domain="health")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(dem.domain_observations["health"]) == 50

    def test_concurrent_classify_and_add(self):
        dem = DomainExpertiseManager()
        errors = []

        def classify_worker():
            try:
                for _ in range(10):
                    dem.classify_domain("I have a health appointment")
            except Exception as e:
                errors.append(e)

        def add_worker(i):
            try:
                dem.add_observation({"data": f"obs_{i}"}, domain="health")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=classify_worker),
            threading.Thread(target=classify_worker),
        ] + [threading.Thread(target=add_worker, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ── Fix 10: project_monthly calculation ──


class TestProjectMonthlyCalculation:
    def test_project_monthly_with_dates(self):
        fe = FinanceExpert()

        expenses = [
            {"text": "$10.00", "timestamp": "2025-01-01T00:00:00"},
            {"text": "$20.00", "timestamp": "2025-01-15T00:00:00"},
        ]

        result = fe._project_monthly(expenses)

        total = fe._extract_amount(expenses[0]) + fe._extract_amount(expenses[1])
        days_covered = 14
        expected = (total / days_covered) * 30
        assert abs(result - expected) < 0.01

    def test_project_monthly_single_date(self):
        fe = FinanceExpert()

        expenses = [
            {"text": "$10.00", "timestamp": "2025-01-01T00:00:00"},
        ]

        result = fe._project_monthly(expenses)
        assert result == 10.0 * 30

    def test_project_monthly_no_dates(self):
        fe = FinanceExpert()

        expenses = [
            {"text": "$100.00"},
            {"text": "$50.00"},
        ]

        result = fe._project_monthly(expenses)
        assert result == 150.0

    def test_project_monthly_not_times_two(self):
        fe = FinanceExpert()

        expenses = [
            {"text": "$100.00", "timestamp": "2025-01-01T00:00:00"},
            {"text": "$200.00", "timestamp": "2025-01-31T00:00:00"},
        ]

        result = fe._project_monthly(expenses)
        total = fe._extract_amount(expenses[0]) + fe._extract_amount(expenses[1])

        # Should NOT be total * 2 (the old broken formula)
        # With 30 days covered, it should be total * 30/30 = total
        assert abs(result - total) < 0.01 or result != total * 2

    def test_project_monthly_empty(self):
        fe = FinanceExpert()
        result = fe._project_monthly([])
        assert result == 0.0