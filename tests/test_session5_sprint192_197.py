"""Sprint 192-197 regression tests: round(), except narrowing, timezone, log injection, _stats_lock."""
import pytest
import copy
import threading
import sys
import os
import re
import inspect


class TestSprint192Round:
    def test_ambient_monitor_pattern_to_dict_rounded(self):
        from shadowd.ambient_monitor import Pattern
        from datetime import datetime, timezone
        p = Pattern(pattern_type="test", description="test", occurrences=1,
                    confidence=0.123456789,
                    first_seen=datetime.now(timezone.utc),
                    last_seen=datetime.now(timezone.utc))
        d = p.to_dict()
        assert d["confidence"] == round(0.123456789, 3)

    def test_ambient_monitor_insight_to_dict_rounded(self):
        from shadowd.ambient_monitor import Insight
        i = Insight(insight_type="test", priority="medium", title="t", message="m", confidence=0.987654321)
        d = i.to_dict()
        assert d["confidence"] == round(0.987654321, 3)

    def test_super_orchestrator_stats_rounded(self):
        from shadowd.super_orchestrator import SuperShadowStats
        s = SuperShadowStats()
        s.cache_hits = 33
        s.cache_misses = 67
        d = s.to_dict()
        assert d["cache_hit_rate"] == round(33 / 100, 4)
        assert isinstance(d["uptime_seconds"], (int, float))


class TestSprint194ExceptNarrowing:
    def test_ambient_monitor_load_narrowed(self):
        from shadowd.ambient_monitor import AmbientMonitor
        source = inspect.getsource(AmbientMonitor._load_state)
        assert "(json.JSONDecodeError, KeyError, TypeError, ValueError, OSError)" in source

    def test_ambient_monitor_save_narrowed(self):
        from shadowd.ambient_monitor import AmbientMonitor
        source = inspect.getsource(AmbientMonitor._save_state)
        assert "except (OSError, TypeError, ValueError, json.JSONDecodeError)" in source

    def test_proactive_insights_rule_narrowed(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        source = inspect.getsource(ProactiveInsightEngine.evaluate_rules)
        assert "except (ValueError, TypeError, AttributeError, RuntimeError)" in source


class TestSprint195Timezone:
    def test_ambient_monitor_commitment_utc(self):
        from shadowd.ambient_monitor import Commitment
        c = Commitment(id="test-1", content="test commit")
        assert c.created_at.tzinfo is not None

    def test_ambient_monitor_insight_utc(self):
        from shadowd.ambient_monitor import Insight
        i = Insight(insight_type="test", priority="medium", title="t", message="m")
        assert i.created_at.tzinfo is not None


class TestSprint196LogInjection:
    def test_shikigami_blocked_command_uses_percent_format(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry)
        assert 'logger.warning("Blocked command: %s"' in source

    def test_performance_optimization_cache_hit_uses_percent_format(self):
        from shadowd.performance_optimization import PerformanceOptimizedShadowAI
        source = inspect.getsource(PerformanceOptimizedShadowAI.optimized_ask)
        assert 'logger.info("Cache hit for: %s"' in source


class TestSprint197StatsLock:
    def test_event_bus_has_stats_lock(self):
        from shadowd.event_bus import EventBus
        eb = EventBus()
        assert hasattr(eb, '_stats_lock')
        assert isinstance(eb._stats_lock, type(threading.Lock()))

    def test_event_bus_get_stats_thread_safe(self):
        from shadowd.event_bus import EventBus
        eb = EventBus()
        stats = eb.get_stats()
        assert stats["events_published"] == 0
        stats["events_published"] = 999
        assert eb.get_stats()["events_published"] == 0

    def test_prompt_cache_get_stats_deepcopy(self):
        from shadowd.cache.prompt_cache import PromptCache
        pc = PromptCache()
        stats = pc.get_stats()
        assert stats.hits == 0
        original_hits = pc.get_stats().hits
        stats.hits = 999
        assert pc.get_stats().hits == original_hits


class TestSprint191WordBoundary:
    def test_domain_expertise_project_word_boundary(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "projection analysis complete"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["projects"] == 0

    def test_domain_expertise_meeting_word_boundary(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "greeting card sent"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["meetings"] == 0

    def test_stealth_layer_name_not_matching_rename(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        result = ce.anonymize({"rename_count": 5})
        assert result["rename_count"] == 5

    def test_device_context_home_location(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "home office"})
        assert result == LocationType.HOME