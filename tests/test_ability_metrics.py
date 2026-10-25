"""Tests for the per-ability metrics tracker."""

import pytest

from shadowd import ability_metrics


@pytest.fixture(autouse=True)
def _isolated():
    """Wipe metrics before and after each test so order-of-tests doesn't matter."""
    ability_metrics.reset()
    yield
    ability_metrics.reset()


class TestRecording:
    def test_starts_empty(self):
        assert ability_metrics.snapshot() == {}

    def test_records_one_invocation(self):
        ability_metrics.record("oracle", True, 0.5)
        snap = ability_metrics.snapshot()
        assert snap["oracle"]["count"] == 1
        assert snap["oracle"]["successes"] == 1
        assert snap["oracle"]["failures"] == 0
        assert snap["oracle"]["p50_latency_s"] == 0.5

    def test_separates_success_and_failure(self):
        ability_metrics.record("oracle", True, 0.1)
        ability_metrics.record("oracle", False, 0.2)
        ability_metrics.record("oracle", True, 0.3)
        snap = ability_metrics.snapshot()
        assert snap["oracle"]["count"] == 3
        assert snap["oracle"]["successes"] == 2
        assert snap["oracle"]["failures"] == 1

    def test_keeps_separate_per_ability(self):
        ability_metrics.record("oracle", True, 0.1)
        ability_metrics.record("x_ray", True, 0.2)
        snap = ability_metrics.snapshot()
        assert set(snap.keys()) == {"oracle", "x_ray"}


class TestLatencyPercentiles:
    def test_p95_with_skewed_distribution(self):
        # 100 fast + 5 slow → p95 should be near slow.
        for _ in range(95):
            ability_metrics.record("a", True, 0.01)
        for _ in range(5):
            ability_metrics.record("a", True, 1.0)
        snap = ability_metrics.snapshot()
        # p95 in sorted ascending = sample at index 95 → 1.0
        assert snap["a"]["p95_latency_s"] >= 0.5

    def test_window_caps_at_100(self):
        for i in range(150):
            ability_metrics.record("a", True, float(i))
        snap = ability_metrics.snapshot()
        # Only the last 100 samples are kept.
        assert snap["a"]["sample_count"] == 100
        # The total count keeps incrementing.
        assert snap["a"]["count"] == 150


class TestPrometheusOutput:
    def test_emits_counter_lines(self):
        ability_metrics.record("oracle", True, 0.1)
        ability_metrics.record("oracle", False, 0.2)
        lines = ability_metrics.prometheus_lines()
        joined = "\n".join(lines)
        assert "shadow_ability_invocations" in joined
        assert 'ability="oracle"' in joined
        assert "shadow_ability_successes" in joined
        assert "shadow_ability_failures" in joined
        assert "shadow_ability_latency_p95" in joined

    def test_empty_state(self):
        lines = ability_metrics.prometheus_lines()
        # HELP / TYPE lines should still appear even when no abilities recorded.
        assert any("# HELP" in line for line in lines)


class TestThreadSafety:
    def test_concurrent_record(self):
        import threading
        def worker():
            for _ in range(50):
                ability_metrics.record("oracle", True, 0.01)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        snap = ability_metrics.snapshot()
        assert snap["oracle"]["count"] == 200
