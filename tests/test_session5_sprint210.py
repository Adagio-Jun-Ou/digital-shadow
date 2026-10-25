#!/usr/bin/env python3
"""Sprint 210 tests: Health Dashboard auto-update from module get_stats()."""

import copy
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from shadowd.cognition.health_dashboard import (
    CognitiveHealthDashboard, ModuleHealth, HealthAlert,
    SystemHealthReport, HealthStatus, AlertSeverity,
)
from shadowd.cognition.social_graph_propagation import SocialGraphPropagation


class FakeModule:
    def __init__(self, latency=10.0, errors=0, total=100, successes=95):
        self._latency = latency
        self._errors = errors
        self._total = total
        self._successes = successes

    def get_stats(self):
        return {
            "avg_latency_ms": self._latency,
            "error_count": self._errors,
            "total_requests": self._total,
            "successful_requests": self._successes,
        }


class BrokenModule:
    def get_stats(self):
        raise RuntimeError("I am broken")


class NoStatsModule:
    pass


class TestAutoUpdateBasics(unittest.TestCase):
    def setUp(self):
        self.hd = CognitiveHealthDashboard()

    def test_auto_update_registers_modules(self):
        modules = {"mod_a": FakeModule(), "mod_b": FakeModule()}
        result = self.hd.auto_update(modules)
        self.assertEqual(result["modules_updated"], 2)
        self.assertEqual(result["modules_total"], 2)

    def test_auto_update_sets_latency(self):
        modules = {"slow_mod": FakeModule(latency=3000.0)}
        self.hd.auto_update(modules)
        health = self.hd.get_module_health("slow_mod")
        self.assertAlmostEqual(health.latency_ms, 3000.0)

    def test_auto_update_calculates_success_rate(self):
        modules = {"good_mod": FakeModule(total=100, successes=90)}
        self.hd.auto_update(modules)
        health = self.hd.get_module_health("good_mod")
        self.assertAlmostEqual(health.success_rate, 0.9)

    def test_auto_update_error_count(self):
        modules = {"err_mod": FakeModule(errors=5)}
        self.hd.auto_update(modules)
        health = self.hd.get_module_health("err_mod")
        self.assertEqual(health.error_count, 5)

    def test_auto_update_stores_details(self):
        modules = {"detail_mod": FakeModule()}
        self.hd.auto_update(modules)
        health = self.hd.get_module_health("detail_mod")
        self.assertIn("avg_latency_ms", health.details)

    def test_auto_update_handles_empty_map(self):
        result = self.hd.auto_update({})
        self.assertEqual(result["modules_updated"], 0)

    def test_auto_update_handles_non_dict_stats(self):
        mod = MagicMock()
        mod.get_stats.return_value = "not a dict"
        result = self.hd.auto_update({"bad": mod})
        self.assertEqual(result["modules_updated"], 0)

    def test_auto_update_handles_broken_module(self):
        modules = {"broken": BrokenModule()}
        result = self.hd.auto_update(modules)
        self.assertEqual(result["modules_updated"], 0)
        health = self.hd.get_module_health("broken")
        self.assertIsNotNone(health)
        self.assertEqual(health.status, "offline")


class TestAutoUpdateAlerts(unittest.TestCase):
    def setUp(self):
        self.hd = CognitiveHealthDashboard()

    def test_high_latency_triggers_alert(self):
        modules = {"slow": FakeModule(latency=6000.0)}
        result = self.hd.auto_update(modules)
        self.assertGreater(result["new_alerts"], 0)
        alerts = self.hd.get_alerts(module_name="slow", resolved=False)
        self.assertGreater(len(alerts), 0)

    def test_low_errors_no_alert(self):
        modules = {"ok": FakeModule(errors=1, latency=10.0, total=100, successes=99)}
        result = self.hd.auto_update(modules)
        self.assertEqual(result["new_alerts"], 0)

    def test_high_errors_triggers_alert(self):
        modules = {"error_prone": FakeModule(errors=60)}
        result = self.hd.auto_update(modules)
        self.assertGreater(result["new_alerts"], 0)


class TestAutoUpdateSuccessRateVariants(unittest.TestCase):
    def setUp(self):
        self.hd = CognitiveHealthDashboard()

    def test_success_rate_from_total_decisions(self):
        mod = MagicMock()
        mod.get_stats.return_value = {"total_decisions": 50, "decisions_made": 45, "avg_latency_ms": 10.0}
        self.hd.auto_update({"dec_mod": mod})
        health = self.hd.get_module_health("dec_mod")
        self.assertAlmostEqual(health.success_rate, 0.9)

    def test_success_rate_from_total_checks(self):
        mod = MagicMock()
        mod.get_stats.return_value = {"total_checks": 200, "successful_requests": 190, "avg_latency_ms": 5.0}
        self.hd.auto_update({"chk_mod": mod})
        health = self.hd.get_module_health("chk_mod")
        self.assertAlmostEqual(health.success_rate, 0.95)

    def test_no_total_defaults_to_1(self):
        mod = MagicMock()
        mod.get_stats.return_value = {"avg_latency_ms": 10.0}
        self.hd.auto_update({"notrack_mod": mod})
        health = self.hd.get_module_health("notrack_mod")
        self.assertEqual(health.success_rate, 1.0)


class TestAutoUpdateConcurrent(unittest.TestCase):
    def test_concurrent_auto_update(self):
        hd = CognitiveHealthDashboard()
        modules = {"mod_%d" % i: FakeModule(latency=float(i * 10)) for i in range(20)}
        errors = []

        def do_update():
            try:
                hd.auto_update(modules)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=do_update) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        all_mods = hd.get_all_modules()
        self.assertGreater(len(all_mods), 0)


class TestAutoUpdateSanitization(unittest.TestCase):
    def setUp(self):
        self.hd = CognitiveHealthDashboard()

    def test_module_name_sanitized(self):
        mod = MagicMock()
        mod.get_stats.return_value = {"avg_latency_ms": 10.0}
        self.hd.auto_update({"<script>evil</script>": mod})
        all_mods = self.hd.get_all_modules()
        for m in all_mods:
            self.assertNotIn("<", m.module_name)


class TestAutoUpdateWithRealModules(unittest.TestCase):
    def test_auto_update_social_propagation(self):
        hd = CognitiveHealthDashboard()
        sgp = SocialGraphPropagation()
        sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test")
        result = hd.auto_update({"social_propagation": sgp})
        self.assertEqual(result["modules_updated"], 1)
        health = hd.get_module_health("social_propagation")
        self.assertIsNotNone(health)


class TestAutoUpdateIdempotent(unittest.TestCase):
    def test_second_update_overwrites(self):
        hd = CognitiveHealthDashboard()
        mod1 = FakeModule(latency=100.0)
        mod2 = FakeModule(latency=500.0)
        hd.auto_update({"mod_a": mod1})
        health1 = hd.get_module_health("mod_a")
        hd.auto_update({"mod_a": mod2})
        health2 = hd.get_module_health("mod_a")
        self.assertAlmostEqual(health2.latency_ms, 500.0)
        self.assertNotAlmostEqual(health1.latency_ms, health2.latency_ms)


class TestAutoUpdateReport(unittest.TestCase):
    def test_generate_report_after_auto_update(self):
        hd = CognitiveHealthDashboard()
        modules = {
            "alpha": FakeModule(latency=50.0, total=100, successes=99),
            "beta": FakeModule(latency=200.0, total=100, successes=95),
        }
        hd.auto_update(modules)
        report = hd.generate_report()
        self.assertEqual(report.module_count, 2)
        self.assertIn("alpha", report.modules)
        self.assertIn("beta", report.modules)

    def test_report_to_dict(self):
        hd = CognitiveHealthDashboard()
        hd.auto_update({"mod": FakeModule()})
        report = hd.generate_report()
        d = report.to_dict()
        self.assertIn("overall_score", d)
        self.assertIn("modules", d)
        self.assertIsInstance(d["modules"], dict)


class TestAutoUpdateNoStatsMethod(unittest.TestCase):
    def test_module_without_get_stats(self):
        hd = CognitiveHealthDashboard()
        mod = NoStatsModule()
        # auto_update should handle AttributeError gracefully
        result = hd.auto_update({"no_method": mod})
        self.assertEqual(result["modules_updated"], 0)
        health = hd.get_module_health("no_method")
        self.assertIsNotNone(health)
        self.assertEqual(health.status, "offline")


class TestAutoUpdateIncrementalErrors(unittest.TestCase):
    def test_broken_module_increments_error_count(self):
        hd = CognitiveHealthDashboard()
        modules = {"broken": BrokenModule()}
        hd.auto_update(modules)
        hd.auto_update(modules)
        health = hd.get_module_health("broken")
        self.assertEqual(health.error_count, 2)


if __name__ == "__main__":
    unittest.main()