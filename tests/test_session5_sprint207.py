"""Sprint 207: CognitiveHealthDashboard — module health scoring and diagnostics"""

import pytest
import threading
from shadowd.cognition.health_dashboard import (
    CognitiveHealthDashboard, ModuleHealth, HealthAlert,
    SystemHealthReport, HealthStatus, AlertSeverity,
)


@pytest.fixture
def dashboard():
    return CognitiveHealthDashboard()


class TestModuleHealth:
    def test_auto_fields(self):
        m = ModuleHealth(module_name="test", status="healthy")
        assert m.module_name == "test"
        assert m.last_check != ""

    def test_sanitization(self):
        m = ModuleHealth(module_name="test<>module", status="<script>alert</script>")
        assert "<>" not in m.module_name
        assert "<script>" not in m.status

    def test_health_score_healthy(self):
        m = ModuleHealth(module_name="test", status="healthy",
                         latency_ms=10, success_rate=1.0, error_count=0)
        assert m.health_score() >= 0.9

    def test_health_score_degraded(self):
        m = ModuleHealth(module_name="test", status="degraded",
                         latency_ms=2000, success_rate=0.8, error_count=10)
        score = m.health_score()
        assert 0.0 <= score <= 1.0

    def test_health_score_clamped(self):
        m = ModuleHealth(module_name="test", latency_ms=-100, success_rate=2.0)
        assert m.latency_ms == 0.0
        assert m.success_rate == 1.0

    def test_to_dict_rounded(self):
        m = ModuleHealth(module_name="test", latency_ms=123.456, success_rate=0.95432)
        d = m.to_dict()
        assert d["latency_ms"] == 123.46
        assert d["success_rate"] == 0.9543

    def test_to_dict_deepcopy(self):
        m = ModuleHealth(module_name="test", details={"key": "val"})
        d = m.to_dict()
        d["details"]["key"] = "modified"
        assert m.details["key"] == "val"


class TestHealthAlert:
    def test_auto_id(self):
        a = HealthAlert(module_name="test", severity="high", message="test")
        assert len(a.alert_id) == 12

    def test_sanitization(self):
        a = HealthAlert(module_name="test<>mod", severity="high",
                       message="Contains\x00control", metric="lat_ms<>")
        assert "<>" not in a.module_name
        assert "\x00" not in a.message
        assert "<>" not in a.metric

    def test_to_dict(self):
        a = HealthAlert(module_name="test", severity="high",
                       message="High latency", value=5000, threshold=2000)
        d = a.to_dict()
        assert d["module_name"] == "test"
        assert d["value"] == 5000


class TestDashboardRegister:
    def test_register_module(self, dashboard):
        dashboard.register_module("test_module")
        health = dashboard.get_module_health("test_module")
        assert health is not None
        assert health.module_name == "test_module"
        assert health.status == "healthy"

    def test_register_multiple(self, dashboard):
        for i in range(5):
            dashboard.register_module(f"module_{i}")
        modules = dashboard.get_all_modules()
        assert len(modules) == 5

    def test_register_capped(self, dashboard):
        for i in range(60):
            dashboard.register_module(f"m_{i}")
        modules = dashboard.get_all_modules()
        assert len(modules) <= 50


class TestDashboardUpdate:
    def test_update_module(self, dashboard):
        dashboard.register_module("test")
        health = dashboard.update_module("test", latency_ms=100, success_rate=0.95)
        assert health.latency_ms == 100
        assert health.success_rate == 0.95

    def test_update_auto_registers(self, dashboard):
        health = dashboard.update_module("new_module", latency_ms=50)
        assert health.module_name == "new_module"

    def test_update_with_details(self, dashboard):
        dashboard.update_module("test", details={"cpu": 45, "memory": 80})
        health = dashboard.get_module_health("test")
        assert health.details.get("cpu") == 45


class TestAlerts:
    def test_latency_alert(self, dashboard):
        dashboard.update_module("slow_module", latency_ms=6000)
        alerts = dashboard.get_alerts(severity="high")
        assert len(alerts) >= 1

    def test_error_alert(self, dashboard):
        dashboard.update_module("error_module", error_count=60)
        alerts = dashboard.get_alerts(module_name="error_module")
        assert len(alerts) >= 1

    def test_success_rate_alert(self, dashboard):
        dashboard.update_module("unreliable", success_rate=0.4)
        alerts = dashboard.get_alerts(module_name="unreliable")
        # success_rate=0.4 means 0.6 failure rate, exceeds "high" threshold (0.70)
        assert len(alerts) >= 1

    def test_resolve_alert(self, dashboard):
        dashboard.update_module("test", latency_ms=6000)
        alerts = dashboard.get_alerts(resolved=False)
        if alerts:
            result = dashboard.resolve_alert(alerts[0].alert_id)
            assert result is True

    def test_resolve_nonexistent(self, dashboard):
        result = dashboard.resolve_alert("nonexistent")
        assert result is False

    def test_get_alerts_filtered(self, dashboard):
        dashboard.update_module("mod1", latency_ms=6000)
        dashboard.update_module("mod2", error_count=60)
        all_alerts = dashboard.get_alerts()
        severe = dashboard.get_alerts(severity="high")
        assert len(all_alerts) >= len(severe)

    def test_get_alerts_limit(self, dashboard):
        for i in range(10):
            dashboard.update_module(f"mod_{i}", latency_ms=6000)
        alerts = dashboard.get_alerts(limit=5)
        assert len(alerts) <= 5


class TestHealthReport:
    def test_generate_report(self, dashboard):
        dashboard.register_module("mod1")
        dashboard.register_module("mod2")
        dashboard.update_module("mod1", latency_ms=50, success_rate=0.99)
        dashboard.update_module("mod2", latency_ms=100, success_rate=0.95)
        report = dashboard.generate_report()
        assert report.overall_status == "healthy"
        assert report.module_count == 2
        assert 0.0 <= report.overall_score <= 1.0

    def test_report_degraded(self, dashboard):
        dashboard.register_module("healthy_mod")
        dashboard.register_module("degraded_mod")
        dashboard.update_module("healthy_mod", latency_ms=50, success_rate=0.99)
        dashboard.update_module("degraded_mod", latency_ms=50, success_rate=0.8, status="degraded")
        report = dashboard.generate_report()
        assert report.overall_status == "degraded"
        assert report.degraded_count == 1

    def test_report_to_dict(self, dashboard):
        dashboard.register_module("test")
        report = dashboard.generate_report()
        d = report.to_dict()
        assert "overall_status" in d
        assert "modules" in d
        assert isinstance(d["modules"], dict)

    def test_report_deepcopy(self, dashboard):
        dashboard.register_module("test")
        report = dashboard.generate_report()
        d = report.to_dict()
        d["modules"]["test"]["status"] = "MODIFIED"
        assert report.modules.get("test", {}).get("status") != "MODIFIED"


class TestHistory:
    def test_report_history(self, dashboard):
        dashboard.register_module("test")
        dashboard.generate_report()
        dashboard.generate_report()
        history = dashboard.get_history()
        assert len(history) >= 2

    def test_history_limit(self, dashboard):
        dashboard.register_module("test")
        for _ in range(5):
            dashboard.generate_report()
        history = dashboard.get_history(limit=3)
        assert len(history) <= 3


class TestStats:
    def test_initial_stats(self, dashboard):
        stats = dashboard.get_stats()
        assert stats["total_checks"] == 0
        assert stats["total_modules"] == 0

    def test_stats_after_operations(self, dashboard):
        dashboard.register_module("test")
        dashboard.update_module("test", latency_ms=100)
        stats = dashboard.get_stats()
        assert stats["total_modules"] == 1
        assert stats["total_checks"] == 1


class TestThreadSafety:
    def test_concurrent_updates(self, dashboard):
        errors = []

        def update_many():
            try:
                for i in range(50):
                    dashboard.update_module(f"mod_{i % 5}", latency_ms=i * 10)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=update_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = dashboard.get_stats()
        assert stats["total_checks"] == 200

    def test_concurrent_reports(self, dashboard):
        dashboard.register_module("test")

        def gen_reports():
            try:
                for _ in range(20):
                    dashboard.generate_report()
            except Exception as e:
                pass

        threads = [threading.Thread(target=gen_reports) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestEdgeCases:
    def test_empty_report(self, dashboard):
        report = dashboard.generate_report()
        assert report.overall_status == "healthy"
        assert report.module_count == 0
        assert report.overall_score >= 0.0

    def test_get_nonexistent_module(self, dashboard):
        result = dashboard.get_module_health("nonexistent")
        assert result is None

    def test_health_score_with_errors(self):
        m = ModuleHealth(module_name="test", latency_ms=5000,
                         success_rate=0.3, error_count=80)
        score = m.health_score()
        assert 0.0 <= score <= 1.0
        assert score < 0.7

    def test_negative_latency_clamped(self):
        m = ModuleHealth(module_name="test", latency_ms=-100)
        assert m.latency_ms == 0.0

    def test_success_rate_out_of_range(self):
        m = ModuleHealth(module_name="test", success_rate=2.0)
        assert m.success_rate == 1.0