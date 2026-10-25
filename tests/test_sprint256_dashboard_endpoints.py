#!/usr/bin/env python3
"""Sprint 256 Tests — Missing Dashboard API Endpoints + Compatibility"""

import pytest
import sys
import os
import json
import asyncio
import copy
import threading
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SHADOW_OS_API_KEY", "")

from shadowd.cognition.api_router import router

ROUTER_PATHS = [r.path for r in router.routes if hasattr(r, "methods")]


def _has_route(path):
    full = f"/os{path}" if not path.startswith("/os") else path
    return full in ROUTER_PATHS


def _get_route(path):
    full = f"/os{path}" if not path.startswith("/os") else path
    for r in router.routes:
        if hasattr(r, "methods") and r.path == full:
            return r
    return None


class TestBriefingGenerateEndpoint:
    def test_briefing_generate_route_exists(self):
        assert _has_route("/briefing/generate")

    def test_briefing_generate_has_post_method(self):
        r = _get_route("/briefing/generate")
        assert r is not None
        assert "POST" in r.methods

    def test_briefing_generate_has_auth(self):
        r = _get_route("/briefing/generate")
        assert r is not None
        assert len(r.dependencies) > 0


class TestAriseListEndpoint:
    def test_arise_list_route_exists(self):
        assert _has_route("/arise/list")

    def test_arise_list_has_get_method(self):
        r = _get_route("/arise/list")
        assert r is not None
        assert "GET" in r.methods

    def test_arise_list_has_auth(self):
        r = _get_route("/arise/list")
        assert r is not None
        assert len(r.dependencies) > 0


class TestDeviceSummaryEndpoint:
    def test_device_summary_route_exists(self):
        assert _has_route("/device/summary")

    def test_device_summary_has_get_method(self):
        r = _get_route("/device/summary")
        assert r is not None
        assert "GET" in r.methods

    def test_device_summary_has_auth(self):
        r = _get_route("/device/summary")
        assert r is not None
        assert len(r.dependencies) > 0


class TestEndpointFunctionality:
    def test_briefing_generate_endpoint_callable(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine

        engine = DailyBriefingEngine()
        briefing = asyncio.run(engine.generate_briefing())
        assert briefing is not None
        assert hasattr(briefing, "to_dict")
        d = briefing.to_dict()
        assert isinstance(d, dict)
        assert "items" in d

    def test_arise_list_returns_agents(self):
        from shadowd.cognition.shadow_arise import ShadowArise

        arise = ShadowArise()
        result = arise.list_agents(limit=20)
        assert isinstance(result, list)

    def test_arise_list_with_agent(self):
        from shadowd.cognition.shadow_arise import ShadowArise

        arise = ShadowArise()
        arise.arise(
            name="TestAgent",
            objective="Test objective",
            priority="medium",
        )
        result = arise.list_agents(limit=20)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_device_summary_returns_dict(self):
        from shadowd.cognition.device_context import DeviceContextEngine

        ctx = DeviceContextEngine()
        result = ctx.get_device_summary()
        assert isinstance(result, dict)

    def test_device_devices_endpoint_still_works(self):
        assert _has_route("/device/devices")

    def test_all_dashboard_endpoints_exist(self):
        dashboard_endpoints = [
            "/status",
            "/health",
            "/device/context",
            "/device/summary",
            "/device/digital-wellbeing",
            "/device/health-trends",
            "/personality",
            "/personality/patterns",
            "/personality/predict",
            "/memory/stats",
            "/memory/blocks",
            "/memory/search",
            "/kg/query",
            "/crew/agents",
            "/crew/mission",
            "/arise/list",
            "/briefing/generate",
            "/stealth/stats",
            "/stealth/audit",
            "/mesh/status",
            "/mesh/peers",
            "/voice/status",
            "/voice/process",
        ]
        for ep in dashboard_endpoints:
            assert _has_route(ep), f"Dashboard endpoint {ep} missing"

    def test_briefing_to_dict_structure(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine

        engine = DailyBriefingEngine()
        briefing = asyncio.run(engine.generate_briefing())
        d = briefing.to_dict()
        assert "briefing_id" in d or "id" in d
        assert "wellness_score" in d
        assert "items" in d


class TestDashboardHTMLReferences:
    def test_dashboard_api_endpoints_exist(self):
        dash_path = os.path.join(
            os.path.dirname(__file__), "..", "web", "shadow_os.html"
        )
        if not os.path.exists(dash_path):
            pytest.skip("Dashboard HTML not found")

        with open(dash_path) as f:
            content = f.read()

        api_calls = re.findall(r"api\('([^']+)'", content)
        missing = []
        for call in api_calls:
            endpoint = "/" + call.split("?")[0]
            endpoint = endpoint.rstrip("/")
            if endpoint.startswith("//"):
                endpoint = endpoint[1:]
            if endpoint == "/":
                continue
            if not _has_route(endpoint):
                missing.append(endpoint)

        assert not missing, f"Dashboard references missing endpoints: {missing}"


class TestRouteConsistency:
    def test_device_summary_and_devices_both_exist(self):
        assert _has_route("/device/summary")
        assert _has_route("/device/devices")

    def test_briefing_generate_post_exists(self):
        r = _get_route("/briefing/generate")
        assert r is not None
        assert "POST" in r.methods

    def test_all_new_endpoints_have_auth_dependency(self):
        new_endpoints = ["/briefing/generate", "/arise/list", "/device/summary"]
        for ep in new_endpoints:
            r = _get_route(ep)
            assert r is not None, f"{ep} not found"
            assert len(r.dependencies) > 0, f"{ep} missing auth dependency"

    def test_method_uniqueness_no_same_method_twice(self):
        method_paths = {}
        for r in router.routes:
            if hasattr(r, "methods"):
                for m in r.methods:
                    key = (m, r.path)
                    if key in method_paths:
                        pytest.fail(f"Duplicate {m} {r.path}")
                    method_paths[key] = True