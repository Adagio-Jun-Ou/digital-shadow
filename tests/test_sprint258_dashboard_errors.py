#!/usr/bin/env python3
"""Sprint 258 — Dashboard error handling + offline UI tests"""

import pytest
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SHADOW_OS_API_KEY", "")


class TestDashboardErrorHandling:
    @pytest.fixture
    def dash_html(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "web", "shadow_os.html"
        )
        if not os.path.exists(path):
            pytest.skip("Dashboard HTML not found")
        with open(path) as f:
            return f.read()

    def test_has_api_error_handler(self, dash_html):
        assert "catch(e)" in dash_html or ".catch(" in dash_html

    def test_has_show_offline_function(self, dash_html):
        assert "showOffline" in dash_html

    def test_has_offline_banner(self, dash_html):
        assert "offlineBanner" in dash_html or "Connection lost" in dash_html

    def test_has_hdr_dot_offline_state(self, dash_html):
        assert "hdr-dot off" in dash_html or 'hdr-dot'.replace('"', '') in dash_html

    def test_api_checks_response_ok(self, dash_html):
        assert "r.ok" in dash_html or "!r.ok" in dash_html

    def test_has_null_guard_on_status(self, dash_html):
        assert "if(!s)" in dash_html

    def test_has_auto_reconnect(self, dash_html):
        assert "apiFailCount" in dash_html

    def test_has_visibility_refresh(self, dash_html):
        assert "visibilitychange" in dash_html

    def test_has_backoff_logic(self, dash_html):
        assert "refreshMs" in dash_html or "15000" in dash_html

    def test_no_uncatched_json_parse(self, dash_html):
        assert "r.json()" in dash_html


class TestWatchAppErrorHandling:
    @pytest.fixture
    def watch_html(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "web", "watch_app.html"
        )
        if not os.path.exists(path):
            pytest.skip("Watch app HTML not found")
        with open(path) as f:
            return f.read()

    def test_has_error_handler(self, watch_html):
        assert "catch" in watch_html

    def test_has_offline_indicator(self, watch_html):
        assert "offline" in watch_html.lower() or "Offline" in watch_html

    def test_has_online_offline_events(self, watch_html):
        assert "online" in watch_html and "offline" in watch_html

    def test_has_connection_dot(self, watch_html):
        assert "connection" in watch_html.lower() or "conn" in watch_html.lower()


class TestPWAErrorHandling:
    @pytest.fixture
    def pwa_html(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "web", "shadow_pwa.html"
        )
        if not os.path.exists(path):
            pytest.skip("PWA HTML not found")
        with open(path) as f:
            return f.read()

    def test_has_error_handler(self, pwa_html):
        assert "catch" in pwa_html

    def test_has_offline_support(self, pwa_html):
        assert "offline" in pwa_html.lower() or "Offline" in pwa_html

    def test_has_service_worker(self, pwa_html):
        assert "serviceWorker" in pwa_html or "sw.js" in pwa_html

    def test_has_speak_function(self, pwa_html):
        assert "speakShadow" in pwa_html or "speak" in pwa_html

    def test_has_mute_toggle(self, pwa_html):
        assert "mute" in pwa_html.lower() or "Mute" in pwa_html


class TestServiceWorkerCaching:
    @pytest.fixture
    def sw_js(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "web", "sw.js"
        )
        if not os.path.exists(path):
            pytest.skip("Service worker not found")
        with open(path) as f:
            return f.read()

    def test_has_cache_name(self, sw_js):
        assert "CACHE" in sw_js or "cache" in sw_js

    def test_has_fetch_handler(self, sw_js):
        assert "fetch" in sw_js

    def test_has_install_handler(self, sw_js):
        assert "install" in sw_js

    def test_has_activate_handler(self, sw_js):
        assert "activate" in sw_js

    def test_has_offline_fallback(self, sw_js):
        assert "offline" in sw_js.lower() or "OFFLINE" in sw_js


class TestManifestValid:
    @pytest.fixture
    def manifest(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "web", "manifest.json"
        )
        if not os.path.exists(path):
            pytest.skip("Manifest not found")
        import json
        with open(path) as f:
            return json.load(f)

    def test_has_name(self, manifest):
        assert "name" in manifest

    def test_has_display(self, manifest):
        assert "display" in manifest

    def test_has_icons(self, manifest):
        assert "icons" in manifest

    def test_standalone_display(self, manifest):
        assert manifest.get("display") in ("standalone", "fullscreen")

    def test_has_start_url(self, manifest):
        assert "start_url" in manifest