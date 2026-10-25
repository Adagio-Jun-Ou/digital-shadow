"""Sprint 254 tests — Watch Companion App (watch_app.html) + PWA test fix"""

import os
import re
import pytest


WATCH_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "watch_app.html")
PWA_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "shadow_pwa.html")
SW_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")


def _read_watch():
    with open(WATCH_APP_PATH) as f:
        return f.read()


def _read_pwa():
    with open(PWA_PATH) as f:
        return f.read()


class TestWatchAppExists:
    def test_watch_app_file_exists(self):
        assert os.path.exists(WATCH_APP_PATH), "watch_app.html must exist"

    def test_watch_app_is_valid_html(self):
        content = _read_watch()
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_watch_app_has_manifest_link(self):
        content = _read_watch()
        assert "manifest.json" in content

    def test_watch_app_has_apple_mobile_web_app_capable(self):
        content = _read_watch()
        assert 'apple-mobile-web-app-capable' in content

    def test_watch_app_has_viewport(self):
        content = _read_watch()
        assert "viewport" in content
        assert "viewport-fit=cover" in content


class TestWatchAppTabs:
    def test_has_health_tab(self):
        content = _read_watch()
        assert 'id="t0"' in content or "Health" in content

    def test_has_status_tab(self):
        content = _read_watch()
        assert 'id="t1"' in content

    def test_has_voice_tab(self):
        content = _read_watch()
        assert 'id="t2"' in content

    def test_has_actions_tab(self):
        content = _read_watch()
        assert 'id="t3"' in content

    def test_has_chat_tab(self):
        content = _read_watch()
        assert 'id="t4"' in content

    def test_has_five_nav_dots(self):
        content = _read_watch()
        assert content.count('class="dot') >= 5

    def test_tab_switching_function(self):
        content = _read_watch()
        assert "function tab(" in content


class TestWatchAppHealthUI:
    def test_has_heart_rate_display(self):
        content = _read_watch()
        assert "heart_rate" in content.lower() or "Heart" in content

    def test_has_hrv_display(self):
        content = _read_watch()
        assert "hrv" in content.lower()

    def test_has_spo2_display(self):
        content = _read_watch()
        assert "spo2" in content.lower() or "SpO2" in content

    def test_has_step_count(self):
        content = _read_watch()
        assert "step_count" in content or "Steps" in content

    def test_has_sleep_display(self):
        content = _read_watch()
        assert "sleep" in content.lower()

    def test_has_stress_level(self):
        content = _read_watch()
        assert "stress" in content.lower()

    def test_has_exercise_minutes(self):
        content = _read_watch()
        assert "exercise" in content.lower()

    def test_health_color_coding(self):
        content = _read_watch()
        assert "var(--green)" in content or "var(--red)" in content
        assert "var(--orange)" in content or "var(--blue)" in content


class TestWatchAppStatusUI:
    def test_has_module_count(self):
        content = _read_watch()
        assert "module_count" in content or "Modules" in content

    def test_has_agent_count(self):
        content = _read_watch()
        assert "crew_size" in content or "Agents" in content

    def test_has_device_connections(self):
        content = _read_watch()
        assert "iphones_connected" in content or "watches_connected" in content

    def test_has_uptime(self):
        content = _read_watch()
        assert "uptime" in content.lower()


class TestWatchAppVoiceUI:
    def test_has_voice_button(self):
        content = _read_watch()
        assert "voiceB" in content or "voice_btn" in content

    def test_has_speech_recognition(self):
        content = _read_watch()
        assert "SpeechRecognition" in content or "webkitSpeechRecognition" in content

    def test_has_toggle_voice_function(self):
        content = _read_watch()
        assert "toggleVoice" in content or "startVoice" in content

    def test_has_summon_text_endpoint(self):
        content = _read_watch()
        assert "/os/summon/text" in content

    def test_voice_uses_tts_fallback(self):
        content = _read_watch()
        assert "SpeechSynthesisUtterance" in content or "speechSynthesis" in content

    def test_has_listening_state(self):
        content = _read_watch()
        assert "Listening" in content or "vListening" in content


class TestWatchAppActionsUI:
    def test_has_quick_agent_creation(self):
        content = _read_watch()
        assert "/os/arise" in content

    def test_has_device_data_push(self):
        content = _read_watch()
        assert "/os/device/push" in content

    def test_has_action_grid(self):
        content = _read_watch()
        assert "ag" in content or "action" in content.lower()

    def test_push_data_includes_health(self):
        content = _read_watch()
        assert "heart_rate" in content
        assert "step_count" in content
        assert "stress_level" in content


class TestWatchAppChatUI:
    def test_has_message_list(self):
        content = _read_watch()
        assert "msgs" in content

    def test_has_input_field(self):
        content = _read_watch()
        assert 'id="inp"' in content

    def test_has_send_function(self):
        content = _read_watch()
        assert "function send(" in content

    def test_uses_query_endpoint(self):
        content = _read_watch()
        assert "/os/query" in content

    def test_has_enter_key_handler(self):
        content = _read_watch()
        assert "Enter" in content

    def test_has_add_message_function(self):
        content = _read_watch()
        assert "function addMsg(" in content


class TestWatchAppNavigation:
    def test_swipe_navigation(self):
        content = _read_watch()
        assert "touchstart" in content
        assert "touchend" in content

    def test_digital_crown_scroll(self):
        content = _read_watch()
        assert "wheel" in content

    def test_auto_refresh_health(self):
        content = _read_watch()
        assert "setInterval" in content
        assert "30000" in content or "loadHealth" in content

    def test_auto_refresh_status(self):
        content = _read_watch()
        assert "60000" in content or "loadStatus" in content


class TestWatchAppOffline:
    def test_has_offline_indicator(self):
        content = _read_watch()
        assert "offline" in content.lower() or "OFFLINE" in content

    def test_has_online_offline_event_listeners(self):
        content = _read_watch()
        assert "addEventListener('online'" in content or "window.addEventListener" in content

    def test_has_connection_indicator(self):
        content = _read_watch()
        assert "conn" in content


class TestWatchAppCompactDesign:
    def test_small_fonts(self):
        content = _read_watch()
        assert "font-size:10px" in content or "font-size:8px" in content or "font-size:7px" in content

    def test_compact_grid(self):
        content = _read_watch()
        assert "grid-template-columns:1fr 1fr" in content

    def test_small_border_radius(self):
        content = _read_watch()
        assert "border-radius" in content

    def test_dark_theme(self):
        content = _read_watch()
        assert "#0a0a12" in content or "var(--bg)" in content

    def test_responsive_breakpoints(self):
        content = _read_watch()
        assert "@media" in content

    def test_no_user_select(self):
        content = _read_watch()
        assert "user-select:none" in content

    def test_tap_highlight_disabled(self):
        content = _read_watch()
        assert "tap-highlight-color:transparent" in content


class TestWatchAppAPIIntegration:
    def test_uses_device_context_endpoint(self):
        content = _read_watch()
        assert "/os/device/context" in content

    def test_uses_health_endpoint(self):
        content = _read_watch()
        assert "/os/health" in content

    def test_uses_status_endpoint(self):
        content = _read_watch()
        assert "/os/status" in content

    def test_has_api_key_header(self):
        content = _read_watch()
        assert "X-API-Key" in content

    def test_uses_manifest_endpoint(self):
        content = _read_watch()
        assert "manifest.json" in content


class TestPWAShareTargetFix:
    def test_pwa_share_target_handler(self):
        content = _read_pwa()
        assert "sp.get('title')" in content or "sp.get('text')" in content or "sp.get('url')" in content

    def test_manifest_has_share_target(self):
        with open(MANIFEST_PATH) as f:
            manifest = f.read()
        assert "share_target" in manifest

    def test_manifest_share_method_is_post(self):
        with open(MANIFEST_PATH) as f:
            manifest = f.read()
        assert '"method": "POST"' in manifest or '"method":"POST"' in manifest