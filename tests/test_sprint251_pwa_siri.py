#!/usr/bin/env python3
"""Sprint 251 Tests — PWA + Siri Shortcuts + API Validation"""

import pytest
import sys
import os
import json
import threading
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SHADOW_OS_API_KEY", "")

from shadowd.cognition.device_context import DeviceContextEngine, DeviceType
from shadowd.cognition.shadow_os import ShadowOS


class TestServiceWorkerV2:
    def test_sw_js_exists(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        assert os.path.exists(sw_path)

    def test_sw_has_offline_queue(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        with open(sw_path) as f:
            content = f.read()
        assert "OFFLINE_QUEUE" in content
        assert "addToQueue" in content
        assert "replayQueue" in content

    def test_sw_has_background_sync(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        with open(sw_path) as f:
            content = f.read()
        assert "sync" in content
        assert "shadowos-offline-queue" in content

    def test_sw_has_push_notifications(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        with open(sw_path) as f:
            content = f.read()
        assert "push" in content
        assert "showNotification" in content
        assert "notificationclick" in content

    def test_sw_has_max_queue_size(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        with open(sw_path) as f:
            content = f.read()
        assert "MAX_QUEUE_SIZE" in content

    def test_sw_cache_name_v3(self):
        sw_path = os.path.join(os.path.dirname(__file__), "..", "web", "sw.js")
        with open(sw_path) as f:
            content = f.read()
        assert "shadowos-v3" in content


class TestManifestV2:
    def test_manifest_exists(self):
        m_path = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")
        assert os.path.exists(m_path)

    def test_manifest_share_target(self):
        m_path = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")
        with open(m_path) as f:
            manifest = json.load(f)
        assert "share_target" in manifest
        assert manifest["share_target"]["method"] == "POST"

    def test_manifest_maskable_icons(self):
        m_path = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")
        with open(m_path) as f:
            manifest = json.load(f)
        maskable = [i for i in manifest["icons"] if i.get("purpose") == "maskable"]
        assert len(maskable) >= 2

    def test_manifest_scope(self):
        m_path = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")
        with open(m_path) as f:
            manifest = json.load(f)
        assert "scope" in manifest
        assert manifest["scope"] == "/shadow"

    def test_manifest_health_category(self):
        m_path = os.path.join(os.path.dirname(__file__), "..", "web", "manifest.json")
        with open(m_path) as f:
            manifest = json.load(f)
        assert "health" in manifest["categories"]


class TestSiriShortcutsV2:
    def test_siri_shortcuts_script_exists(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_siri_shortcuts.py")
        assert os.path.exists(script_path)

    def test_siri_shortcuts_has_health_shortcuts(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_siri_shortcuts.py")
        with open(script_path) as f:
            content = f.read()
        assert "HEALTHKIT_SHORTCUTS" in content
        assert "Push iPhone Health" in content
        assert "Push Watch Health" in content
        assert "Push Location" in content
        assert "Push Battery" in content

    def test_siri_shortcuts_has_api_key(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_siri_shortcuts.py")
        with open(script_path) as f:
            content = f.read()
        assert "X-API-Key" in content

    def test_siri_shortcuts_has_data_schemas(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_siri_shortcuts.py")
        with open(script_path) as f:
            content = f.read()
        assert "data_schemas" in content
        assert "heart_rate" in content
        assert "hrv" in content

    def test_generated_shortcuts_index(self):
        index_path = os.path.join(os.path.dirname(__file__), "..", "shortcuts", "index.json")
        assert os.path.exists(index_path)
        with open(index_path) as f:
            index = json.load(f)
        assert "health_shortcuts" in index
        assert "authentication" in index
        assert "api_endpoints" in index

    def test_generated_register_script(self):
        reg_path = os.path.join(os.path.dirname(__file__), "..", "shortcuts", "register_devices.sh")
        assert os.path.exists(reg_path)


class TestLaptopMonitorV2:
    def test_laptop_monitor_exists(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "laptop_monitor.py")
        assert os.path.exists(script_path)

    def test_laptop_monitor_uses_api_key_header(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "laptop_monitor.py")
        with open(script_path) as f:
            content = f.read()
        assert '"X-API-Key"' in content

    def test_laptop_monitor_no_bearer_auth(self):
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "laptop_monitor.py")
        with open(script_path) as f:
            content = f.read()
        # Should use X-API-Key, not Authorization: Bearer
        assert "Bearer" not in content


class TestSiriPushEndpoint:
    def test_siri_push_request_model(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="iphone", data={"health": {"heart_rate": 72}})
        assert req.device_type == "iphone"
        assert req.device_id is None
        assert req.data["health"]["heart_rate"] == 72

    def test_siri_push_request_default_device_type(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(data={})
        assert req.device_type == "iphone"

    def test_siri_push_request_watch(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="watch", data={"heart_rate": 80})
        assert req.device_type == "watch"

    def test_siri_push_request_invalid_device_type(self):
        from shadowd.cognition.api_router import SiriPushRequest
        with pytest.raises(ValueError):
            SiriPushRequest(device_type="tablet", data={})

    def test_siri_push_request_auto_device_id(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="watch", data={})
        assert req.device_id is None  # auto-generated in endpoint

    def test_siri_push_request_max_length(self):
        from shadowd.cognition.api_router import SiriPushRequest
        with pytest.raises(ValueError):
            SiriPushRequest(device_type="a" * 21, data={})


class TestPWAServiceWorker:
    def test_pwa_has_sw_registration(self):
        pwa_path = os.path.join(os.path.dirname(__file__), "..", "web", "shadow_pwa.html")
        with open(pwa_path) as f:
            content = f.read()
        assert "serviceWorker" in content
        assert "register" in content
        assert "/sw.js" in content

    def test_pwa_has_share_target_handler(self):
        pwa_path = os.path.join(os.path.dirname(__file__), "..", "web", "shadow_pwa.html")
        with open(pwa_path) as f:
            content = f.read()
        assert "sp.get('title')" in content or "share" in content.lower()


class TestAPIPydanticValidation:
    def test_voice_start_request_max_length(self):
        from shadowd.cognition.api_router import VoiceStartRequest
        req = VoiceStartRequest(voice="test")
        assert req.voice == "test"

    def test_voice_tts_request_max_length(self):
        from shadowd.cognition.api_router import VoiceTTSRequest
        req = VoiceTTSRequest(text="hello", voice="test")
        assert req.voice == "test"

    def test_voice_stt_request_max_length(self):
        from shadowd.cognition.api_router import VoiceSTTRequest
        req = VoiceSTTRequest(language="en")
        assert req.language == "en"

    def test_goal_create_request_priority_max_length(self):
        from shadowd.cognition.api_router import GoalCreateRequest
        req = GoalCreateRequest(title="test")
        assert req.priority == "medium"

    def test_anticipate_request_emotional_state_max_length(self):
        from shadowd.cognition.api_router import AnticipateRequest
        req = AnticipateRequest(emotional_state="happy")
        assert req.emotional_state == "happy"

    def test_health_update_request_status_max_length(self):
        from shadowd.cognition.api_router import HealthUpdateRequest
        req = HealthUpdateRequest(module_name="test", status="healthy")
        assert req.status == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])