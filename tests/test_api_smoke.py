#!/usr/bin/env python3
"""API smoke tests - verify all endpoint models compile and validate."""
import pytest
from pydantic import ValidationError
from shadowd.cognition.api_router import (
    VisionAnalyzeRequest, VisionSearchRequest, VisionFileRequest,
    PushNotifyRequest, MarkReadRequest,
    ProvisionShadowRequest, ShadowReportRequest, ShadowHeartbeatRequest,
    ShadowConfigRequest, ShadowDeprovisionRequest,
    JournalRecordRequest, VisualProcessRequest, VisualStreamRequest,
    FeedbackEmitRequest,
)

class TestVisionModels:
    def test_vision_analyze(self):
        r = VisionAnalyzeRequest(description="Test", source="api")
        assert r.description == "Test"
    
    def test_vision_search(self):
        r = VisionSearchRequest(query="test", limit=10)
        assert r.query == "test"
    
    def test_vision_file_blocks_traversal(self):
        with pytest.raises(ValidationError):
            VisionFileRequest(filepath="../../etc/passwd")
    
    def test_vision_file_blocks_restricted(self):
        with pytest.raises(ValidationError):
            VisionFileRequest(filepath="/etc/shadow")

class TestNotificationModels:
    def test_push_notify(self):
        r = PushNotifyRequest(title="Test", body="Body", priority="high")
        assert r.title == "Test"
    
    def test_push_notify_validates_title(self):
        with pytest.raises(ValidationError):
            PushNotifyRequest(title="", body="Body")
    
    def test_mark_read(self):
        r = MarkReadRequest(notification_id="abc123")
        assert r.notification_id == "abc123"

class TestDeployModels:
    def test_provision_shadow(self):
        r = ProvisionShadowRequest(name="Test Cam", device_id="d1", mode="sentry")
        assert r.name == "Test Cam"
    
    def test_provision_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            ProvisionShadowRequest(name="Test", device_id="d1", mode="invalid")
    
    def test_shadow_report(self):
        r = ShadowReportRequest(shadow_id="abc", auth_token="tok", faces_detected=1)
        assert r.faces_detected == 1
    
    def test_heartbeat(self):
        r = ShadowHeartbeatRequest(shadow_id="abc", auth_token="tok")
        assert r.shadow_id == "abc"
    
    def test_config(self):
        r = ShadowConfigRequest(shadow_id="abc", auth_token="tok", camera_interval=30)
        assert r.camera_interval == 30

class TestJournalModels:
    def test_journal_record(self):
        r = JournalRecordRequest(entry_type="observation", content="Test entry")
        assert r.entry_type == "observation"
    
    def test_journal_rejects_empty_content(self):
        with pytest.raises(ValidationError):
            JournalRecordRequest(entry_type="observation", content="")

class TestBridgeModels:
    def test_visual_process(self):
        r = VisualProcessRequest(description="Scene", faces=2, importance=0.8)
        assert r.faces == 2
    
    def test_visual_stream(self):
        r = VisualStreamRequest(interval_seconds=10, max_frames=100)
        assert r.interval_seconds == 10

class TestFeedbackModels:
    def test_feedback_emit(self):
        r = FeedbackEmitRequest(source_module="test", signal_type="pattern", intensity=0.5)
        assert r.intensity == 0.5
    
    def test_feedback_rejects_invalid_intensity(self):
        with pytest.raises(ValidationError):
            FeedbackEmitRequest(source_module="test", signal_type="pattern", intensity=2.0)
