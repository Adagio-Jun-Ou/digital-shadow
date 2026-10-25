#!/usr/bin/env python3
"""Tests for VisionPerception module."""

import pytest
import copy
import threading
from shadowd.cognition.vision_perception import (
    VisionPerceptionEngine, SceneCategory, VisionFeature,
    ChangeType, VisualMemory, DetectedObject, VisualChange, ColorInfo,
    _sanitize_content, _sanitize_strict,
)


class TestSanitization:
    def test_sanitize_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms") == "latency < 100ms"

    def test_sanitize_content_strips_control_chars(self):
        assert _sanitize_content("hello\x00world\x7f") == "helloworld"

    def test_sanitize_content_strips_and_truncates(self):
        assert _sanitize_content("  hello  ", 5) == "hello"

    def test_sanitize_strict_strips_angle_brackets(self):
        assert _sanitize_strict("latency < 100ms") == "latency  100ms"

    def test_sanitize_strict_strips_all_control(self):
        assert _sanitize_strict("hello\tworld\n") == "helloworld"


class TestDataclasses:
    def test_detected_object_sanitize_and_clamp(self):
        obj = DetectedObject(label="<script>alert(1)</script>", confidence=1.5)
        assert obj.label == "scriptalert(1)/script"
        assert obj.confidence == 1.0

    def test_detected_object_confidence_clamped_low(self):
        obj = DetectedObject(label="car", confidence=-0.5)
        assert obj.confidence == 0.0

    def test_detected_object_to_dict(self):
        obj = DetectedObject(label="car", confidence=0.8, x=10, y=20, width=100, height=50, area_ratio=0.15)
        d = obj.to_dict()
        assert d["label"] == "car"
        assert d["confidence"] == 0.8
        assert "bbox" in d

    def test_visual_change_sanitize_and_clamp(self):
        vc = VisualChange(
            change_type=ChangeType.SCENE_CHANGE,
            description="<b>bold</b> scene",
            severity=2.0,
        )
        assert vc.description == "<b>bold</b> scene"
        assert vc.severity == 1.0

    def test_visual_change_severity_clamped_low(self):
        vc = VisualChange(change_type=ChangeType.MOVED, description="test", severity=-0.5)
        assert vc.severity == 0.0

    def test_visual_memory_sanitize(self):
        vm = VisualMemory(
            description="test <b>scene</b>",
            source="<script>alert(1)</script>",
            extracted_text="<div>hello</div>",
        )
        assert vm.description == "test <b>scene</b>"
        assert vm.source == "scriptalert(1)/script"
        assert "div" in vm.extracted_text

    def test_visual_memory_importance_clamped(self):
        vm = VisualMemory(importance=2.0)
        assert vm.importance == 1.0

    def test_visual_memory_to_dict_deepcopy_metadata(self):
        vm = VisualMemory(metadata={"key": "val"})
        d = vm.to_dict()
        d["metadata"]["key"] = "changed"
        assert vm.metadata["key"] == "val"

    def test_visual_memory_tags_capped(self):
        tags = [f"tag{i}" for i in range(100)]
        vm = VisualMemory(tags=tags)
        assert len(vm.tags) == 50

    def test_visual_memory_objects_capped(self):
        objs = [DetectedObject(label=f"obj{i}", confidence=0.5) for i in range(200)]
        vm = VisualMemory(objects=objs)
        assert len(vm.objects) == 100

    def test_color_info_to_dict(self):
        ci = ColorInfo(
            dominant_colors=[(255, 0, 0), (0, 255, 0)],
            avg_brightness=0.7,
            avg_saturation=0.4,
            color_diversity=0.3,
        )
        d = ci.to_dict()
        assert len(d["dominant_colors"]) == 2
        assert d["avg_brightness"] == 0.7
        assert d["color_diversity"] == 0.3


class TestVisionEngine:
    def setup_method(self):
        self.engine = VisionPerceptionEngine(data_dir="/tmp/test_vision")

    def test_capture_and_analyze_no_image(self):
        result = self.engine.capture_and_analyze(description="a dark room")
        assert result.memory_id.startswith("vm_")
        assert result.description != ""
        assert result.importance >= 0.0

    def test_analyze_image(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = self.engine.analyze_image(img, description="gray rectangle")
        assert result.memory_id.startswith("vm_")
        assert result.width == 100
        assert result.height == 100

    def test_analyze_image_with_objects(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), color=(200, 200, 200))
        result = self.engine.analyze_image(img, description="a desk with computer and keyboard")
        assert len(result.objects) > 0
        object_labels = [o.label for o in result.objects]
        assert len(result.objects) > 0  # Objects detected from description

    def test_analyze_from_file_nonexistent(self):
        result = self.engine.analyze_from_file("/nonexistent/file.png")
        assert result.importance <= 0.3

    def test_search_memory(self):
        self.engine.capture_and_analyze(description="an outdoor scene with trees and sky")
        self.engine.capture_and_analyze(description="indoor office with desk and monitor")
        results = self.engine.search_memory("outdoor", limit=5)
        assert len(results) >= 1

    def test_search_memory_empty_query(self):
        self.engine.capture_and_analyze(description="test scene")
        results = self.engine.search_memory("", limit=5)
        assert len(results) >= 1

    def test_get_recent(self):
        for i in range(5):
            self.engine.capture_and_analyze(description=f"scene {i}")
        recent = self.engine.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_stats(self):
        self.engine.capture_and_analyze(description="test")
        stats = self.engine.get_stats()
        assert "stats" in stats
        assert "categories" in stats
        assert stats["stats"]["captures"] >= 1

    def test_classify_scene_keywords(self):
        result = self.engine._classify_scene("screen monitor keyboard laptop ide", ["screen"], [])
        assert result == SceneCategory.WORKSPACE

    def test_classify_scene_outdoor(self):
        result = self.engine._classify_scene("beautiful sky and trees", [], [])
        assert result == SceneCategory.OUTDOOR

    def test_detect_features_text(self):
        ci = ColorInfo(avg_brightness=0.8, avg_saturation=0.3, color_diversity=0.2)
        features = self.engine._detect_features(ci, faces=0, has_text=True, change_severity=0.0, width=1920, height=1080)
        assert VisionFeature.TEXT_DETECTED in features

    def test_detect_features_bright(self):
        ci = ColorInfo(avg_brightness=0.9, avg_saturation=0.1, color_diversity=0.05)
        features = self.engine._detect_features(ci, faces=0, has_text=False, change_severity=0.0, width=200, height=200)
        assert VisionFeature.BRIGHTNESS_HIGH in features

    def test_detect_features_dim(self):
        ci = ColorInfo(avg_brightness=0.1, avg_saturation=0.1, color_diversity=0.05)
        features = self.engine._detect_features(ci, faces=0, has_text=False, change_severity=0.0, width=200, height=200)
        assert VisionFeature.BRIGHTNESS_LOW in features

    def test_forget_memory(self):
        self.engine.capture_and_analyze(description="forget me")
        mem_id = None
        for m in list(self.engine._memories):
            if m.description == "forget me":
                mem_id = m.memory_id
                break
        assert mem_id is not None
        result = self.engine.forget(mem_id)
        assert result is True

    def test_forget_nonexistent(self):
        result = self.engine.forget("nonexistent_id")
        assert result is False

    def test_clear_memory(self):
        for i in range(3):
            self.engine.capture_and_analyze(description=f"scene {i}")
        count = self.engine.clear_memory()
        assert count >= 3
        assert len(self.engine._memories) == 0

    def test_change_detection(self):
        from PIL import Image
        img1 = Image.new("RGB", (100, 100), color=(100, 100, 100))
        self.engine.analyze_image(img1, description="gray scene")
        img2 = Image.new("RGB", (100, 100), color=(200, 200, 200))
        result = self.engine.analyze_image(img2, description="bright scene")
        assert result.similarity_to_previous >= 0.0

    def test_path_traversal_rejection(self):
        result = self.engine.analyze_from_file("../../etc/passwd")
        assert result.importance == 0.0

    def test_restricted_path_rejection(self):
        result = self.engine.analyze_from_file("/etc/shadow")
        assert result.importance == 0.0

    def test_get_memory_existing(self):
        self.engine.capture_and_analyze(description="test memory retrieval")
        mem_id = list(self.engine._memories)[-1].memory_id if self.engine._memories else "nonexistent"
        result = self.engine.get_memory(mem_id)
        assert result is not None
        assert result.memory_id == mem_id

    def test_get_memory_nonexistent(self):
        result = self.engine.get_memory("nonexistent_id_12345")
        assert result is None

    def test_thread_safety(self):
        results = []
        errors = []

        def analyze(n):
            try:
                r = self.engine.capture_and_analyze(description=f"thread scene {n}")
                results.append(r.memory_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyze, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 10


class TestShadowOSIntegration:
    def test_vision_in_health_check(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os._running = True
        os._start_time = 1734000000.0
        health = os.get_health()
        assert "vision" in health["modules"]

    def test_vision_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        status = os.get_status()
        assert "vision_stats" in status

    def test_analyze_visual_convenience(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        result = os.analyze_visual(description="test visual scene")
        assert "memory_id" in result

    def test_search_visual_memory_convenience(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os.analyze_visual(description="test visual scene")
        results = os.search_visual_memory("test", limit=5)
        assert isinstance(results, list)
        assert len(results) >= 1


class TestAPIModels:
    def test_vision_analyze_request(self):
        from shadowd.cognition.api_router import VisionAnalyzeRequest
        req = VisionAnalyzeRequest(description="test", source="api")
        assert req.description == "test"
        assert req.source == "api"

    def test_vision_search_request(self):
        from shadowd.cognition.api_router import VisionSearchRequest
        req = VisionSearchRequest(query="find images", limit=5)
        assert req.query == "find images"
        assert req.limit == 5

    def test_vision_file_request_traversal(self):
        from shadowd.cognition.api_router import VisionFileRequest
        with pytest.raises(Exception):
            VisionFileRequest(filepath="../../etc/passwd")

    def test_vision_file_request_restricted(self):
        from shadowd.cognition.api_router import VisionFileRequest
        with pytest.raises(Exception):
            VisionFileRequest(filepath="/etc/shadow")

    def test_vision_file_request_valid(self):
        from shadowd.cognition.api_router import VisionFileRequest
        req = VisionFileRequest(filepath="/home/user/image.py")
        assert req.filepath == "/home/user/image.py"
