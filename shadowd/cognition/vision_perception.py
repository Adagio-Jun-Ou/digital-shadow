#!/usr/bin/env python3
"""
VisionPerception - Visual Perception Engine for ShadowOS

Captures, describes, remembers, and reasons about visual information
from cameras, screenshots, and image files. Feeds visual observations
into the cognitive pipeline for cross-modal synthesis.

Architecture:
  ImageSource → VisionPerceptionEngine → Cognitive Modules
                    │
                    ├── SceneDescription (what's in the frame)
                    ├── ObjectDetection (what objects are present)
                    ├── TextExtraction (OCR-like text from images)
                    ├── ChangeDetection (what changed between frames)
                    ├── FaceDetection (presence/count of faces)
                    ├── SceneClassification (indoor/outdoor/social/etc)
                    ├── ColorAnalysis (dominant colors, brightness)
                    └── VisualMemory (remember what was seen)

Data Sources:
  Webcam (via PIL/Pillow):
    - Periodic frame capture
    - Scene description and object tagging
    - Face presence detection
    
  Screenshots (via PIL/Pillow):
    - Active window capture
    - Text extraction via pixel analysis
    - UI element detection
    
  Image Files:
    - Direct analysis of provided images
    - Feature extraction and embedding
    - Similarity comparison

Vision Pipeline:
  1. Capture/Acquire image (from camera, file, or base64)
  2. Analyze image properties (dimensions, colors, brightness)
  3. Detect features (edges, regions, text-like areas)
  4. Classify scene (category tags)
  5. Extract text (OCR simulation via pixel patterns)
  6. Compare with previous frames (change detection)
  7. Store in visual memory with metadata
  8. Feed observations into cognitive modules
"""

import copy
import hashlib
import io
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("shadowd.cognition.vision_perception")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_DESCRIPTION = 5000
_MAX_TEXT = 10000
_MAX_TAGS = 50
_MAX_OBJECTS = 100
_MAX_MEMORY = 5000
_MAX_HISTORY = 2000
_MAX_CHANGE_AREAS = 50
_MAX_FACES_PER_FRAME = 20


def _sanitize_content(text: str, max_len: int = _MAX_DESCRIPTION) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


def _sanitize_strict(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


class SceneCategory(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    SOCIAL = "social"
    WORKSPACE = "workspace"
    NATURE = "nature"
    URBAN = "urban"
    TRANSPORT = "transport"
    FOOD = "food"
    SPORTS = "sports"
    MEDICAL = "medical"
    ABSTRACT = "abstract"
    TEXT_HEAVY = "text_heavy"
    DARK = "dark"
    UNKNOWN = "unknown"


class VisionFeature(str, Enum):
    FACE_DETECTED = "face_detected"
    TEXT_DETECTED = "text_detected"
    BRIGHTNESS_HIGH = "brightness_high"
    BRIGHTNESS_LOW = "brightness_low"
    MOTION_DETECTED = "motion_detected"
    DOMINANT_RED = "dominant_red"
    DOMINANT_BLUE = "dominant_blue"
    DOMINANT_GREEN = "dominant_green"
    DOMINANT_WHITE = "dominant_white"
    DOMINANT_BLACK = "dominant_black"
    COMPLEX_SCENE = "complex_scene"
    SIMPLE_SCENE = "simple_scene"
    SCREEN_CONTENT = "screen_content"
    DOCUMENT = "document"


class ChangeType(str, Enum):
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    MOVED = "moved"
    COLOR_CHANGED = "color_changed"
    BRIGHTNESS_CHANGED = "brightness_changed"
    TEXT_CHANGED = "text_changed"
    SCENE_CHANGE = "scene_change"


@dataclass
class ColorInfo:
    """Dominant color information from an image region."""
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    avg_brightness: float = 0.5
    avg_saturation: float = 0.3
    color_diversity: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "dominant_colors": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.dominant_colors[:5]],
            "avg_brightness": round(self.avg_brightness, 3),
            "avg_saturation": round(self.avg_saturation, 3),
            "color_diversity": round(self.color_diversity, 3),
        }


@dataclass
class DetectedObject:
    """An object detected in an image with bounding box."""
    label: str
    confidence: float
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    area_ratio: float = 0.0

    def __post_init__(self):
        self.label = _sanitize_strict(self.label, 100)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": {"x": self.x, "y": self.y, "w": self.width, "h": self.height},
            "area_ratio": round(self.area_ratio, 3),
        }


@dataclass
class VisualChange:
    """A change detected between two frames."""
    change_type: ChangeType
    description: str
    severity: float
    region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        self.description = _sanitize_content(self.description, 500)
        self.severity = max(0.0, min(1.0, self.severity))

    def to_dict(self) -> Dict:
        return {
            "change_type": self.change_type.value,
            "description": self.description,
            "severity": round(self.severity, 3),
            "region": list(self.region),
            "timestamp": self.timestamp,
        }


@dataclass
class VisualMemory:
    """A stored visual observation."""
    memory_id: str = ""
    source: str = ""
    scene_category: SceneCategory = SceneCategory.UNKNOWN
    description: str = ""
    extracted_text: str = ""
    objects: List[DetectedObject] = field(default_factory=list)
    features: List[VisionFeature] = field(default_factory=list)
    color_info: ColorInfo = field(default_factory=ColorInfo)
    faces_detected: int = 0
    tags: List[str] = field(default_factory=list)
    change_from_previous: Optional[VisualChange] = None
    similarity_to_previous: float = 0.0
    importance: float = 0.5
    image_hash: str = ""
    width: int = 0
    height: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = f"vm_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"
        self.description = _sanitize_content(self.description, _MAX_DESCRIPTION)
        self.extracted_text = _sanitize_content(self.extracted_text, _MAX_TEXT)
        self.source = _sanitize_strict(self.source, 100)
        self.objects = self.objects[:_MAX_OBJECTS]
        self.tags = [_sanitize_strict(t, 50) for t in self.tags[:_MAX_TAGS]]
        self.importance = max(0.0, min(1.0, self.importance))

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "source": self.source,
            "scene_category": self.scene_category.value,
            "description": self.description,
            "extracted_text": self.extracted_text[:500] if self.extracted_text else "",
            "objects": [o.to_dict() for o in self.objects[:10]],
            "features": [f.value for f in self.features],
            "color_info": self.color_info.to_dict(),
            "faces_detected": self.faces_detected,
            "tags": list(self.tags),
            "change_from_previous": self.change_from_previous.to_dict() if self.change_from_previous else None,
            "similarity_to_previous": round(self.similarity_to_previous, 3),
            "importance": round(self.importance, 3),
            "image_hash": self.image_hash,
            "dimensions": f"{self.width}x{self.height}",
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


class VisionPerceptionEngine:
    """
    Visual Perception Engine for ShadowOS.

    Captures images from cameras/files, analyzes visual content,
    detects changes, extracts text, and feeds observations into the
    cognitive pipeline. Operates entirely with Pillow for image I/O
    and heuristic analysis (no GPU/ML model required for core operation).

    Optional: LLM-based scene description when Ollama is available.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), "digital-shadow", "data", "vision"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._memories: deque = deque(maxlen=_MAX_MEMORY)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._previous_hash: Optional[str] = None
        self._previous_description: Optional[str] = None
        self._previous_objects: List[str] = []
        self._capture_count: int = 0
        self._face_count: int = 0
        self._change_count: int = 0
        self._text_count: int = 0

        self._scene_keywords: Dict[SceneCategory, List[str]] = {
            SceneCategory.INDOOR: ["office", "room", "desk", "computer", "monitor", "chair", "table", "kitchen", "bedroom", "bathroom", "living", "lamp", "furniture"],
            SceneCategory.OUTDOOR: ["sky", "tree", "grass", "road", "building", "mountain", "field", "park", "horizon", "cloud", "sun", "sidewalk"],
            SceneCategory.SOCIAL: ["people", "group", "crowd", "party", "meeting", "conference", "dinner", "gathering"],
            SceneCategory.WORKSPACE: ["screen", "monitor", "keyboard", "mouse", "laptop", "code", "terminal", "ide", "spreadsheet"],
            SceneCategory.NATURE: ["forest", "river", "lake", "ocean", "mountain", "flower", "animal", "bird", "fish"],
            SceneCategory.URBAN: ["car", "bus", "street", "traffic", "store", "sign", "bridge", "tower", "skyscraper"],
            SceneCategory.TRANSPORT: ["car", "bus", "train", "plane", "boat", "bicycle", "motorcycle", "road", "highway"],
            SceneCategory.FOOD: ["plate", "food", "drink", "meal", "restaurant", "cooking", "fruit", "vegetable"],
            SceneCategory.MEDICAL: ["hospital", "medicine", "doctor", "stethoscope", "x-ray", "syringe", "blood"],
            SceneCategory.TEXT_HEAVY: ["text", "document", "paper", "book", "screen", "code", "sign", "label"],
        }

        self._object_keywords: Dict[str, List[str]] = {
            "person": ["person", "man", "woman", "child", "people", "face"],
            "screen": ["screen", "monitor", "display", "laptop", "phone", "tablet", "tv"],
            "document": ["paper", "document", "book", "letter", "page", "notebook"],
            "vehicle": ["car", "bus", "truck", "bicycle", "motorcycle", "train", "plane"],
            "furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet"],
            "animal": ["dog", "cat", "bird", "fish", "horse", "rabbit"],
            "food": ["food", "fruit", "vegetable", "meal", "plate", "cup", "bowl"],
            "nature": ["tree", "flower", "grass", "sky", "cloud", "water", "stone"],
            "electronics": ["phone", "laptop", "keyboard", "mouse", "headphones", "camera"],
            "clothing": ["shirt", "pants", "dress", "hat", "jacket", "shoe"],
        }

        self._text_patterns = [
            re.compile(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+'),
            re.compile(r'(?:\d+\.?\d*\s*)(?:px|mm|cm|m|km|in|ft|lb|kg|mph|kb|mb|gb|%|°)'),
            re.compile(r'[A-Z]{2,}'),
            re.compile(r'\b\d{1,2}[:/]\d{2}\b'),
            re.compile(r'\b\w+@\w+\.\w+\b'),
            re.compile(r'https?://\S+'),
        ]

        self._face_region_params = {
            "min_height_ratio": 0.08,
            "max_height_ratio": 0.45,
            "min_width_ratio": 0.05,
            "max_width_ratio": 0.35,
            "aspect_low": 0.6,
            "aspect_high": 1.4,
        }

        self._llm_available: bool = False
        self._llm = None

        self._streaming = False
        self._stream_interval = 5.0
        self._stream_max_frames = 3600
        self._stream_frame_count = 0
        self._stream_thread = None

        self._stats = {
            "captures": 0,
            "scenes_classified": 0,
            "objects_detected": 0,
            "texts_extracted": 0,
            "changes_detected": 0,
            "faces_detected": 0,
            "memories_stored": 0,
            "llm_descriptions": 0,
            "streams_started": 0,
            "streams_stopped": 0,
            "screenshots_captured": 0,
        }

    def _inc_stat(self, key: str, delta: int = 1):
        with self._stats_lock:
            self._stats[key] += delta

    def _compute_image_hash(self, img) -> str:
        try:
            if hasattr(img, 'tobytes'):
                data = img.tobytes()
            else:
                data = str(img).encode()
            return hashlib.sha256(data).hexdigest()[:16]
        except (OSError, TypeError, AttributeError):
            return hashlib.sha256(os.urandom(16)).hexdigest()[:16]

    def _analyze_colors(self, img) -> ColorInfo:
        try:
            if not hasattr(img, 'size') or not hasattr(img, 'convert'):
                return ColorInfo()
            small = img.convert('RGB').resize((32, 32))
            pixels = list(small.getdata())
            if not pixels:
                return ColorInfo()

            total_r = sum(p[0] for p in pixels)
            total_g = sum(p[1] for p in pixels)
            total_b = sum(p[2] for p in pixels)
            n = len(pixels)

            avg_r = total_r / n
            avg_g = total_g / n
            avg_b = total_b / n

            brightness_values = [(p[0] * 299 + p[1] * 587 + p[2] * 114) / 1000 for p in pixels]
            avg_brightness = sum(brightness_values) / n / 255.0

            max_b = max(brightness_values) / 255.0
            min_b = min(brightness_values) / 255.0
            saturation_values = []
            for p in pixels:
                r, g, b = p[0] / 255.0, p[1] / 255.0, p[2] / 255.0
                mx = max(r, g, b)
                mn = min(r, g, b)
                s = (mx - mn) / mx if mx > 0 else 0
                saturation_values.append(s)
            avg_saturation = sum(saturation_values) / len(saturation_values)

            buckets: Dict[Tuple[int, int, int], int] = {}
            for p in pixels:
                qr = (p[0] // 64) * 64
                qg = (p[1] // 64) * 64
                qb = (p[2] // 64) * 64
                key = (qr, qg, qb)
                buckets[key] = buckets.get(key, 0) + 1
            dominant = sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:5]
            dominant_colors = [(r, g, b) for (r, g, b), _ in dominant]

            color_diversity = len(buckets) / max(1, (32 * 32))

            return ColorInfo(
                dominant_colors=dominant_colors,
                avg_brightness=avg_brightness,
                avg_saturation=avg_saturation,
                color_diversity=color_diversity,
            )
        except (OSError, TypeError, ValueError, AttributeError):
            return ColorInfo()

    def _classify_scene(self, description: str, objects: List[str], features: List[VisionFeature]) -> SceneCategory:
        desc_lower = (description or "").lower()
        all_text = desc_lower + " " + " ".join(objects).lower()

        best_category = SceneCategory.UNKNOWN
        best_score = 0

        for category, keywords in self._scene_keywords.items():
            score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', all_text))
            if score > best_score:
                best_score = score
                best_category = category

        if best_score == 0 and features:
            if VisionFeature.TEXT_HEAVY in features:
                return SceneCategory.TEXT_HEAVY
            if VisionFeature.BRIGHTNESS_LOW in features:
                return SceneCategory.DARK

        return best_category

    def _detect_features(self, color_info: ColorInfo, faces: int,
                         has_text: bool, change_severity: float,
                         width: int, height: int) -> List[VisionFeature]:
        features = []

        if faces > 0:
            features.append(VisionFeature.FACE_DETECTED)
        if has_text:
            features.append(VisionFeature.TEXT_DETECTED)
        if color_info.avg_brightness > 0.75:
            features.append(VisionFeature.BRIGHTNESS_HIGH)
        elif color_info.avg_brightness < 0.25:
            features.append(VisionFeature.BRIGHTNESS_LOW)
        if change_severity > 0.5:
            features.append(VisionFeature.MOTION_DETECTED)

        if color_info.dominant_colors:
            r, g, b = color_info.dominant_colors[0]
            if r > 180 and g < 100 and b < 100:
                features.append(VisionFeature.DOMINANT_RED)
            elif b > 180 and r < 100 and g < 100:
                features.append(VisionFeature.DOMINANT_BLUE)
            elif g > 150 and r < 100 and b < 100:
                features.append(VisionFeature.DOMINANT_GREEN)
            elif r > 200 and g > 200 and b > 200:
                features.append(VisionFeature.DOMINANT_WHITE)
            elif r < 50 and g < 50 and b < 50:
                features.append(VisionFeature.DOMINANT_BLACK)

        complexity = (len(set(features)) + color_info.color_diversity * 10) / 2
        if complexity > 5:
            features.append(VisionFeature.COMPLEX_SCENE)
        else:
            features.append(VisionFeature.SIMPLE_SCENE)

        aspect = width / max(1, height)
        if 1.3 < aspect < 2.5 and has_text:
            features.append(VisionFeature.SCREEN_CONTENT)
        if has_text and not any(f in features for f in [VisionFeature.FACE_DETECTED, VisionFeature.MOTION_DETECTED]):
            features.append(VisionFeature.DOCUMENT)

        return features

    def _detect_text_regions(self, img) -> Tuple[bool, str, List[Dict]]:
        try:
            if not hasattr(img, 'size') or not hasattr(img, 'convert'):
                return False, "", []

            gray = img.convert('L')
            width, height = gray.size

            if width < 10 or height < 10:
                return False, "", []

            block_w = max(8, width // 16)
            block_h = max(8, height // 16)
            text_blocks: List[Dict] = []
            pixels = gray.load()

            for by in range(0, height - block_h, block_h):
                for bx in range(0, width - block_w, block_w):
                    edges = 0
                    total = 0
                    for y in range(by, min(by + block_h, height - 1)):
                        for x in range(bx, min(bx + block_w, width - 1)):
                            dx = abs(pixels[x + 1, y] - pixels[x, y])
                            dy = abs(pixels[x, y + 1] - pixels[x, y])
                            if dx > 30 or dy > 30:
                                edges += 1
                            total += 1

                    if total > 0 and edges / total > 0.15:
                        text_blocks.append({
                            "x": bx, "y": by,
                            "w": block_w, "h": block_h,
                            "edge_ratio": round(edges / max(1, total), 3),
                        })

            has_text = len(text_blocks) > 3
            text_density = len(text_blocks) / max(1, (width // block_w) * (height // block_h))

            extracted = ""
            if has_text and text_density > 0.05:
                extracted = f"[{len(text_blocks)} text regions detected, density={round(text_density, 3)}]"

            return has_text, extracted, text_blocks[:_MAX_CHANGE_AREAS]
        except (OSError, TypeError, ValueError, AttributeError):
            return False, "", []

    def _detect_faces_heuristic(self, img, width: int, height: int) -> int:
        try:
            if not hasattr(img, 'convert'):
                return 0

            skin_hsv_lower = (0, 30, 60)
            skin_hsv_upper = (30, 180, 255)

            small = img.convert('HSV').resize((64, 64))
            pixels = list(small.getdata())

            skin_pixels = 0
            region_width = 8
            region_height = 16

            for by in range(0, 64 - region_height, 8):
                consecutive_skin = 0
                max_consecutive_skin = 0
                for bx in range(0, 64 - region_width, 4):
                    skin_count = 0
                    total_count = 0
                    for y in range(by, min(by + region_height, 64)):
                        for x in range(bx, min(bx + region_width, 64)):
                            idx = y * 64 + x
                            if idx < len(pixels):
                                h, s, v = pixels[idx]
                                h_norm = h * 2 if isinstance(h, int) or h < 180 else h
                                total_count += 1
                                if (skin_hsv_lower[0] <= h_norm <= skin_hsv_upper[0] and
                                    skin_hsv_lower[1] <= s <= skin_hsv_upper[1] and
                                    skin_hsv_lower[2] <= v <= skin_hsv_upper[2]):
                                    skin_count += 1

                    skin_ratio = skin_count / max(1, total_count)
                    if skin_ratio > 0.55:
                        consecutive_skin += 1
                        max_consecutive_skin = max(max_consecutive_skin, consecutive_skin)
                    else:
                        consecutive_skin = 0

            face_estimate = max(0, max_consecutive_skin // 3)
            return min(face_estimate, _MAX_FACES_PER_FRAME)

        except (OSError, TypeError, ValueError, AttributeError):
            return 0

    def _detect_changes(self, current_hash: str, current_desc: str,
                        current_objects: List[str], previous_hash: Optional[str],
                        previous_desc: Optional[str], previous_objects: List[str],
                        color_info: ColorInfo, previous_color: Optional[ColorInfo]) -> Optional[VisualChange]:
        if previous_hash is None:
            return None

        if current_hash == previous_hash:
            return None

        changes = []

        if not previous_desc:
            return VisualChange(
                change_type=ChangeType.SCENE_CHANGE,
                description="First observation",
                severity=1.0,
            )

        from difflib import SequenceMatcher
        text_sim = SequenceMatcher(None, current_desc.lower(), previous_desc.lower()).ratio()
        if text_sim < 0.3:
            return VisualChange(
                change_type=ChangeType.SCENE_CHANGE,
                description=f"Major scene change (similarity={round(text_sim, 2)})",
                severity=1.0 - text_sim,
            )

        current_set = set(current_objects)
        previous_set = set(previous_objects)
        appeared = current_set - previous_set
        disappeared = previous_set - current_set

        if appeared:
            return VisualChange(
                change_type=ChangeType.APPEARED,
                description=f"New objects appeared: {', '.join(list(appeared)[:5])}",
                severity=min(1.0, len(appeared) * 0.2),
            )

        if disappeared:
            return VisualChange(
                change_type=ChangeType.DISAPPEARED,
                description=f"Objects disappeared: {', '.join(list(disappeared)[:5])}",
                severity=min(1.0, len(disappeared) * 0.2),
            )

        if previous_color and color_info.avg_brightness != 0:
            brightness_diff = abs(color_info.avg_brightness - previous_color.avg_brightness)
            if brightness_diff > 0.2:
                return VisualChange(
                    change_type=ChangeType.BRIGHTNESS_CHANGED,
                    description=f"Brightness shift: {round(previous_color.avg_brightness, 2)} -> {round(color_info.avg_brightness, 2)}",
                    severity=brightness_diff,
                )

        if text_sim < 0.7:
            return VisualChange(
                change_type=ChangeType.MOVED,
                description=f"Scene shifted (similarity={round(text_sim, 2)})",
                severity=1.0 - text_sim,
            )

        return None

    def _compute_similarity(self, current_hash: str, previous_hash: Optional[str],
                            current_desc: str, previous_desc: Optional[str]) -> float:
        if previous_hash is None:
            return 0.0
        if current_hash == previous_hash:
            return 1.0

        if previous_desc and current_desc:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, current_desc.lower(), previous_desc.lower()).ratio()

        return 0.5

    def capture_and_analyze(self, image_source=None, description: str = "",
                           source: str = "webcam") -> VisualMemory:
        if image_source is not None:
            return self.analyze_image(image_source, description=description, source=source)

        memory = VisualMemory(
            source=_sanitize_strict(source),
            scene_category=SceneCategory.UNKNOWN,
            description=_sanitize_content(description or "No image captured"),
            importance=0.3,
        )
        with self._lock:
            self._memories.append(memory)
            self._history.append(memory.to_dict())
        self._inc_stat("captures")
        return memory

    def analyze_image(self, img, description: str = "",
                     source: str = "file") -> VisualMemory:
        width, height = 0, 0
        image_hash = ""
        color_info = ColorInfo()
        has_text = False
        extracted_text = ""
        text_regions: List[Dict] = []
        faces = 0

        try:
            width, height = img.size
            image_hash = self._compute_image_hash(img)
            self._inc_stat("captures")
        except (OSError, TypeError, AttributeError) as e:
            logger.warning("Image analysis failed: %s", type(e).__name__)

        try:
            color_info = self._analyze_colors(img)
        except (OSError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Color analysis failed: %s", type(e).__name__)

        try:
            has_text, extracted_text, text_regions = self._detect_text_regions(img)
            if has_text:
                self._inc_stat("texts_extracted")
                self._text_count += 1
        except (OSError, TypeError, ValueError, AttributeError):
            pass

        try:
            faces = self._detect_faces_heuristic(img, width, height)
            if faces > 0:
                self._inc_stat("faces_detected")
                self._face_count += faces
        except (OSError, TypeError, ValueError, AttributeError):
            pass

        detected_objects: List[DetectedObject] = []
        if description:
            for obj_label, keywords in self._object_keywords.items():
                for kw in keywords:
                    if re.search(r'\b' + re.escape(kw) + r'\b', description.lower()):
                        detected_objects.append(DetectedObject(
                            label=obj_label,
                            confidence=0.7,
                            area_ratio=0.1,
                        ))
                        break

        if not description:
            brightness_desc = "bright" if color_info.avg_brightness > 0.6 else "dark" if color_info.avg_brightness < 0.3 else "moderate"
            dominant = color_info.dominant_colors[0] if color_info.dominant_colors else (128, 128, 128)
            color_name = {0: "dark", 64: "muted", 128: "medium", 192: "bright", 255: "pure"}.get(
                (dominant[0] // 64) * 64, "colored"
            )
            text_note = f", with text regions" if has_text else ""
            face_note = f", {faces} face(s)" if faces > 0 else ""
            description = f"{brightness_desc} scene with {color_name} tones{text_note}{face_note}"

        obj_labels = [o.label for o in detected_objects]

        with self._lock:
            previous_hash = self._previous_hash
            previous_desc = self._previous_description
            previous_objects = list(self._previous_objects)
            previous_color = None
            for m in reversed(self._memories):
                if m.color_info and m.color_info.avg_brightness > 0:
                    previous_color = m.color_info
                    break

        change = self._detect_changes(
            image_hash, description, obj_labels,
            previous_hash, previous_desc, previous_objects,
            color_info, previous_color,
        )
        similarity = self._compute_similarity(
            image_hash, previous_hash, description, previous_desc,
        )

        if change:
            self._inc_stat("changes_detected")
            self._change_count += 1

        scene_category = self._classify_scene(description, obj_labels, [])
        self._inc_stat("scenes_classified")

        features = self._detect_features(color_info, faces, has_text, 0.0, width, height)

        importance = 0.5
        if faces > 0:
            importance = min(1.0, importance + faces * 0.1)
        if has_text:
            importance = min(1.0, importance + 0.1)
        if change and change.severity > 0.5:
            importance = min(1.0, importance + 0.15)
        if scene_category in (SceneCategory.SOCIAL, SceneCategory.WORKSPACE):
            importance = min(1.0, importance + 0.05)

        memory = VisualMemory(
            source=_sanitize_strict(source, 100),
            scene_category=scene_category,
            description=_sanitize_content(description, _MAX_DESCRIPTION),
            extracted_text=extracted_text,
            objects=detected_objects,
            features=features,
            color_info=color_info,
            faces_detected=faces,
            tags=obj_labels[:_MAX_TAGS],
            change_from_previous=change,
            similarity_to_previous=similarity,
            importance=importance,
            image_hash=image_hash,
            width=width,
            height=height,
        )

        with self._lock:
            self._memories.append(memory)
            self._history.append(memory.to_dict())
            self._previous_hash = image_hash
            self._previous_description = description
            self._previous_objects = obj_labels
            self._inc_stat("memories_stored")
            self._inc_stat("objects_detected", len(detected_objects))

        return memory

    def analyze_from_file(self, filepath: str, description: str = "") -> VisualMemory:
        filepath = _sanitize_strict(filepath, 500)
        description = _sanitize_content(description, _MAX_DESCRIPTION) if description else ""
        # Normalise separators so POSIX-style requests are evaluated
        # consistently on any host platform.
        normalized = os.path.normpath(filepath).replace("\\", "/")
        if '..' in normalized.split("/"):
            logger.warning("Path traversal rejected: %s", filepath[:100])
            return VisualMemory(source="file", description="Invalid path", importance=0.0)
        restricted = ('/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/run/',
                      '/var/', '/boot/', '/bin/', '/sbin/', '/usr/')
        lowered = normalized.lower()
        for prefix in restricted:
            if lowered.startswith(prefix):
                logger.warning("Restricted path rejected: %s", filepath[:100])
                return VisualMemory(source="file", description="Restricted path", importance=0.0)

        try:
            from PIL import Image
            img = Image.open(filepath)
            img.load()
            if not description:
                description = f"Image from file: {_sanitize_strict(os.path.basename(filepath), 200)}"
            return self.analyze_image(img, description=description, source=f"file:{os.path.basename(filepath)}")
        except ImportError:
            logger.warning("Pillow not available for image analysis")
            return VisualMemory(
                source="file",
                description=_sanitize_content(f"Image: {filepath} (Pillow unavailable)", _MAX_DESCRIPTION),
                importance=0.3,
            )
        except (OSError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Failed to open image file %s: %s", filepath[:100], type(e).__name__)
            return VisualMemory(
                source="file",
                description=_sanitize_content(f"Failed to load image: {filepath[:100]}", _MAX_DESCRIPTION),
                importance=0.1,
            )

    def analyze_from_bytes(self, data: bytes, description: str = "",
                           source: str = "upload") -> VisualMemory:
        if len(data) > 10 * 1024 * 1024:
            logger.warning("Image too large: %d bytes (max 10MB)", len(data))
            return VisualMemory(
                source=_sanitize_strict(source),
                description="Image too large (max 10MB)",
                importance=0.0,
            )

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.load()
            return self.analyze_image(img, description=description, source=source)
        except ImportError:
            logger.warning("Pillow not available for image analysis")
            return VisualMemory(
                source=_sanitize_strict(source),
                description=_sanitize_content(description or "Image (Pillow unavailable)", _MAX_DESCRIPTION),
                importance=0.3,
            )
        except (OSError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Failed to decode image: %s", type(e).__name__)
            return VisualMemory(
                source=_sanitize_strict(source),
                description=_sanitize_content(f"Image decode failed: {type(e).__name__}", _MAX_DESCRIPTION),
                importance=0.1,
            )

    def capture_frame(self, camera_index: int = 0) -> VisualMemory:
        try:
            from PIL import Image
            import subprocess
            result = subprocess.run(
                ["fswebcam", "-d", f"/dev/video{camera_index}", "--resolution", "640x480",
                 "--no-banner", "--png", "0", "-"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                img = Image.open(io.BytesIO(result.stdout))
                img.load()
                return self.analyze_image(img, description="Webcam capture", source=f"webcam:{camera_index}")
        except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            import subprocess
            result = subprocess.run(
                ["imagesnap", "-w", "2", "-"],
                capture_output=True, timeout=10,
            )
            if result.stdout:
                from PIL import Image
                img = Image.open(io.BytesIO(result.stdout))
                img.load()
                return self.analyze_image(img, description="iSight capture", source="isight")
        except (OSError, FileNotFoundError, subprocess.TimeoutExpired, ImportError):
            pass

        self._inc_stat("captures")
        return VisualMemory(
            source="webcam",
            description="Camera unavailable",
            importance=0.0,
        )

    def get_memory(self, memory_id: str) -> Optional[VisualMemory]:
        memory_id = _sanitize_strict(memory_id, 50)
        with self._lock:
            for m in self._memories:
                if m.memory_id == memory_id:
                    return copy.deepcopy(m)
        return None

    def search_memory(self, query: str = "", scene_category: Optional[str] = None,
                      tags: Optional[List[str]] = None, limit: int = 20) -> List[Dict]:
        query = _sanitize_content(query, 500) if query else ""
        limit = max(1, min(100, limit))
        results = []

        with self._lock:
            for m in reversed(self._memories):
                score = 0.0

                if query:
                    query_lower = query.lower()
                    if query_lower in m.description.lower():
                        score += 0.5
                    if query_lower in m.extracted_text.lower():
                        score += 0.3
                    for tag in m.tags:
                        if query_lower in tag.lower():
                            score += 0.2

                if scene_category and m.scene_category.value == scene_category:
                    score += 0.3

                if tags:
                    for tag in tags:
                        tag_lower = tag.lower()
                        if any(tag_lower in t.lower() for t in m.tags):
                            score += 0.2

                if m.faces_detected > 0:
                    score += 0.1

                if m.change_from_previous and m.change_from_previous.severity > 0.5:
                    score += 0.1

                if score > 0 or not (query or scene_category or tags):
                    results.append((m, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [m.to_dict() for m, _ in results[:limit]]

    def get_recent(self, limit: int = 20) -> List[Dict]:
        limit = max(1, min(100, limit))
        with self._lock:
            return [m.to_dict() for m in list(self._memories)[-limit:]]

    def get_memory_stats(self) -> Dict:
        with self._lock:
            categories = {}
            for m in self._memories:
                cat = m.scene_category.value
                categories[cat] = categories.get(cat, 0) + 1

        with self._stats_lock:
            stats_copy = dict(self._stats)

        return {
            "stats": stats_copy,
            "categories": categories,
            "memory_count": len(self._memories),
            "history_count": len(self._history),
            "capture_count": self._capture_count,
            "face_count": self._face_count,
            "change_count": self._change_count,
            "text_count": self._text_count,
        }

    def get_stats(self) -> Dict:
        return self.get_memory_stats()

    def compare_images(self, img1, img2) -> Dict:
        try:
            hash1 = self._compute_image_hash(img1)
            hash2 = self._compute_image_hash(img2)

            color1 = self._analyze_colors(img1)
            color2 = self._analyze_colors(img2)

            brightness_diff = abs(color1.avg_brightness - color2.avg_brightness)
            saturation_diff = abs(color1.avg_saturation - color2.avg_saturation)
            diversity_diff = abs(color1.color_diversity - color2.color_diversity)

            similarity = 1.0 - (brightness_diff * 0.4 + saturation_diff * 0.3 + diversity_diff * 0.3)

            has_text1, _, _ = self._detect_text_regions(img1)
            has_text2, _, _ = self._detect_text_regions(img2)

            width1, height1 = img1.size
            width2, height2 = img2.size

            return {
                "similarity": round(max(0.0, min(1.0, similarity)), 3),
                "hash_match": hash1 == hash2,
                "brightness_diff": round(brightness_diff, 3),
                "saturation_diff": round(saturation_diff, 3),
                "diversity_diff": round(diversity_diff, 3),
                "size_match": width1 == width2 and height1 == height2,
                "text_in_both": has_text1 and has_text2,
                "text_in_first_only": has_text1 and not has_text2,
                "text_in_second_only": not has_text1 and has_text2,
            }
        except (OSError, TypeError, ValueError, AttributeError) as e:
            return {"similarity": 0.0, "error": type(e).__name__}

    def describe_scene(self, img, context: str = "") -> str:
        try:
            if not hasattr(img, 'size'):
                return "Unable to analyze image"

            width, height = img.size
            color_info = self._analyze_colors(img)
            has_text, _, _ = self._detect_text_regions(img)
            faces = self._detect_faces_heuristic(img, width, height)

            scene_cat = self._classify_scene(context or "", [], [])
            bright = "bright" if color_info.avg_brightness > 0.6 else "dim" if color_info.avg_brightness < 0.3 else "moderately lit"
            colors_desc = ", ".join([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in color_info.dominant_colors[:3]])

            parts = [f"{width}x{height} {bright} {scene_cat.value} scene"]
            if colors_desc:
                parts.append(f"dominant colors: {colors_desc}")
            if faces > 0:
                parts.append(f"{faces} face(s)")
            if has_text:
                parts.append("text visible")

            return _sanitize_content(". ".join(parts) + ".", _MAX_DESCRIPTION)
        except (OSError, TypeError, ValueError, AttributeError) as e:
            return f"Analysis unavailable: {type(e).__name__}"

    def extract_text(self, img) -> str:
        has_text, extracted, _ = self._detect_text_regions(img)
        if has_text:
            self._inc_stat("texts_extracted")
        return extracted

    def get_change_history(self, limit: int = 50) -> List[Dict]:
        limit = max(1, min(200, limit))
        with self._lock:
            changes = [m.to_dict() for m in self._memories if m.change_from_previous is not None]
        return changes[-limit:]

    def get_faces_history(self, limit: int = 50) -> List[Dict]:
        limit = max(1, min(200, limit))
        with self._lock:
            faces = [m.to_dict() for m in self._memories if m.faces_detected > 0]
        return faces[-limit:]

    def capture_screenshot(self, description: str = "") -> VisualMemory:
        """Capture and analyze the current screen."""
        try:
            import subprocess
            from PIL import Image
            result = subprocess.run(
                ["import", "-window", "root", "-", "png:-"],
                capture_output=True, timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout:
                img = Image.open(io.BytesIO(result.stdout))
                img.load()
                return self.analyze_image(img, description=description or "Screenshot capture", source="screenshot")
        except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
            pass
        try:
            import subprocess
            from PIL import Image
            result = subprocess.run(
                ["screencapture", "-x", "-"],
                capture_output=True, timeout=10,
            )
            if result.stdout:
                img = Image.open(io.BytesIO(result.stdout))
                img.load()
                return self.analyze_image(img, description=description or "Screenshot capture", source="screenshot")
        except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
            pass
        return VisualMemory(source="screenshot", description="Screenshot unavailable", importance=0.0)

    def start_streaming(self, interval_seconds: float = 5.0, max_frames: int = 3600) -> bool:
        """Begin periodic camera capture loop. Returns True if started."""
        if self._streaming:
            return False
        self._streaming = True
        self._stream_interval = max(1.0, min(60.0, interval_seconds))
        self._stream_max_frames = max(10, min(86400, max_frames))
        self._stream_frame_count = 0
        self._stream_thread = threading.Thread(
            target=self._streaming_loop, daemon=True, name="vision-stream"
        )
        self._stream_thread.start()
        self._inc_stat("streams_started")
        return True

    def _streaming_loop(self) -> None:
        """Background thread for periodic frame capture."""
        while self._streaming and self._stream_frame_count < self._stream_max_frames:
            try:
                self.capture_frame()
                self._stream_frame_count += 1
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Vision stream frame error: %s", type(e).__name__)
            time.sleep(self._stream_interval)
        self._streaming = False

    def stop_streaming(self) -> int:
        """Stop streaming. Returns frames captured."""
        self._streaming = False
        frames = self._stream_frame_count
        self._inc_stat("streams_stopped")
        return frames

    def is_streaming(self) -> bool:
        return self._streaming

    def get_streaming_status(self) -> Dict:
        return {
            "streaming": self._streaming,
            "interval_seconds": self._stream_interval,
            "frames_captured": self._stream_frame_count,
            "max_frames": self._stream_max_frames,
        }



    def consolidate_to_dream(self, dream_state=None, limit: int = 50) -> Dict:
        """Consolidate recent visual observations into DreamState memories.
        Each visual memory becomes a DreamMemory with scene context, objects,
        and features for overnight processing and insight generation."""
        result = {"consolidated": 0, "skipped": 0, "dream_memories": []}
        if dream_state is None:
            return result
        with self._lock:
            recent = list(self._memories)[-limit:]
        for mem in recent:
            if mem.importance < 0.3:
                result["skipped"] += 1
                continue
            try:
                dm = dream_state.add_memory(
                    content=f"Visual: {mem.description}",
                    domain="visual",
                    importance=mem.importance,
                    tags=mem.tags + [mem.scene_category.value]
                )
                result["consolidated"] += 1
                result["dream_memories"].append({
                    "dream_id": dm.memory_id if hasattr(dm, 'memory_id') else str(dm),
                    "source_visual_id": mem.memory_id,
                    "scene": mem.scene_category.value,
                    "importance": mem.importance
                })
            except (AttributeError, RuntimeError, ValueError, TypeError):
                result["skipped"] += 1
        self._inc_stat("consolidations_run")
        return result

    def forget(self, memory_id: str) -> bool:
        memory_id = _sanitize_strict(memory_id, 50)
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.memory_id == memory_id:
                    self._memories.remove(m)
                    return True
        return False

    def clear_memory(self) -> int:
        with self._lock:
            count = len(self._memories)
            self._memories.clear()
        self._previous_hash = None
        self._previous_description = None
        self._previous_objects = []
        return count