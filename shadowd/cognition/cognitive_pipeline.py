#!/usr/bin/env python3
"""
CognitivePipeline - Multi-Module Cognitive Orchestrator

Runs text through multiple cognitive modules in sequence and synthesizes
a combined analysis result.

Pipeline: ThoughtSense -> EmotionalIntelligence -> MetacognitionEngine -> MoralReasoning

Thread-safe, bounded history, input sanitization.
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.cognitive_pipeline")

_MAX_PIPELINE_RESULTS = 1000
_MAX_INPUT_LENGTH = 10000
_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_DICT_KEYS = 50
_MAX_DICT_VALUE_LENGTH = 200


def _sanitize_pipeline_input(text: str, max_len: int = _MAX_INPUT_LENGTH) -> str:
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_len]


def _truncate_dict(d, max_keys=_MAX_DICT_KEYS):
    if d is None:
        return None
    result = {}
    for k, v in list(d.items())[:max_keys]:
        if isinstance(v, str):
            result[k] = v[:_MAX_DICT_VALUE_LENGTH]
        elif isinstance(v, (dict, list)):
            result[k] = copy.deepcopy(v)
        else:
            result[k] = v
    return result


@dataclass
class PipelineResult:
    result_id: str = ""
    text: str = ""
    context: str = ""
    thought_analysis: Optional[Dict] = None
    emotion_state: Optional[Dict] = None
    bias_detection: Optional[Dict] = None
    moral_evaluation: Optional[Dict] = None
    overall_assessment: str = ""
    overall_confidence: float = 0.5
    modules_used: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.result_id:
            self.result_id = f"pipe_{uuid.uuid4().hex[:12]}"
        self.overall_confidence = max(0.0, min(1.0, self.overall_confidence))
        self.thought_analysis = _truncate_dict(self.thought_analysis)
        self.emotion_state = _truncate_dict(self.emotion_state)
        self.bias_detection = _truncate_dict(self.bias_detection)
        self.moral_evaluation = _truncate_dict(self.moral_evaluation)

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "text": self.text.strip()[:200],
            "context": self.context.strip()[:200],
            "thought_analysis": _truncate_dict(self.thought_analysis),
            "emotion_state": _truncate_dict(self.emotion_state),
            "bias_detection": _truncate_dict(self.bias_detection),
            "moral_evaluation": _truncate_dict(self.moral_evaluation),
            "overall_assessment": self.overall_assessment.strip()[:500],
            "overall_confidence": round(self.overall_confidence, 3),
            "modules_used": list(self.modules_used),
            "timestamp": self.timestamp,
        }


_ETHICAL_KEYWORDS = [
    "should", "must", "ought", "right", "wrong", "fair", "unfair",
    "ethical", "moral", "duty", "obligation", "justice", "harm",
    "benefit", "consequence", "principle", "value", "virtue",
]

_ETHICAL_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _ETHICAL_KEYWORDS]


class CognitivePipeline:
    def __init__(self, thought_sense=None, emotional_intelligence=None,
                 metacognition=None, moral_reasoning=None):
        self._thought_sense = thought_sense
        self._emotional_intelligence = emotional_intelligence
        self._metacognition = metacognition
        self._moral_reasoning = moral_reasoning
        self._results: deque = deque(maxlen=_MAX_PIPELINE_RESULTS)
        self._lock = threading.Lock()
        self._total_pipelines = 0
        self._stats = {
            "pipelines_run": 0,
            "avg_confidence": 0.5,
            "ethical_evaluations": 0,
        }
        self._stats_lock = threading.Lock()

    def process(self, text: str, context: str = "") -> Dict:
        text = _sanitize_pipeline_input(text)
        context = _sanitize_pipeline_input(context, max_len=1000)
        modules_used = []
        thought_analysis = None
        emotion_state = None
        bias_detection = None
        moral_evaluation = None

        if self._thought_sense is not None:
            try:
                result = self._thought_sense.analyze(text)
                thought_analysis = result.to_dict()
                modules_used.append("thought_sense")
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("ThoughtSense failed: %s", type(e).__name__)
                thought_analysis = None

        if self._emotional_intelligence is not None:
            try:
                result = self._emotional_intelligence.detect_emotion(text, context=context)
                emotion_state = result.to_dict()
                modules_used.append("emotional_intelligence")
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("EmotionalIntelligence failed: %s", type(e).__name__)
                emotion_state = None

        if self._metacognition is not None:
            try:
                biases = self._metacognition.detect_biases(text)
                bias_detection = {
                    "biases_found": len(biases),
                    "biases": [b.to_dict() for b in biases[:5]],
                }
                modules_used.append("metacognition")
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("MetacognitionEngine failed: %s", type(e).__name__)
                bias_detection = None

        lower = text.lower()
        has_ethical_content = any(
            p.search(lower)
            for p in _ETHICAL_PATTERNS
        )

        if has_ethical_content and self._moral_reasoning is not None:
            try:
                evaluation = self._moral_reasoning.evaluate(text)
                moral_evaluation = evaluation.to_dict()
                modules_used.append("moral_reasoning")
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("MoralReasoning failed: %s", type(e).__name__)
                moral_evaluation = None

        confidence = 0.5
        assessment_parts = []
        n_modules = len(modules_used)

        if thought_analysis:
            intents = thought_analysis.get("intents", [])
            primary_intent = intents[0].get("intent", "unknown") if intents else "unknown"
            urgency = thought_analysis.get("urgency", "low")
            cognitive_load = thought_analysis.get("cognitive_load", 50)
            assessment_parts.append(f"Intent: {primary_intent}, urgency: {urgency}, cognitive load: {cognitive_load}")
            confidence += 0.1

        if emotion_state:
            primary = emotion_state.get("primary", "neutral")
            intensity = emotion_state.get("intensity", 0.5)
            assessment_parts.append(f"Emotional state: {primary} (intensity {intensity:.1f})")
            confidence += 0.1

        if bias_detection and bias_detection.get("biases_found", 0) > 0:
            n_biases = bias_detection["biases_found"]
            assessment_parts.append(f"Cognitive biases detected: {n_biases}")
            confidence -= 0.05 * n_biases
        elif bias_detection:
            assessment_parts.append("No cognitive biases detected")
            confidence += 0.05

        if moral_evaluation:
            score = moral_evaluation.get("overall_score", 0.5)
            assessment_parts.append(f"Ethical evaluation score: {score:.2f}")
            confidence += 0.05

        confidence = max(0.1, min(1.0, confidence))
        if n_modules > 0:
            confidence = confidence * (0.5 + 0.5 * n_modules / 4)

        overall_assessment = "; ".join(assessment_parts) if assessment_parts else "Minimal analysis available"

        result = PipelineResult(
            text=text,
            context=context,
            thought_analysis=_truncate_dict(thought_analysis),
            emotion_state=_truncate_dict(emotion_state),
            bias_detection=_truncate_dict(bias_detection),
            moral_evaluation=_truncate_dict(moral_evaluation),
            overall_assessment=overall_assessment,
            overall_confidence=confidence,
            modules_used=modules_used,
        )

        with self._lock:
            if len(self._results) >= _MAX_PIPELINE_RESULTS:
                evicted = self._results[0]
                logger.warning("Pipeline results deque full (%d), evicting oldest result %s", _MAX_PIPELINE_RESULTS, evicted.result_id)
            self._results.append(result)
            with self._stats_lock:
                self._stats["pipelines_run"] += 1
            self._total_pipelines += 1
            if moral_evaluation is not None:
                with self._stats_lock:
                    self._stats["ethical_evaluations"] += 1
            total = self._total_pipelines
            old_avg = self._stats["avg_confidence"]
            self._stats["avg_confidence"] = old_avg + (confidence - old_avg) / total

        return result.to_dict()

    def get_pipeline_stats(self) -> Dict:
        with self._lock:
            return {
                "stats": dict(self._stats),
                "results_stored": len(self._results),
                "recent_results": [r.to_dict() for r in list(self._results)[-5:]],
            }