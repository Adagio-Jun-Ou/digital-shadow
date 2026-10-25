#!/usr/bin/env python3

import copy
import hashlib
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize_content(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text))

def _sanitize_strict(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text))


class LoopType(Enum):
    REPEATED_ACTION = "repeated_action"
    REPEATED_ERROR = "repeated_error"
    MONOLOGUE = "monologue"
    ACTION_OBSERVATION_LOOP = "action_observation_loop"
    CONTEXT_WINDOW_LOOP = "context_window_loop"


@dataclass
class StuckAnalysis:
    is_stuck: bool = False
    loop_type: Optional[LoopType] = None
    repeat_count: int = 0
    pattern_description: str = ""
    recovery_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_stuck": self.is_stuck,
            "loop_type": self.loop_type.value if self.loop_type else None,
            "repeat_count": self.repeat_count,
            "pattern_description": self.pattern_description,
            "recovery_suggestions": list(self.recovery_suggestions),
            "confidence": round(self.confidence, 3),
        }


class ActionType(Enum):
    TOOL_CALL = "tool_call"
    THOUGHT = "thought"
    OBSERVATION = "observation"
    ERROR = "error"


@dataclass
class HistoryEntry:
    action_type: ActionType
    content: str
    tool_name: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def fingerprint(self) -> str:
        parts = [self.action_type.value]
        if self.tool_name:
            parts.append(self.tool_name)
        content_snippet = self.content[:120] if self.content else ""
        parts.append(content_snippet)
        if self.error_message:
            parts.append(self.error_message[:80])
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:10]


class StuckDetector:
    REPEATED_ACTION_THRESHOLD = 3
    REPEATED_ERROR_THRESHOLD = 3
    MONOLOGUE_THRESHOLD = 4
    ACTION_OBSERVATION_LOOP_WINDOW = 6
    CONTEXT_WINDOW_LOOP_THRESHOLD = 2
    CONTEXT_WINDOW_MIN_HISTORY = 2
    MAX_HISTORY = 100

    def __init__(self, max_history: int = 100):
        self.history: List[HistoryEntry] = []
        self.max_history = max_history
        self._lock = threading.Lock()

    def record_action(self, content: str, tool_name: Optional[str] = None) -> HistoryEntry:
        content = _sanitize_content(content)[:50000]
        tool_name = _sanitize_strict(tool_name) if tool_name else None
        entry = HistoryEntry(
            action_type=ActionType.TOOL_CALL,
            content=content,
            tool_name=tool_name,
        )
        with self._lock:
            self._add(entry)
        return entry

    def record_thought(self, content: str) -> HistoryEntry:
        content = _sanitize_content(content)[:50000]
        entry = HistoryEntry(
            action_type=ActionType.THOUGHT,
            content=content,
        )
        with self._lock:
            self._add(entry)
        return entry

    def record_observation(self, content: str) -> HistoryEntry:
        content = _sanitize_content(content)[:50000]
        entry = HistoryEntry(
            action_type=ActionType.OBSERVATION,
            content=content,
        )
        with self._lock:
            self._add(entry)
        return entry

    def record_error(self, error_message: str, context: str = "") -> HistoryEntry:
        error_message = _sanitize_content(error_message)[:5000]
        context = _sanitize_content(context)[:50000]
        entry = HistoryEntry(
            action_type=ActionType.ERROR,
            content=context,
            error_message=error_message,
        )
        with self._lock:
            self._add(entry)
        return entry

    def _add(self, entry: HistoryEntry):
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def analyze(self) -> StuckAnalysis:
        with self._lock:
            checks = [
                self._check_context_window_loop,
                self._check_repeated_error,
                self._check_monologue,
                self._check_action_observation_loop,
                self._check_repeated_action,
            ]
            for check in checks:
                result = check()
                if result.is_stuck:
                    return result
            return StuckAnalysis(is_stuck=False)

    def _check_repeated_action(self) -> StuckAnalysis:
        if len(self.history) < self.REPEATED_ACTION_THRESHOLD:
            return StuckAnalysis()
        recent = self.history[-10:]
        recent_actions = [e for e in recent if e.action_type == ActionType.TOOL_CALL]
        fingerprints = [e.fingerprint for e in recent_actions]
        if not fingerprints:
            return StuckAnalysis()
        counter = Counter(fingerprints)
        most_common_fingerprint, count = counter.most_common(1)[0]
        if count >= self.REPEATED_ACTION_THRESHOLD:
            matched = [e for e in recent_actions if e.fingerprint == most_common_fingerprint]
            tool_info = matched[0].tool_name or "unknown"
            return StuckAnalysis(
                is_stuck=True,
                loop_type=LoopType.REPEATED_ACTION,
                repeat_count=count,
                pattern_description=f"Action '{tool_info}' repeated {count} times with same content",
                recovery_suggestions=[
                    "Try a different approach or tool",
                    "Break the task into smaller steps",
                    "Review previous observations for clues",
                    "Ask the user for clarification",
                ],
                confidence=min(count / 5.0, 1.0),
            )
        return StuckAnalysis()

    def _check_repeated_error(self) -> StuckAnalysis:
        if len(self.history) < self.REPEATED_ERROR_THRESHOLD:
            return StuckAnalysis()
        recent = self.history[-20:]
        errors = [e for e in recent if e.action_type == ActionType.ERROR]
        if len(errors) < self.REPEATED_ERROR_THRESHOLD:
            return StuckAnalysis()
        preceding = []
        for i, entry in enumerate(recent):
            if entry.action_type == ActionType.ERROR:
                for j in range(i - 1, max(i - 4, -1), -1):
                    if recent[j].action_type == ActionType.TOOL_CALL:
                        preceding.append((recent[j], entry))
                        break
        action_error_pairs = Counter()
        for action, error in preceding:
            key = f"{action.fingerprint}:{error.fingerprint}"
            action_error_pairs[key] += 1
        if action_error_pairs:
            best_key, best_count = action_error_pairs.most_common(1)[0]
            if best_count >= self.REPEATED_ERROR_THRESHOLD:
                return StuckAnalysis(
                    is_stuck=True,
                    loop_type=LoopType.REPEATED_ERROR,
                    repeat_count=best_count,
                    pattern_description=f"Same action causing same error {best_count} times",
                    recovery_suggestions=[
                        "Change the action parameters to avoid the error",
                        "Use a different tool or method",
                        "Check prerequisites before retrying",
                        "Inspect the error details to understand root cause",
                    ],
                    confidence=min(best_count / 4.0, 1.0),
                )
        return StuckAnalysis()

    def _check_monologue(self) -> StuckAnalysis:
        if len(self.history) < self.MONOLOGUE_THRESHOLD:
            return StuckAnalysis()
        recent = self.history[-self.MONOLOGUE_THRESHOLD:]
        has_observation = any(e.action_type == ActionType.OBSERVATION for e in recent)
        has_tool_call = any(e.action_type == ActionType.TOOL_CALL for e in recent)
        if not has_observation and not has_tool_call:
            thought_count = sum(1 for e in recent if e.action_type == ActionType.THOUGHT)
            if thought_count >= self.MONOLOGUE_THRESHOLD:
                return StuckAnalysis(
                    is_stuck=True,
                    loop_type=LoopType.MONOLOGUE,
                    repeat_count=thought_count,
                    pattern_description=f"Agent produced {thought_count} thoughts without taking action or receiving observations",
                    recovery_suggestions=[
                        "Take a concrete action using a tool",
                        "Request an observation to ground reasoning",
                        "Make a decision and execute it",
                        "Summarize current understanding and pick next step",
                    ],
                    confidence=min(thought_count / 6.0, 1.0),
                )
        return StuckAnalysis()

    def _check_action_observation_loop(self) -> StuckAnalysis:
        window = self.ACTION_OBSERVATION_LOOP_WINDOW
        if len(self.history) < window:
            return StuckAnalysis()
        recent = self.history[-window:]
        types = tuple(e.action_type for e in recent)
        half = window // 2
        first_half = types[:half]
        second_half = types[half:]
        if first_half == second_half:
            actions_in_half = [e for e in recent[:half] if e.action_type == ActionType.TOOL_CALL]
            tool_info = ", ".join(e.tool_name or "unknown" for e in actions_in_half) if actions_in_half else ""
            desc = f"Action/observation pattern repeating: {' -> '.join(t.value for t in first_half)}"
            if tool_info:
                desc += f" (tools: {tool_info})"
            return StuckAnalysis(
                is_stuck=True,
                loop_type=LoopType.ACTION_OBSERVATION_LOOP,
                repeat_count=2,
                pattern_description=desc,
                recovery_suggestions=[
                    "Change the action parameters to break the cycle",
                    "Try a completely different approach",
                    "Review what has already been tried",
                    "Combine multiple steps into one action",
                ],
                confidence=0.7,
            )
        for period in [2, 3]:
            if len(types) < period * 2:
                continue
            pattern = types[:period]
            matches = 0
            for offset in range(0, len(types) - period + 1, period):
                if types[offset:offset + period] == pattern:
                    matches += 1
            if matches >= 3:
                return StuckAnalysis(
                    is_stuck=True,
                    loop_type=LoopType.ACTION_OBSERVATION_LOOP,
                    repeat_count=matches,
                    pattern_description=f"Detected repeating pattern of period {period}: {' -> '.join(t.value for t in pattern)}",
                    recovery_suggestions=[
                        "Change the action to produce different observations",
                        "Try a different tool or strategy",
                        "Summarize findings and decide on next action",
                    ],
                    confidence=min(0.5 + matches * 0.1, 0.9),
                )
        return StuckAnalysis()

    def _check_context_window_loop(self) -> StuckAnalysis:
        if len(self.history) < self.CONTEXT_WINDOW_MIN_HISTORY:
            return StuckAnalysis()
        recent = self.history[-6:]
        errors = [e for e in recent if e.action_type == ActionType.ERROR]
        context_errors = []
        _context_kw_re = re.compile(r'\b(context|token.limit|max.tokens|context.window|too.long)\b')
        for e in errors:
            if e.error_message and _context_kw_re.search(e.error_message.lower()):
                context_errors.append(e)
        if len(context_errors) >= self.CONTEXT_WINDOW_LOOP_THRESHOLD:
            return StuckAnalysis(
                is_stuck=True,
                loop_type=LoopType.CONTEXT_WINDOW_LOOP,
                repeat_count=len(context_errors),
                pattern_description=f"Context window errors repeated {len(context_errors)} times",
                recovery_suggestions=[
                    "Summarize the conversation to reduce context",
                    "Remove unnecessary history",
                    "Start a fresh session with only essential context",
                    "Break the task into smaller isolated sub-tasks",
                ],
                confidence=0.9,
            )
        return StuckAnalysis()

    def reset(self):
        with self._lock:
            self.history = []

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            type_counts = Counter(e.action_type.value for e in self.history)
            tool_counts = Counter(
                e.tool_name for e in self.history if e.action_type == ActionType.TOOL_CALL and e.tool_name
            )
            return {
                "total_entries": len(self.history),
                "type_counts": dict(type_counts),
                "tool_counts": dict(tool_counts),
                "unique_fingerprints": len(set(e.fingerprint for e in self.history)),
            }