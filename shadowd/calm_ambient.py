#!/usr/bin/env python3
"""
Enhanced Ambient Intelligence - Based on Calm Technology & Ubiquitous Computing Research

Implements:
- Calm Technology principles (Amber Case)
- Ubiquitous Computing patterns (Mark Weiser)
- Privacy-preserving ambient sensing
- Attention management
- Graceful degradation
"""

import asyncio
import copy
import re as _re
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json
import logging

logger = logging.getLogger("shadowd.calm_ambient")

_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 10000) -> str:
    if not text:
        return text
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not text:
        return text
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


@dataclass
class AttentionEvent:
    """An event requiring user attention (from Calm Technology)."""
    event_type: str
    message: str
    priority: str  # "periphery", "focus", "critical"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action_required: bool = False
    expires_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Current context (from Ubiquitous Computing)."""
    timestamp: datetime
    time_of_day: str  # morning, afternoon, evening, night
    day_of_week: str
    location: Optional[str] = None
    activity: Optional[str] = None
    people_present: List[str] = field(default_factory=list)
    devices_active: List[str] = field(default_factory=list)
    apps_active: List[str] = field(default_factory=list)
    physiological: Dict = field(default_factory=dict)  # heart rate, stress, etc.
    environmental: Dict = field(default_factory=dict)  # temperature, noise, light


class AttentionManager:
    """
    Manages user attention (Calm Technology Principle: Attention is sacred).
    
    Implements:
    - Attention queue with priorities
    - Do not disturb modes
    - Glanceability scoring
    - Graceful degradation
    """
    
    _MAX_CRITICAL = 20
    _MAX_FOCUS = 10
    
    def __init__(self, max_focus_items: int = 3):
        self.max_focus_items = max_focus_items
        self._lock = threading.Lock()
        
        # Three attention levels (Calm Technology)
        self.periphery: deque = deque(maxlen=10)  # Background awareness
        self.focus: List[AttentionEvent] = []     # Needs attention soon
        self.critical: List[AttentionEvent] = []  # Immediate attention
        
        # Attention budget (how much we can interrupt)
        self.daily_interruptions = 0
        self.max_daily_interruptions = 20
        self._last_reset_date = datetime.now(timezone.utc).date()
        
        # Do not disturb mode
        self.dnd_mode = False
        self.dnd_until: Optional[datetime] = None
        
        # Sacred times (no interruptions)
        self.sacred_times = {
            "sleep": (23, 7),   # 11pm - 7am
            "dinner": (18, 19), # 6pm - 7pm
        }
    
    def can_interrupt(self, priority: str = "focus") -> bool:
        """Check if we can interrupt (Calm Tech principle: respect attention)."""
        with self._lock:
            # Check sacred times
            hour = datetime.now(timezone.utc).hour
            for period, (start, end) in self.sacred_times.items():
                # Handle overnight periods (e.g., 23-7 means 11pm to 7am)
                if start > end:  # Overnight (e.g., 23-7)
                    if hour >= start or hour < end:
                        return False
                else:  # Normal period (e.g., 18-19)
                    if start <= hour < end:
                        return False
            
            # Check DND mode
            if self.dnd_mode:
                if self.dnd_until and datetime.now(timezone.utc) < self.dnd_until:
                    return False
                else:
                    self.dnd_mode = False  # DND expired
            
            # Check interruption budget
            if self.daily_interruptions >= self.max_daily_interruptions:
                return False
            
            # Only focus and critical can interrupt
            return priority in ["focus", "critical"]
    
    def add_event(self, event: AttentionEvent) -> bool:
        """Add an attention event. Returns True if added."""
        event.message = _sanitize_content(event.message)
        
        # Periphery events don't require interruption check (always allowed)
        if event.priority == "periphery":
            with self._lock:
                self.periphery.append(event)
            return True
        
        # Check if we should interrupt (for focus and critical only)
        if not self.can_interrupt(event.priority):
            return False
        
        with self._lock:
            if event.priority == "focus":
                if len(self.focus) >= self._MAX_FOCUS:
                    oldest = self.focus.pop(0)
                    self.periphery.append(oldest)
                self.focus.append(event)
                self.daily_interruptions += 1
                return True
            elif event.priority == "critical":
                if len(self.critical) >= self._MAX_CRITICAL:
                    self.critical = self.critical[-self._MAX_CRITICAL + 1:]
                self.critical.append(event)
                self.daily_interruptions += 1
                return True
        
        return False
    
    def set_dnd(self, hours: int = 1):
        """Set Do Not Disturb mode."""
        self.dnd_mode = True
        self.dnd_until = datetime.now(timezone.utc) + timedelta(hours=hours)
    
    def get_glanceable_summary(self) -> Dict:
        """Get a glanceable summary (Calm Tech: glanceable)."""
        with self._lock:
            return {
                "critical": len(self.critical),
                "focus": len(self.focus),
                "periphery": len(self.periphery),
                "can_interrupt": True,  # Safe default
                "daily_interruptions": f"{self.daily_interruptions}/{self.max_daily_interruptions}"
            }
    
    def prune_expired(self):
        """Remove expired events."""
        now = datetime.now(timezone.utc)
        
        with self._lock:
            self.critical = [
                e for e in self.critical
                if e.expires_at is None or e.expires_at >= now
            ]
            
            self.focus = [
                e for e in self.focus
                if e.expires_at is None or e.expires_at >= now
            ]
            
            # Reset daily interruptions at date change
            today = datetime.now(timezone.utc).date()
            if today != self._last_reset_date:
                self.daily_interruptions = 0
                self._last_reset_date = today


class CalmAmbient:
    _MAX_DETECTED_PATTERNS = 500
    _MAX_SUGGESTIONS = 100
    """
    Calm Ambient Intelligence Service.
    
    Principles from Calm Technology:
    1. Attention is sacred
    2. Technology should be invisible (periphery)
    3. Provide glanceable, then detailed views
    4. Respect human limitations
    5. Graceful degradation
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self.attention = AttentionManager()
        
        # Passive observation (silent, no interruptions)
        self.observation_buffer: deque = deque(maxlen=1000)
        
        # Context tracking
        self.current_context: Optional[ContextSnapshot] = None
        self.context_history: deque = deque(maxlen=100)
        
        # Pattern detection (passive)
        self.detected_patterns: Dict[str, int] = defaultdict(int)
        
        # Proactive suggestions queue (only when relevant)
        self.suggestions: List[Dict] = []
        
        # Callbacks for observations
        self.observation_callbacks: List[Callable] = []
    
    def observe(self, event: Dict) -> bool:
        """
        Observe an event (passive, invisible).
        
        Returns True if added to observation buffer.
        Never interrupts the user.
        """
        # Sanitize content strings
        if "content" in event and isinstance(event["content"], str):
            event["content"] = _sanitize_content(event["content"])
        
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        with self._lock:
            self.observation_buffer.append(event)
            self._detect_pattern(event)
            self._update_context(event)
        
        for callback in self.observation_callbacks[:50]:
            try:
                callback(event)
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                logger.error("Observation callback error: %s", type(e).__name__)
        
        return True
    
    def _detect_pattern(self, event: Dict):
        """Detect patterns silently (passive learning). CALLED WITH self._lock HELD."""
        event_type = event.get("type", "unknown")
        
        self.detected_patterns[f"type:{event_type}"] += 1
        
        hour = datetime.now(timezone.utc).hour
        self.detected_patterns[f"hour:{hour}"] += 1
        
        if "activity" in event:
            self.detected_patterns[f"activity:{event['activity']}"] += 1
        
        if hour > 9 and hour < 17:
            self.detected_patterns["time:business_hours"] += 1
        elif hour >= 22 or hour < 6:
            self.detected_patterns["time:sleep_hours"] += 1
        
        if len(self.detected_patterns) > self._MAX_DETECTED_PATTERNS:
            sorted_patterns = sorted(self.detected_patterns.items(), key=lambda x: x[1])
            self.detected_patterns = defaultdict(int, sorted_patterns[-self._MAX_DETECTED_PATTERNS:])
    
    def _update_context(self, event: Dict):
        """Update current context."""
        now = datetime.now(timezone.utc)
        
        # Create snapshot
        snapshot = ContextSnapshot(
            timestamp=now,
            time_of_day=self._get_time_of_day(now.hour),
            day_of_week=now.strftime("%A"),
            location=event.get("location"),
            activity=event.get("activity"),
            people_present=event.get("people", []),
            devices_active=event.get("devices", []),
            apps_active=event.get("apps", []),
            physiological=event.get("physiological", {}),
            environmental=event.get("environmental", {})
        )
        
        self.current_context = snapshot
        self.context_history.append(snapshot)
    
    def _get_time_of_day(self, hour: int) -> str:
        """Get time of day category."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def suggest(self, suggestion: str, confidence: float = 0.8, relevant_now: bool = True) -> bool:
        """
        Suggest something proactively.
        
        Only adds if:
        1. Confidence is high
        2. Relevant to current context
        3. Can interrupt (attention budget)
        """
        suggestion = _sanitize_content(suggestion)
        if confidence < 0.7:
            # Low confidence: keep in periphery
            return self._add_to_periphery(suggestion)
        
        if not relevant_now:
            # Not relevant: keep for later
            return self._add_to_periphery(suggestion)
        
        # Create attention event
        event = AttentionEvent(
            event_type="suggestion",
            message=suggestion,
            priority="focus" if confidence > 0.9 else "periphery",
            action_required=True
        )
        
        result = self.attention.add_event(event)
        
        with self._lock:
            if result:
                self.suggestions.append({"message": suggestion, "confidence": confidence})
                if len(self.suggestions) > self._MAX_SUGGESTIONS:
                    self.suggestions = self.suggestions[-self._MAX_SUGGESTIONS:]
        
        return result
    
    def _add_to_periphery(self, message: str) -> bool:
        """Add to periphery (background awareness)."""
        event = AttentionEvent(
            event_type="info",
            message=message,
            priority="periphery",
            action_required=False
        )
        self.attention.periphery.append(event)
        return True
    
    def get_glanceable_context(self) -> Dict:
        """
        Get a glanceable summary (Calm Tech principle).
        
        One glance should give the main information.
        """
        attention_summary = self.attention.get_glanceable_summary()
        
        with self._lock:
            context_summary = {
                "time_of_day": self.current_context.time_of_day if self.current_context else "unknown",
                "day_of_week": self.current_context.day_of_week if self.current_context else "unknown",
                "activity": self.current_context.activity if self.current_context else None
            }
            
            pattern_summary = dict(list(self.detected_patterns.items())[-5:])
            
            observations_today = len([
                e for e in self.observation_buffer
                if self._is_today(e.get("timestamp"))
            ])
        
        return {
            "attention": attention_summary,
            "context": context_summary,
            "recent_patterns": pattern_summary,
            "observations_today": observations_today,
        }
    
    def _is_today(self, timestamp_str: str) -> bool:
        """Check if timestamp is today."""
        try:
            ts = datetime.fromisoformat(timestamp_str)
            return ts.date() == datetime.now(timezone.utc).date()
        except (ValueError, TypeError):
            return False
    
    def get_detailed_context(self) -> Dict:
        """Get detailed context (when user asks for details)."""
        with self._lock:
            current_ctx = copy.deepcopy(self.current_context.__dict__) if self.current_context else {}
            critical = [
                {"message": e.message, "timestamp": e.timestamp.isoformat()}
                for e in list(self.attention.critical)
            ]
            focus = [
                {"message": e.message, "timestamp": e.timestamp.isoformat()}
                for e in list(self.attention.focus)
            ]
            periphery = [
                {"message": e.message, "timestamp": e.timestamp.isoformat()}
                for e in list(self.attention.periphery)
            ]
            obs_count = len(self.observation_buffer)
            patterns = dict(list(self.detected_patterns.items())[:20])
            suggestions = [copy.deepcopy(s) if isinstance(s, dict) else s for s in self.suggestions[-10:]]
        return {
            "current_context": current_ctx,
            "attention_queue": {
                "critical": critical,
                "focus": focus,
                "periphery": periphery,
            },
            "observations": obs_count,
            "patterns": patterns,
            "suggestions": suggestions,
        }
    
    async def run_background_monitoring(self, interval: int = 60):
        """Run background monitoring (invisible, periphery)."""
        logger.info("Starting calm ambient monitoring (invisible mode)")
        
        while True:
            try:
                # Prune expired attention events
                self.attention.prune_expired()
                
                # Check for context changes (passive)
                if self.current_context:
                    # Detect context transitions
                    if len(self.context_history) > 1:
                        prev = self.context_history[-2]
                        curr = self.current_context
                        
                        # Location change (passive observation)
                        if prev.location != curr.location and curr.location:
                            self._add_to_periphery(f"Location changed to {curr.location}")
                        
                        # Activity change (passive observation)
                        if prev.activity != curr.activity and curr.activity:
                            self._add_to_periphery(f"Activity changed to {curr.activity}")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error("Error in background monitoring: %s", type(e).__name__)
                await asyncio.sleep(5)
    
    def register_observer(self, callback: Callable):
        """Register a callback for observations (passive)."""
        if len(self.observation_callbacks) >= 50:
            return  # Cap at 50 callbacks
        self.observation_callbacks.append(callback)


# Integration with Shadow AI

class AmbientShadowInterface:
    """Interface between Shadow AI and Ambient service."""
    
    def __init__(self, ambient: CalmAmbient):
        self.ambient = ambient
        
        # Register as observer
        ambient.register_observer(self._handle_observation)
    
    def _handle_observation(self, event: Dict):
        """Handle passive observation (no interruption)."""
        # This would integrate with Shadow AI's memory
        pass
    
    def ask_context(self) -> Dict:
        """User asks for context (pull-based)."""
        return self.ambient.get_glanceable_context()
    
    def ask_details(self) -> Dict:
        """User asks for details (pull-based)."""
        return self.ambient.get_detailed_context()
    
    def observe(self, event: Dict):
        """Passive observation (push-based)."""
        return self.ambient.observe(event)
    
    def suggest(self, suggestion: str, confidence: float = 0.8) -> bool:
        """Proactive suggestion (respects attention)."""
        return self.ambient.suggest(suggestion, confidence)
    
    def set_dnd(self, hours: int = 1):
        """Set Do Not Disturb."""
        self.ambient.attention.set_dnd(hours)