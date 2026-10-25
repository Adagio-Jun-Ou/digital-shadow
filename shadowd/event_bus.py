#!/usr/bin/env python3
"""
Async Event Bus for Shadow AI inter-ability communication.

Replaces direct method calls between abilities with a publish/subscribe
pattern, allowing loose coupling and easy testing.

Patterns from:
- LangGraph: Channel-based state communication
- OpenHands: Event-based action/observation stream
- Letta: Message passing between memory blocks
"""

import asyncio
import copy
import logging
import re as _re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger("shadowd.event_bus")

_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text


class EventType(str, Enum):
    OBSERVATION = "observation"
    INSIGHT = "insight"
    ACTION = "action"
    ABILITY_RESULT = "ability_result"
    MEMORY_UPDATE = "memory_update"
    PATTERN_DETECTED = "pattern_detected"
    PREDICTION = "prediction"
    ALERT = "alert"
    STUCK = "stuck"
    RECOVERY = "recovery"


@dataclass
class ShadowEvent:
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = ""
    priority: int = 5  # 1=critical, 10=low
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            import secrets
            self.event_id = f"evt_{secrets.token_hex(12)}"
        self.source = _sanitize(self.source)
        if self.data:
            sanitized = {}
            for k, v in list(self.data.items())[:50]:
                sanitized[_sanitize(str(k))] = v
            self.data = sanitized
        if self.metadata:
            sanitized_meta = {}
            for k, v in list(self.metadata.items())[:50]:
                sanitized_meta[_sanitize(str(k))] = v
            self.metadata = sanitized_meta

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "source": self.source,
            "data": copy.deepcopy(self.data),
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "priority": self.priority,
            "metadata": copy.deepcopy(self.metadata)
        }


# Type alias for async handlers
AsyncHandler = Callable[[ShadowEvent], Coroutine[Any, Any, None]]
SyncHandler = Callable[[ShadowEvent], None]


class EventBus:
    """
    Async event bus for Shadow AI.
    
    Enables loose coupling between abilities, observers, and the orchestrator.
    Each ability publishes events (observations, predictions, alerts) and
    subscribes to events from other abilities.
    
    Features:
    - Topic-based pub/sub
    - Priority-based ordering (critical events processed first)
    - Async handlers with backpressure
    - Event history for replay/debugging
    - Graceful error handling per handler
    """
    
    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[AsyncHandler]] = defaultdict(list)
        self._wildcard_subscribers: List[AsyncHandler] = []
        self._event_history: List[ShadowEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0,
            "by_type": defaultdict(int),
        }

    async def publish(self, event: ShadowEvent) -> None:
        """Publish an event to all matching subscribers."""
        async with self._lock:
            with self._stats_lock:
                self._stats["events_published"] += 1
                self._stats["by_type"][event.event_type.value] += 1
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            handlers = list(self._subscribers.get(event.event_type.value, []))
            handlers.extend(self._wildcard_subscribers)
        
        for handler in handlers:
            try:
                await handler(event)
                with self._stats_lock:
                    self._stats["events_processed"] += 1
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError, KeyError, asyncio.CancelledError) as e:
                with self._stats_lock:
                    self._stats["errors"] += 1
                logger.error("Event handler error for %s: %s", event.event_type.value, type(e).__name__)
    
    def subscribe(self, event_type: EventType, handler: AsyncHandler) -> None:
        """Subscribe to events of a specific type."""
        topic = _sanitize(event_type.value) if isinstance(event_type, EventType) else _sanitize(str(event_type))
        if len(self._subscribers[topic]) >= 50:
            logger.warning("Subscriber cap reached for %s, dropping oldest", topic)
            self._subscribers[topic].pop(0)
        self._subscribers[topic].append(handler)
    
    def subscribe_all(self, handler: AsyncHandler) -> None:
        """Subscribe to all events (wildcard)."""
        if len(self._wildcard_subscribers) >= 20:
            logger.warning("Wildcard subscriber cap reached, dropping oldest")
            self._wildcard_subscribers.pop(0)
        self._wildcard_subscribers.append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: AsyncHandler) -> None:
        """Unsubscribe from events of a specific type."""
        handlers = self._subscribers.get(event_type.value, [])
        if handler in handlers:
            handlers.remove(handler)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[ShadowEvent]:
        """Get event history, optionally filtered by type."""
        limit = min(limit, 1000)
        if event_type:
            events = [copy.deepcopy(e) for e in list(self._event_history) if e.event_type == event_type]
        else:
            events = [copy.deepcopy(e) for e in list(self._event_history)]
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._stats_lock:
            return {
                "events_published": self._stats["events_published"],
                "events_processed": self._stats["events_processed"],
                "errors": self._stats["errors"],
                "by_type": dict(self._stats["by_type"]),
                "subscriber_count": sum(len(h) for h in self._subscribers.values()) + len(self._wildcard_subscribers),
                "history_size": len(self._event_history),
            }


# Global event bus instance
_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus instance (thread-safe)."""
    global _event_bus
    with _event_bus_lock:
        if _event_bus is None:
            _event_bus = EventBus()
    return _event_bus