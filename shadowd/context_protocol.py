#!/usr/bin/env python3
"""
Context Protocol System - Unified context sharing between abilities (MCP-style)

Implements Model Context Protocol patterns:
- Unified context provider
- Structured context schemas
- Context inheritance
- Real-time context updates
"""

import asyncio
import copy
import os
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger("shadowd.context_protocol")


class ContextType(Enum):
    """Types of context that can be shared."""
    DEVICE = "device"           # IoT/device data
    LOCATION = "location"       # GPS, wifi-based location
    ACTIVITY = "activity"       # Current user activity
    TEMPORAL = "temporal"       # Time-based context
    SOCIAL = "social"          # People/presence data
    MEMORY = "memory"          # Historical context
    ENVIRONMENT = "environment" # Physical environment
    DIGITAL = "digital"         # Digital context (files, apps, etc.)


@dataclass
class ContextSource:
    """A source of context information."""
    name: str
    context_type: ContextType
    provider: Callable
    update_interval: int = 60  # seconds
    last_update: datetime = None
    cached_data: Dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ContextSnapshot:
    """A snapshot of all context at a point in time."""
    timestamp: datetime
    device_context: Dict = field(default_factory=dict)
    location_context: Dict = field(default_factory=dict)
    activity_context: Dict = field(default_factory=dict)
    temporal_context: Dict = field(default_factory=dict)
    social_context: Dict = field(default_factory=dict)
    memory_context: Dict = field(default_factory=dict)
    environment_context: Dict = field(default_factory=dict)
    digital_context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "device": dict(self.device_context) if self.device_context else {},
            "location": dict(self.location_context) if self.location_context else {},
            "activity": dict(self.activity_context) if self.activity_context else {},
            "temporal": dict(self.temporal_context) if self.temporal_context else {},
            "social": dict(self.social_context) if self.social_context else {},
            "memory": dict(self.memory_context) if self.memory_context else {},
            "environment": dict(self.environment_context) if self.environment_context else {},
            "digital": dict(self.digital_context) if self.digital_context else {},
        }
    
    def get_unified_context(self) -> Dict:
        """Get a unified context for LLM prompts."""
        return {
            "when": dict(self.temporal_context) if self.temporal_context else {},
            "where": dict(self.location_context) if self.location_context else {},
            "who": dict(self.social_context) if self.social_context else {},
            "what": dict(self.activity_context) if self.activity_context else {},
            "devices": {**dict(self.device_context or {}), **dict(self.digital_context or {})},
            "environment": dict(self.environment_context) if self.environment_context else {},
            "history": dict(self.memory_context) if self.memory_context else {},
        }


_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return _SANITIZE_STRICT_RE.sub('', text)


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return _SANITIZE_CONTENT_RE.sub('', text)


class ContextProtocol:
    """
    Model Context Protocol implementation for Shadow AI.
    
    Provides:
    - Structured context management
    - Multi-source context aggregation
    - Real-time context updates
    - Context inheritance for abilities
    """
    
    _MAX_HISTORY = 100
    _MAX_SUBSCRIBERS = 50
    _MAX_SOURCES_PER_TYPE = 20
    _MAX_SOURCE_NAME_LEN = 64
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "context")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        
        self.sources: Dict[ContextType, List[ContextSource]] = {
            ctx_type: [] for ctx_type in ContextType
        }
        
        self.current_snapshot: Optional[ContextSnapshot] = None
        self.context_history: List[ContextSnapshot] = []
        
        self.subscribers: List[Callable] = []
        
        self._init_default_sources()
    
    def _init_default_sources(self):
        """Register default context sources."""
        # Temporal context (always available)
        self.register_source(ContextSource(
            name="time",
            context_type=ContextType.TEMPORAL,
            provider=self._get_temporal_context,
            update_interval=1
        ))
        
        # Activity context (from shadow observations)
        self.register_source(ContextSource(
            name="activity_tracker",
            context_type=ContextType.ACTIVITY,
            provider=self._get_activity_context,
            update_interval=10
        ))
        
        # Digital context (files, apps)
        self.register_source(ContextSource(
            name="digital_tracker",
            context_type=ContextType.DIGITAL,
            provider=self._get_digital_context,
            update_interval=30
        ))
    
    def register_source(self, source: ContextSource):
        """Register a context source."""
        source.name = _sanitize(source.name)[:self._MAX_SOURCE_NAME_LEN]
        with self._lock:
            if len(self.sources[source.context_type]) >= self._MAX_SOURCES_PER_TYPE:
                oldest = self.sources[source.context_type][0]
                self.sources[source.context_type] = self.sources[source.context_type][1:]
                logger.warning(f"Evicted source {oldest.name} from {source.context_type.value}")
            self.sources[source.context_type].append(source)
        logger.info(f"Registered context source: {source.name} ({source.context_type.value})")
    
    async def update_context(self):
        """Update all context sources."""
        snapshot = ContextSnapshot(timestamp=datetime.now(timezone.utc))
        
        with self._lock:
            for context_type in ContextType:
                type_data = {}
                for source in self.sources[context_type]:
                    if source.enabled:
                        try:
                            data = await self._update_source(source)
                            type_data[source.name] = data
                        except Exception as e:
                            logger.error(f"Error updating {source.name}: {type(e).__name__}")
                
                if context_type == ContextType.DEVICE:
                    snapshot.device_context = type_data
                elif context_type == ContextType.LOCATION:
                    snapshot.location_context = type_data
                elif context_type == ContextType.ACTIVITY:
                    snapshot.activity_context = type_data
                elif context_type == ContextType.TEMPORAL:
                    snapshot.temporal_context = type_data
                elif context_type == ContextType.SOCIAL:
                    snapshot.social_context = type_data
                elif context_type == ContextType.MEMORY:
                    snapshot.memory_context = type_data
                elif context_type == ContextType.ENVIRONMENT:
                    snapshot.environment_context = type_data
                elif context_type == ContextType.DIGITAL:
                    snapshot.digital_context = type_data
            
            self.current_snapshot = snapshot
            self.context_history.append(snapshot)
            
            if len(self.context_history) > self._MAX_HISTORY:
                self.context_history = self.context_history[-self._MAX_HISTORY:]
        
        # Notify subscribers outside lock
        for subscriber in list(self.subscribers):
            try:
                await subscriber(copy.deepcopy(snapshot))
            except Exception as e:
                logger.error(f"Error notifying subscriber: {type(e).__name__}")
        
        return snapshot
    
    async def _update_source(self, source: ContextSource) -> Dict:
        """Update a single source."""
        result = source.provider()
        if asyncio.iscoroutine(result):
            result = await result
        
        source.cached_data = result
        source.last_update = datetime.now(timezone.utc)
        return result
    
    # Default context providers
    def _get_temporal_context(self) -> Dict:
        """Get temporal context (time, date, etc.)."""
        now = datetime.now(timezone.utc)
        
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "hour": now.hour,
            "is_weekend": now.weekday() >= 5,
            "is_business_hours": 9 <= now.hour < 18,
            "time_of_day": self._get_time_of_day(now.hour),
            "season": self._get_season(now.month)
        }
    
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
    
    def _get_season(self, month: int) -> str:
        """Get current season (northern hemisphere)."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    async def _get_activity_context(self) -> Dict:
        """Get current activity context."""
        # This would be populated by observers
        return {
            "current_activity": "unknown",
            "active_app": None,
            "idle_time": 0,
            "keyboard_activity": 0,
            "mouse_activity": 0
        }
    
    async def _get_digital_context(self) -> Dict:
        """Get digital context (files, apps, etc.)."""
        return {
            "open_files": [],
            "active_projects": [],
            "recent_commands": [],
            "git_status": None
        }
    
    # Context query methods
    def get_context_for_ability(self, ability_name: str, snapshot: ContextSnapshot = None) -> Dict:
        """
        Get context tailored for a specific ability.
        
        Different abilities need different context:
        - Oracle: temporal, location, social, memory
        - X-Ray: device, environment, location
        - Life Pulse: device, social
        - Perfect Memory: memory, temporal
        - Pattern Sight: memory, activity
        - Social Radar: social, temporal, location
        - Intent Seer: activity, digital, temporal
        - Omni Sense: ALL context
        - Shadow Council: memory, temporal, social
        - Reality Forge: digital, activity
        """
        if snapshot is None:
            snapshot = self.current_snapshot
        
        if snapshot is None:
            return {}
        
        unified = snapshot.get_unified_context()
        
        # Ability-specific context
        ability_contexts = {
            "oracle": {
                "when": unified["when"],
                "where": unified["where"],
                "who": unified["who"],
                "history": unified["history"]
            },
            "x_ray": {
                "where": unified["where"],
                "devices": unified["devices"],
                "environment": unified["environment"]
            },
            "life_pulse": {
                "devices": unified["devices"],
                "who": unified["who"]
            },
            "perfect_memory": {
                "history": unified["history"],
                "when": unified["when"]
            },
            "pattern_sight": {
                "history": unified["history"],
                "what": unified["what"]
            },
            "social_radar": {
                "who": unified["who"],
                "when": unified["when"],
                "where": unified["where"]
            },
            "intent_seer": {
                "what": unified["what"],
                "devices": unified["devices"],
                "when": unified["when"]
            },
            "omni_sense": unified,  # ALL context
            "shadow_council": {
                "history": unified["history"],
                "when": unified["when"],
                "who": unified["who"]
            },
            "reality_forge": {
                "devices": unified["devices"],
                "what": unified["what"]
            }
        }
        
        return copy.deepcopy(ability_contexts.get(ability_name, unified))
    
    def get_context_history(self, context_type: ContextType = None, limit: int = 10) -> List[Dict]:
        """Get context history, optionally filtered by type."""
        limit = max(1, min(limit, 1000))
        if context_type is None:
            return [copy.deepcopy(s.to_dict()) for s in self.context_history[-limit:]]
        
        return [
            copy.deepcopy(s.to_dict()) for s in self.context_history[-limit:]
            if hasattr(s, f"{context_type.value}_context")
        ]
    
    def subscribe(self, callback: Callable):
        """Subscribe to context updates."""
        with self._lock:
            if len(self.subscribers) >= self._MAX_SUBSCRIBERS:
                logger.warning(f"Max subscribers ({self._MAX_SUBSCRIBERS}) reached, dropping oldest")
                self.subscribers.pop(0)
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from context updates."""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous context monitoring."""
        logger.info(f"Starting context monitoring (interval: {interval}s)")
        
        while True:
            try:
                await self.update_context()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in context monitoring: {e}")
                await asyncio.sleep(5)
    
    def save_snapshot(self, snapshot: ContextSnapshot = None):
        """Save a snapshot to disk (atomic with fsync)."""
        if snapshot is None:
            with self._lock:
                snapshot = self.current_snapshot
        
        if snapshot is None:
            return
        
        snapshot_file = self.data_dir / f"snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            tmp = snapshot_file.with_suffix(f".tmp.{os.getpid()}")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(snapshot.to_dict(), f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            except (OSError, TypeError, ValueError):
                os.close(fd)
                raise
            os.replace(str(tmp), str(snapshot_file))
            logger.debug(f"Saved snapshot: {snapshot_file.name}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save snapshot: {type(e).__name__}")
    
    def load_latest_snapshot(self) -> Optional[ContextSnapshot]:
        """Load the most recent snapshot."""
        snapshots = sorted(self.data_dir.glob("snapshot_*.json"), reverse=True)
        
        if not snapshots:
            return None
        
        try:
            data = json.loads(snapshots[0].read_text())
            _ts = data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(_ts) if _ts else datetime.now(timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)
            snapshot = ContextSnapshot(
                timestamp=timestamp,
                device_context=data.get("device", {}),
                location_context=data.get("location", {}),
                activity_context=data.get("activity", {}),
                temporal_context=data.get("temporal", {}),
                social_context=data.get("social", {}),
                memory_context=data.get("memory", {}),
                environment_context=data.get("environment", {}),
                digital_context=data.get("digital", {})
            )
            with self._lock:
                self.current_snapshot = snapshot
            return copy.deepcopy(snapshot)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to load snapshot: {type(e).__name__}")
            return None
    
    def get_status(self) -> Dict:
        """Get context protocol status."""
        with self._lock:
            return {
                "sources": {ctx_type.value: len(sources) for ctx_type, sources in self.sources.items()},
                "current_snapshot": self.current_snapshot is not None,
                "history_size": len(self.context_history),
                "subscribers": len(self.subscribers),
            }


# Integration with Shadow AI
class ContextAwareAbility:
    """Mixin for abilities to get context-aware execution."""
    
    def __init__(self, context_protocol: ContextProtocol):
        self.context_protocol = context_protocol
    
    def get_my_context(self) -> Dict:
        """Get context for this ability."""
        ability_name = getattr(self, "name", self.__class__.__name__.lower())
        return self.context_protocol.get_context_for_ability(ability_name)
    
    def get_unified_context(self) -> Dict:
        """Get unified context."""
        with self.context_protocol._lock:
            if self.context_protocol.current_snapshot:
                return copy.deepcopy(self.context_protocol.current_snapshot.get_unified_context())
        return {}