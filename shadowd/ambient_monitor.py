#!/usr/bin/env python3
"""
Ambient Intelligence Monitor - LifeBuddy-style continuous monitoring

Implements:
- Continuous observation
- Pattern detection
- Anomaly recognition
- Proactive insights
- Commitment tracking
"""

import asyncio
import copy
import os
import re
import tempfile
import uuid
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import logging

from .context_protocol import ContextProtocol, ContextType

logger = logging.getLogger("shadowd.ambient_monitor")

_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 10000) -> str:
    if not text:
        return text
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not text:
        return text
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


@dataclass
class Commitment:
    """A commitment (promise, deadline, meeting)."""
    id: str
    content: str
    person: Optional[str] = None
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending, completed, missed
    source: str = "unknown"  # where it was captured
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "person": self.person,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "source": self.source,
            "metadata": copy.deepcopy(self.metadata)
        }


@dataclass
class Pattern:
    """A detected pattern."""
    pattern_type: str  # frequency, sequence, correlation
    description: str
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "confidence": round(self.confidence, 3),
            "metadata": copy.deepcopy(self.metadata)
        }


@dataclass
class Anomaly:
    """A detected anomaly."""
    anomaly_type: str
    description: str
    severity: str  # low, medium, high, critical
    detected_at: datetime
    context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "context": copy.deepcopy(self.context)
        }


@dataclass
class Insight:
    """A proactive insight."""
    insight_type: str  # commitment, pattern, anomaly, prediction
    priority: str  # low, medium, high, critical
    title: str
    message: str
    action: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "insight_type": self.insight_type,
            "priority": self.priority,
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class AmbientMonitor:
    """
    Continuous ambient intelligence monitor.
    
    Monitors:
    - Commitments (promises, deadlines)
    - Patterns (recurring events, behaviors)
    - Anomalies (unusual activity)
    - Context changes (location, time, people)
    
    Generates proactive insights (LifeBuddy-style).
    """
    
    def __init__(
        self, 
        context_protocol: ContextProtocol,
        data_dir: Path = None
    ):
        self.context = context_protocol
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "ambient")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Caps
        self._MAX_PATTERNS = 500
        self._MAX_EVENT_COUNTS = 200
        self._MAX_PEOPLE = 500
        self._MAX_PER_PERSON_ENCOUNTERS = 100
        
        # Tracking
        self.commitments: List[Commitment] = []
        self.patterns: Dict[str, Pattern] = {}
        self._pattern_order: List[str] = []
        self.anomalies: List[Anomaly] = []
        self.insights: List[Insight] = []
        
        # Event tracking (for pattern detection)
        self.event_history: List[Dict] = []
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.seqences: Dict[str, int] = defaultdict(int)  # "A->B" sequences
        
        # Person tracking
        self.people_encounters: Dict[str, List[datetime]] = defaultdict(list)
        self._people_order: List[str] = []
        
        # Load persisted data
        self._load_state()
    
    def _load_state(self):
        """Load persisted state."""
        state_file = self.data_dir / "ambient_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                
                # Load commitments
                self.commitments = [
                    Commitment(
                        id=c["id"],
                        content=c["content"],
                        person=c.get("person"),
                        deadline=datetime.fromisoformat(c["deadline"]) if c.get("deadline") else None,
                        created_at=datetime.fromisoformat(c["created_at"]),
                        status=c.get("status", "pending"),
                        source=c.get("source", "unknown"),
                        metadata=c.get("metadata", {})
                    )
                    for c in data.get("commitments", [])
                ]
                
                # Load patterns
                self.patterns = {
                    p_name: Pattern(
                        pattern_type=p["pattern_type"],
                        description=p["description"],
                        occurrences=p["occurrences"],
                        first_seen=datetime.fromisoformat(p["first_seen"]),
                        last_seen=datetime.fromisoformat(p["last_seen"]),
                        confidence=p.get("confidence", 0.0),
                        metadata=p.get("metadata", {})
                    )
                    for p_name, p in data.get("patterns", {}).items()
                }
                
                logger.info(f"Loaded {len(self.commitments)} commitments, {len(self.patterns)} patterns")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
                logger.error("Failed to load state: %s", type(e).__name__)
    
    def _save_state(self):
        """Persist state to disk."""
        state_file = self.data_dir / "ambient_state.json"
        try:
            data = {
                "commitments": [
                    {
                        "id": c.id,
                        "content": c.content,
                        "person": c.person,
                        "deadline": c.deadline.isoformat() if c.deadline else None,
                        "created_at": c.created_at.isoformat(),
                        "status": c.status,
                        "source": c.source,
                        "metadata": c.metadata
                    }
                    for c in self.commitments
                ],
                "patterns": {
                    p_name: {
                        "pattern_type": p.pattern_type,
                        "description": p.description,
                        "occurrences": p.occurrences,
                        "first_seen": p.first_seen.isoformat(),
                        "last_seen": p.last_seen.isoformat(),
                        "confidence": round(p.confidence, 3),
                        "metadata": p.metadata
                    }
                    for p_name, p in self.patterns.items()
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            state_file = self.data_dir / "ambient_state.json"
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.data_dir), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(state_file))
            except (OSError, json.JSONDecodeError, ValueError):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to save state: %s", type(e).__name__)
    
    async def observe(self, event: Dict):
        """
        Observe an event and process it.
        
        Routes to:
        - Commitment detection
        - Pattern detection
        - Anomaly detection
        - Insight generation
        """
        # Sanitize content
        if "content" in event and isinstance(event["content"], str):
            event["content"] = _sanitize_content(event["content"])
        if "type" in event and isinstance(event["type"], str):
            event["type"] = _sanitize(event["type"], 100)
        
        event_time = datetime.now(timezone.utc)
        event_type = event.get("type", "unknown")
        
        with self._lock:
            # Add to history
            self.event_history.append({
                **event,
                "observed_at": event_time.isoformat()
            })
            
            # Keep last 1000 events
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-1000:]
            
            # Update event counts
            self.event_counts[event_type] += 1
            # Evict oldest event counts if over cap
            if len(self.event_counts) > self._MAX_EVENT_COUNTS:
                oldest_key = next(iter(self.event_counts))
                del self.event_counts[oldest_key]
        
        # Detect commitments
        if event_type in ["message", "email", "calendar", "note"]:
            await self._detect_commitment(event)
        
        # Detect patterns
        await self._detect_pattern(event)
        
        # Detect anomalies
        await self._detect_anomaly(event)
        
        # Generate insights
        await self._generate_insights(event)
        
        # Save state periodically
        with self._lock:
            should_save = len(self.event_history) % 50 == 0
        if should_save:
            self._save_state()
    
    async def _detect_commitment(self, event: Dict):
        """Detect commitments from event content."""
        content = str(event.get("content", "")).lower()
        
        # Commitment indicators
        commitment_phrases = [
            "i will", "i'll", "i promise", "i commit",
            "i'll call", "i'll email", "i'll send",
            "i'll be there", "i'll meet", "i'll come",
            "tomorrow", "next week", "on monday", "on tuesday",
            "by", "deadline", "due", "promise"
        ]
        
        if any(re.search(r'\b' + re.escape(phrase) + r'\b', content) for phrase in commitment_phrases):
            safe_content = str(event.get("content", ""))[:200]
            safe_content = _sanitize_content(safe_content)
            commitment = Commitment(
                id=f"commit_{uuid.uuid4().hex[:12]}",
                content=safe_content,
                source=_sanitize(event.get("type", "unknown"), 100),
                metadata={"original_event": event}
            )
            
            with self._lock:
                self.commitments.append(commitment)
                if len(self.commitments) > 1000:
                    self.commitments = self.commitments[-1000:]
            
            log_content = commitment.content[:50].replace('\n', ' ').replace('\r', '')
            logger.info("Detected commitment: %s", log_content)
            
            # Generate insight
            insight = Insight(
                insight_type="commitment",
                priority="high",
                title="New Commitment Detected",
                message=f"Detected: {commitment.content[:100]}",
                action="Track this commitment?",
                confidence=0.7,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
            )
            with self._lock:
                self.insights.append(insight)
                if len(self.insights) > 500:
                    self.insights = self.insights[-500:]
    async def _detect_pattern(self, event: Dict):
        """Detect recurring patterns."""
        event_type = _sanitize(event.get("type", "unknown"), 100)
        event_time = datetime.now(timezone.utc)
        
        with self._lock:
            # Pattern: Frequent event type
            
            if event_type in self.event_counts:
                # Check if this event type occurs frequently
                hour = event_time.hour
                pattern_key = f"{event_type}_hour_{hour}"
                
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = Pattern(
                        pattern_type="frequency",
                        description=f"{event_type} often occurs around {hour}:00",
                        occurrences=1,
                        first_seen=event_time,
                        last_seen=event_time
                    )
                    self._pattern_order.append(pattern_key)
                    # Evict oldest pattern if over cap
                    while len(self._pattern_order) > self._MAX_PATTERNS:
                        old_key = self._pattern_order.pop(0)
                        self.patterns.pop(old_key, None)
                else:
                    self.patterns[pattern_key].occurrences += 1
                    self.patterns[pattern_key].last_seen = event_time
                    self.patterns[pattern_key].confidence = min(
                        1.0, 
                        self.patterns[pattern_key].occurrences / 10.0
                    )
            
            # Pattern: Sequences (A -> B)
            if len(self.event_history) >= 2:
                prev_event = self.event_history[-2]
                prev_type = prev_event.get("type", "unknown")
                current_type = event_type
                
                sequence_key = f"{prev_type}->{current_type}"
                self.seqences[sequence_key] += 1
                
                if self.seqences[sequence_key] >= 3:
                    p_name = f"sequence_{sequence_key}"
                    if p_name not in self.patterns:
                        self.patterns[p_name] = Pattern(
                            pattern_type="sequence",
                            description=f"{prev_type} is often followed by {current_type}",
                            occurrences=self.seqences[sequence_key],
                            first_seen=event_time,
                            last_seen=event_time
                        )
                        self._pattern_order.append(p_name)
                        while len(self._pattern_order) > self._MAX_PATTERNS:
                            old_key = self._pattern_order.pop(0)
                            self.patterns.pop(old_key, None)
    
    async def _detect_anomaly(self, event: Dict):
        """Detect anomalies (unusual events)."""
        event_type = _sanitize(event.get("type", "unknown"), 100)
        event_time = datetime.now(timezone.utc)
        
        with self._lock:
            history_snapshot = list(self.event_history[-100:])
        
        # Anomaly: Event at unusual time
        hour = event_time.hour
        
        # If this event type usually happens at different times
        usual_hours = defaultdict(int)
        for e in history_snapshot:
            if e.get("type") == event_type:
                try:
                    e_time = datetime.fromisoformat(e.get("observed_at", ""))
                except (ValueError, TypeError):
                    continue
                usual_hours[e_time.hour] += 1
        
        if usual_hours:
            most_common_hour = max(usual_hours.items(), key=lambda x: x[1])[0]
            
            if abs(hour - most_common_hour) > 6:  # More than 6 hours difference
                anomaly = Anomaly(
                    anomaly_type="unusual_time",
                    description=f"{event_type} at unusual time ({hour}:00 vs usual {most_common_hour}:00)",
                    severity="low",
                    detected_at=event_time,
                    context=copy.deepcopy(event)
                )
                with self._lock:
                    self.anomalies.append(anomaly)
                    if len(self.anomalies) > 100:
                        self.anomalies = self.anomalies[-100:]
    
    async def _generate_insights(self, event: Dict):
        """Generate proactive insights."""
        event_time = datetime.now(timezone.utc)
        
        with self._lock:
            pending_commitments = [c for c in self.commitments if c.status == "pending"]
            recent_pattern_insights_len = len([
                i for i in self.insights
                if i.insight_type == "pattern" 
                and (event_time - i.created_at).total_seconds() < 3600
            ])
        
        # Check commitments
        for commitment in pending_commitments:
            # If deadline within 24 hours
            if commitment.deadline:
                deadline = commitment.deadline
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                time_until = deadline - event_time
                
                if 0 < time_until.total_seconds() < 86400:
                    insight = Insight(
                        insight_type="commitment",
                        priority="critical",
                        title="Upcoming Deadline",
                        message=f"{commitment.content[:100]}",
                        action=f"Complete before {commitment.deadline.strftime('%H:%M')}",
                        confidence=1.0
                    )
                    with self._lock:
                        self.insights.append(insight)
                        if len(self.insights) > 500:
                            self.insights = self.insights[-500:]
        
        # Check for strong patterns
        with self._lock:
            patterns_snapshot = list(self.patterns.values())
        
        for pattern in patterns_snapshot:
            if pattern.confidence > 0.8 and pattern.pattern_type == "frequency":
                # Only suggest if not recently suggested
                if recent_pattern_insights_len < 3:
                    insight = Insight(
                        insight_type="pattern",
                        priority="medium",
                        title="Pattern Detected",
                        message=pattern.description,
                        confidence=pattern.confidence,
                        expires_at=event_time + timedelta(hours=2)
                    )
                    with self._lock:
                        self.insights.append(insight)
                        if len(self.insights) > 500:
                            self.insights = self.insights[-500:]
    
    def track_person_encounter(self, person: str, location: str = None):
        """Track encounter with a person."""
        if not person or not isinstance(person, str):
            return
        person = re.sub(r'[^\w\s-]', '', person).strip()[:100]
        if not person:
            return
        now = datetime.now(timezone.utc)
        
        self.people_encounters[person].append(now)
        if person not in self._people_order:
            self._people_order.append(person)
            # Evict oldest person if over cap
            while len(self._people_order) > self._MAX_PEOPLE:
                old_person = self._people_order.pop(0)
                self.people_encounters.pop(old_person, None)
        
        # Keep last 100 encounters per person (already capped)
        if len(self.people_encounters[person]) > self._MAX_PER_PERSON_ENCOUNTERS:
            self.people_encounters[person] = self.people_encounters[person][-self._MAX_PER_PERSON_ENCOUNTERS // 2:]
        
        # Detect pattern
        if len(self.people_encounters[person]) >= 3:
            # Check for time patterns
            hours = [e.hour for e in self.people_encounters[person]]
            common_hour = max(set(hours), key=hours.count)
            
            if hours.count(common_hour) / len(hours) > 0.5:
                pattern_name = f"person_{person}_time"
                self.patterns[pattern_name] = Pattern(
                    pattern_type="social",
                    description=f"{person} often encountered around {common_hour}:00",
                    occurrences=len(self.people_encounters[person]),
                    first_seen=self.people_encounters[person][0],
                    last_seen=now,
                    confidence=hours.count(common_hour) / len(hours)
                )
                if pattern_name not in self._pattern_order:
                    self._pattern_order.append(pattern_name)
                while len(self._pattern_order) > self._MAX_PATTERNS:
                    old_key = self._pattern_order.pop(0)
                    self.patterns.pop(old_key, None)
    
    def get_pending_commitments(self, hours: int = 24) -> List[Commitment]:
        """Get commitments due within X hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        with self._lock:
            return [copy.deepcopy(c) for c in self.commitments if c.status == "pending" and c.deadline and c.deadline <= cutoff and c.deadline >= now]
    
    def get_strong_patterns(self, min_confidence: float = 0.7) -> List[Pattern]:
        """Get patterns above confidence threshold."""
        with self._lock:
            return [copy.deepcopy(p) for p in self.patterns.values() if p.confidence >= min_confidence]
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Anomaly]:
        """Get anomalies from last X hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            return [copy.deepcopy(a) for a in self.anomalies if a.detected_at >= cutoff]
    
    def get_active_insights(self) -> List[Insight]:
        """Get non-expired insights."""
        now = datetime.now(timezone.utc)
        with self._lock:
            return [copy.deepcopy(i) for i in self.insights if i.expires_at is None or i.expires_at >= now]
    
    def mark_commitment_completed(self, commitment_id: str):
        """Mark a commitment as completed."""
        commitment_id = _sanitize(commitment_id, 100)
        with self._lock:
            for c in self.commitments:
                if c.id == commitment_id:
                    c.status = "completed"
                    break
        self._save_state()
    
    def get_status(self) -> Dict:
        """Get monitor status."""
        with self._lock:
            commitments_total = len(self.commitments)
            commitments_pending = len([c for c in self.commitments if c.status == "pending"])
            commitments_completed = len([c for c in self.commitments if c.status == "completed"])
            patterns_total = len(self.patterns)
            anomalies_total = len(self.anomalies)
            people_count = len(self.people_encounters)
            events_count = len(self.event_history)
        return {
            "commitments": {
                "total": commitments_total,
                "pending": commitments_pending,
                "completed": commitments_completed,
            },
            "patterns": {
                "total": patterns_total,
                "strong": len(self.get_strong_patterns()),
            },
            "anomalies": {
                "total": anomalies_total,
                "recent": len(self.get_recent_anomalies()),
            },
            "insights": {
                "active": len(self.get_active_insights()),
            },
            "people_tracked": people_count,
            "events_observed": events_count,
        }
    
    async def run_continuous_monitoring(self, interval: int = 60):
        """Run continuous ambient monitoring."""
        logger.info("Starting ambient monitoring (interval: %ds)", interval)
        
        while True:
            try:
                # Get current context
                if self.context.current_snapshot:
                    # Process temporal context
                    temporal = self.context.current_snapshot.temporal_context
                    
                    # Check for time-based patterns
                    await self._check_time_patterns(temporal)
                
                # Clean up expired insights
                now = datetime.now(timezone.utc)
                with self._lock:
                    self.insights = [
                        i for i in self.insights
                        if i.expires_at is None or i.expires_at >= now
                    ]
                
                # Save state periodically
                with self._lock:
                    count = len(self.event_history)
                if count % 10 == 0:
                    self._save_state()
                
                await asyncio.sleep(interval)
            
            except (OSError, ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.error("Error in ambient monitoring: %s", type(e).__name__)
                await asyncio.sleep(5)
    
    async def _check_time_patterns(self, temporal_context: Dict):
        """Check for time-based patterns and generate insights."""
        hour = temporal_context.get("hour", 0)
        day_of_week = temporal_context.get("day_of_week", "")
        
        # Morning check
        if hour == 9 and day_of_week not in ["Saturday", "Sunday"]:
            # Check for untracked commitments
            pending = len([c for c in self.commitments if c.status == "pending"])
            if pending > 3:
                self.insights.append(Insight(
                    insight_type="commitment",
                    priority="medium",
                    title="Many Pending Commitments",
                    message=f"You have {pending} pending commitments",
                    action="Would you like to review them?",
                    confidence=1.0,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=2)
                ))
        
        # Evening check
        if hour == 18:
            # Review day patterns
            pass