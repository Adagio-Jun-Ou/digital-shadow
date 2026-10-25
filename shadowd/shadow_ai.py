#!/usr/bin/env python3
"""
UNIFIED SHADOW AI
Orchestrates all 10 Shikigami to PROACTIVELY optimize your life.

Not just tools you call - an AI that:
- Watches everything
- Learns continuously
- Proactively advises
- Takes action when needed
- Unified, not fragmented
"""

import asyncio
import copy
import json
import re
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

from shikigami import ShikigamiType

from .event_bus import EventBus, EventType, ShadowEvent, get_event_bus
from .reasoning.stuck_detector import StuckDetector, ActionType as StuckActionType

logger = logging.getLogger("shadowd.shadow_ai")

ABILITY_KEYWORDS = {
    "oracle": ["predict", "will it", "forecast", "future", "expect", "would", "outcome", "probability", "will there be"],
    "x_ray": ["anyone", "someone", "nearby", "around", "detect", "locate", "who's", "who is", "everyone", "everybody", "nobody", "people"],
    "life_pulse": ["is sarah", "is john", "is mike", "is here", "where is", "status", "is there", "present", "available", "arrived", "sarah", "john", "mike"],
    "perfect_memory": ["remember", "when did", "what did", "recall", "history", "past", "before", "happened", "track", "log"],
    "pattern_sight": ["why", "pattern", "patterns", "trend", "cause", "correlate", "recurring", "always", "keeps", "never", "usually", "productivity", "my work", "in my work"],
    "social_radar": ["will they", "behavior", "what will person", "think", "feel", "react", "respond", "believe", "opinion"],
    "intent_seer": ["about to", "upcoming", "soon", "what's next", "coming up", "tonight", "next week", "next month", "schedule", "agenda", "reminder", "appointment", "calendar"],
    "shadow_council": ["should i", "chose", "decide", "choose", "which", "better", "recommend", "advise"],
    "reality_forge": ["create", "build", "make", "set up", "start", "organize", "arrange", "new", "design", "develop", "launch"],
    "habit_tracker": ["habit", "habits", "routine", "streak", "consistency", "every day", "every week", "regular", "recurring activity"],
    "energy_flow": ["energy", "tired", "fatigue", "alert", "wired", "circadian", "peak hour", "low energy"],
    "focus_sight": ["focus", "deep work", "flow", "scattered", "distracted", "concentration", "focused"],
    "connection": ["relationship", "friendship", "bond", "drifted", "haven't talked", "lost touch", "close to", "connected with"],
    "compass": ["goal", "goals", "aim", "objective", "target", "milestone", "aligned with", "off track", "drifting"],
    "hearth": ["where do i", "location pattern", "places i go", "home work", "commute", "travel pattern", "transitions"],
    "time_witness": ["where did the time go", "where did my time", "time budget", "how did i spend", "hours went", "time tracking"],
    "mirror": ["self-talk", "feel about myself", "inner state", "reflection", "introspect", "themes about me", "telling myself"],
    "lantern": ["project", "projects", "stale", "dormant", "active project", "abandoned", "side project", "shelved", "what am i working on", "what was i working on"],
    "tide": ["mood", "feelings", "emotional", "sentiment", "feeling lately", "mood swings", "emotional trend", "how have i felt"],
    "echo": ["what if", "should have", "regret", "counterfactual", "alternative", "would have", "could have", "instead"],
}

_ABILITY_MAP = {
    "oracle": ShikigamiType.ORACLE,
    "x_ray": ShikigamiType.X_RAY,
    "life_pulse": ShikigamiType.LIFE_PULSE,
    "perfect_memory": ShikigamiType.PERFECT_MEMORY,
    "pattern_sight": ShikigamiType.PATTERN_SIGHT,
    "social_radar": ShikigamiType.SOCIAL_RADAR,
    "intent_seer": ShikigamiType.INTENT_SEER,
    "shadow_council": ShikigamiType.SHADOW_COUNCIL,
    "reality_forge": ShikigamiType.REALITY_FORGE,
    "omni_sense": ShikigamiType.OMNI_SENSE,
    "habit_tracker": ShikigamiType.HABIT_TRACKER,
    "energy_flow": ShikigamiType.ENERGY_FLOW,
    "focus_sight": ShikigamiType.FOCUS_SIGHT,
    "connection": ShikigamiType.CONNECTION,
    "compass": ShikigamiType.COMPASS,
    "hearth": ShikigamiType.HEARTH,
    "time_witness": ShikigamiType.TIME_WITNESS,
    "mirror": ShikigamiType.MIRROR,
    "lantern": ShikigamiType.LANTERN,
    "tide": ShikigamiType.TIDE,
    "echo": ShikigamiType.ECHO,
}

_COMMITMENT_WORDS = [
    "promise", "commit", "plan to", "going to",
    "schedule", "agree", "ensure", "guarantee", "intend",
    "volunteer", "offer", "pledge", "resolve", "swear", "vow",
    "handle", "take care", "deal with",
]

_NOUN_WILL_PATTERNS = [
    r'\bfree\s+will\b', r'\blast\s+will\b', r'\bwill\s+power\b',
    r'\bwill\s+of\b', r'\bwith\s+will\b', r'\bagainst\s+(?:my|his|her|their|our|your)\s+will\b',
    r'\bwill\s+test\b', r'\bwill\s+document\b', r'\bgood\s+will\b', r'\bwill\s+and\s+testament\b',
    r'\bstrong\s+will\b', r'\biron\s+will\b', r'\bhis\s+will\b', r'\bher\s+will\b',
    r'\bmy\s+will\b', r'\btheir\s+will\b', r'\bour\s+will\b', r'\byour\s+will\b',
    r'\bthe\s+will\b',
    r'\bwill\s+is\b',
    r'\bwill\s+is\s+\w+\b',
    r'\bwill\s+remains?\b',
    r'\bwill\s+was\b',
    r'\bwill\s+appears?\b',
    r'\bwill\s+seems?\b',
    r'\bwill\s+looks?\b',
    r'\bwill\s+is\s+legal\b',
    r'\bwill\s+remains?\b',
    r'\bwill\s+was\b',
]

_QUESTION_STARTERS = [
    r'^will\s+(?:it|they|he|she|we|you|there|this|that|the|sarah|john|mike|i)\b',
    r'^is\s+', r'^are\s+', r'^do\s+', r'^does\s+', r'^did\s+',
    r'^can\s+', r'^could\s+', r'^would\s+', r'^should\s+',
    r'^shall\s+', r'^may\s+', r'^might\s+', r'^must\s+',
    r'^has\s+', r'^have\s+', r'^had\s+',
    r'^what\b', r'^when\b', r'^where\b', r'^who\b', r'^whom\b',
    r'^why\b', r'^how\b', r'^which\b',
]

_CONTRACTIONS = [r"\b\w+'ll\b", r"\b\w+'d\b"]


def _detect_abilities(question: str) -> List[str]:
    q = question.lower()
    abilities = []
    for ability, keywords in ABILITY_KEYWORDS.items():
        for w in keywords:
            if ' ' in w:
                if w in q:
                    abilities.append(ability)
                    break
            else:
                if re.search(r'\b' + re.escape(w) + r'\b', q):
                    abilities.append(ability)
                    break
    _personal_commitment = re.search(r'\b(?:i|we|they|you|he|she)\s+will\s+(?:be|have|do|go|come|arrive|attend|meet|join|start|finish|complete)\b', q)
    if _personal_commitment:
        if "intent_seer" not in abilities:
            abilities.append("intent_seer")
    elif re.search(r'\bwill\s+[a-z]', q) and "oracle" not in abilities:
        if not re.search(r'\bgoing\s+to\b', q):
            if not any(re.search(p, q) for p in _NOUN_WILL_PATTERNS):
                abilities.append("oracle")
    if re.search(r'\btomorrow\b', q):
        if re.search(r'\b(?:schedule|plan|reminder|appointment|calendar|agenda|commit|meeting|next|what\'s)\b', q):
            if "intent_seer" not in abilities:
                abilities.append("intent_seer")
    if re.search(r'\bnext\b', q) and not re.search(r'\bnext\s+(?:week|month)\b', q):
        if "intent_seer" not in abilities:
            abilities.append("intent_seer")
    if "intent_seer" in abilities and ("oracle" in abilities or "pattern_sight" in abilities):
        if not re.search(r'\b(?:schedule|plan|reminder|appointment|calendar|agenda|commit|meeting)\b', q):
            abilities.remove("intent_seer")
    if re.search(r'\b(?:schedule|organize|arrange)\s+(?:a|an|the|some)?\s*\w', q):
        if "reality_forge" not in abilities:
            abilities.append("reality_forge")
    if re.search(r'\b(?:was|were|had)\s+(?:planning|going|intending)\b', q):
        if "intent_seer" in abilities:
            abilities.remove("intent_seer")
    elif re.search(r'\b(?:planning|going|intending)\s+to\b', q) and "intent_seer" not in abilities:
        abilities.append("intent_seer")
    if re.search(r"\b(?:i|we|they|you|he|she)'ll\b", q) and "intent_seer" not in abilities:
        abilities.append("intent_seer")
    if re.search(r'\bwhere\b', q):
        if "x_ray" not in abilities:
            abilities.append("x_ray")
        if "life_pulse" not in abilities:
            abilities.append("life_pulse")
    if not abilities:
        abilities.append("omni_sense")
    return list(dict.fromkeys(abilities))


def detect_relevant_abilities(question: str) -> List["ShikigamiType"]:
    """Map a natural-language question to the relevant ShikigamiType list.

    Module-level so callers (e.g. the /shikigami/council endpoint) can pick a
    relevant ability subset without constructing a ShadowAI instance. Mirrors
    the instance method ``ShadowAI._detect_relevant_abilities`` and shares the
    same keyword routing covering all 21 abilities.
    """
    return [_ABILITY_MAP.get(a, ShikigamiType.OMNI_SENSE) for a in _detect_abilities(question)]


@dataclass
class Insight:
    """A proactive insight from Shadow AI."""
    priority: str  # "critical", "high", "medium", "low"
    ability: str   # Which Shikigami generated it
    message: str
    action: str    # What to do
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ShadowAI:
    """
    UNIFIED SHADOW - One AI, 10 Abilities, Proactive Optimization.
    
    Philosophy:
    - Don't wait to be asked
    - Surface what matters
    - Learn what works
    - Optimize continuously
    """
    
    def __init__(self, shikigami_registry, mind_engine, data_dir=None):
        self.registry = shikigami_registry
        self.mind = mind_engine
        self.observations: List[Dict] = []
        self.people: Dict = {}
        self.decisions: List[Dict] = []
        self.patterns: Dict = {}
        self.insights: List[Insight] = []
        self._lock = threading.Lock()
        
        # Event bus for inter-ability communication
        self.event_bus = get_event_bus()
        
        # Stuck detector for proactive monitoring
        self.stuck_detector = StuckDetector()
        
        # Load persisted data
        self.data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_data()
        
        # Configuration
        self.config = {
            "notify_threshold": "medium",
            "learning_rate": 0.1,
            "observation_window": 3600,
            "proactive_interval": 60,
        }
    
    def _load_data(self):
        """Load persisted data."""
        for name, attr in [("people", "people"), ("decisions", "decisions"), ("patterns", "patterns"), ("observations", "observations")]:
            path = self.data_dir / f"{name}.json"
            try:
                if path.exists():
                    setattr(self, attr, json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", name, type(e).__name__)
                setattr(self, attr, {} if name not in ("decisions", "observations") else [])
    
    def _save_data(self):
        """Persist data with atomic writes."""
        import tempfile
        import os
        for name, attr in [("people", "people"), ("decisions", "decisions"), ("patterns", "patterns"), ("observations", "observations")]:
            path = self.data_dir / f"{name}.json"
            try:
                data = getattr(self, attr)
                payload = json.dumps(data[-1000:] if isinstance(data, list) else data, indent=2, default=str)
                tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.data_dir), suffix=".json.tmp")
                try:
                    with os.fdopen(tmp_fd, 'w') as f:
                        f.write(payload)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, str(path))
                except (OSError, ValueError):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except (OSError, TypeError, ValueError, json.JSONDecodeError, AttributeError) as e:
                logger.error("Failed to save %s: %s", name, type(e).__name__)
    
    # ═════════════════════════════════════════════════════════════════
    # CORE: OBSERVE
    # ═════════════════════════════════════════════════════════════════
    
    async def observe(self, event: Dict):
        """
        Every observation feeds ALL 10 abilities simultaneously.
        This is how we achieve unity.
        """
        if not isinstance(event, dict):
            logger.warning("observe() called with non-dict: %s", type(event).__name__)
            return
        MAX_OBS_SIZE = 10000
        _SANITIZE_VAL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        for key in list(event.keys())[:20]:
            val = event[key]
            if isinstance(val, str) and len(val) > MAX_OBS_SIZE:
                event[key] = _SANITIZE_VAL_RE.sub('', val[:MAX_OBS_SIZE])
            elif isinstance(val, str):
                event[key] = _SANITIZE_VAL_RE.sub('', val)

        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self.observations.append(event)
            
            # Keep last 1000
            if len(self.observations) > 1000:
                self.observations = self.observations[-1000:]

            # Keep last 200 insights
            if len(self.insights) > 200:
                self.insights = self.insights[-200:]
        
        # Bridge observations into shikigami registry so all 21 abilities
        # (Perfect Memory, Pattern Sight, Intent Seer, etc.) have context
        if self.registry and hasattr(self.registry, 'observations'):
            try:
                self.registry.observations.append(event)
                with self.registry._lock:
                    if len(self.registry.observations) > 5000:
                        self.registry.observations = self.registry.observations[-5000:]
            except (AttributeError, RuntimeError, TypeError):
                pass
        
        # Record in stuck detector
        event_type = event.get("type", "unknown")
        _SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        raw_content = json.dumps(event.get('content', {}))[:100]
        safe_content = _SANITIZE_RE.sub('', raw_content)
        self.stuck_detector.record_observation(f"{event_type}: {safe_content}")
        
        # Publish event on the bus
        try:
            await self.event_bus.publish(ShadowEvent(
                event_type=EventType.OBSERVATION,
                source="shadow_ai",
                data=event,
                priority=3,
            ))
        except (ValueError, RuntimeError, AttributeError, ConnectionError, OSError) as e:
            logger.debug("Event bus publish error: %s", type(e).__name__)
        
        # Trigger unified analysis
        await self._analyze_event(event)
    
    def _add_insight(self, insight: Insight):
        with self._lock:
            self.insights.append(insight)
            if len(self.insights) > 200:
                self.insights = self.insights[-200:]

    async def _analyze_event(self, event: Dict):
        """
        One event triggers analysis from ALL perspectives:
        1. Oracle: What will happen next?
        2. Pattern: What does this correlate with?
        3. Intent: What action follows this?
        4. Memory: Is this a commitment?
        5. Social: Who is mentioned?
        """
        # Check for commitments (Perfect Memory)
        if self._is_commitment(event):
            insight = Insight(
                priority="high",
                ability="perfect_memory",
                message=f"Commitment detected: {str(event.get('content', ''))[:100]}",
                action="Track this promise",
                confidence=0.7
            )
            self._add_insight(insight)
        
        # Check for patterns (Pattern Sight)
        if len(self.observations) > 10:
            pattern = await self._detect_pattern(event)
            if pattern:
                self._add_insight(pattern)
        
        # Check stuck detector
        stuck_analysis = self.stuck_detector.analyze()
        if stuck_analysis.is_stuck:
            insight = Insight(
                priority="critical",
                ability="pattern_sight",
                message=f"Stuck loop detected: {stuck_analysis.pattern_description}",
                action=stuck_analysis.recovery_suggestions[0] if stuck_analysis.recovery_suggestions else "Try a different approach",
                confidence=stuck_analysis.confidence,
            )
            self._add_insight(insight)
            try:
                await self.event_bus.publish(ShadowEvent(
                    event_type=EventType.STUCK,
                    source="stuck_detector",
                    data=stuck_analysis.to_dict(),
                    priority=1,
                ))
            except (ValueError, RuntimeError, AttributeError, ConnectionError, OSError):
                pass    
    _SANITIZE_COMMIT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def _is_commitment(self, event: Dict) -> bool:
        """Detect if event contains a commitment using evolved v10 algorithm."""
        raw_content = str(event.get("content", ""))[:500]
        content = self._SANITIZE_COMMIT_RE.sub('', raw_content).lower()
        event_type = str(event.get("type", "")).lower()
        combined = f"{event_type} {content}"
        stripped = combined.strip()
        if stripped.endswith('?'):
            return False
        if any(re.search(p, combined) for p in _QUESTION_STARTERS):
            return False
        if any(re.search(c, combined) for c in _CONTRACTIONS):
            return True
        for w in _COMMITMENT_WORDS:
            if w == "going to":
                if re.search(r'\bgoing\s+to\s+[a-z]', combined):
                    return True
            elif re.search(r'\b' + re.escape(w) + r'\b', combined):
                return True
        if re.search(r'\bwill\s+[a-z]', combined):
            if not any(re.search(p, combined) for p in _NOUN_WILL_PATTERNS):
                return True
        return False
    
    async def _detect_pattern(self, event: Dict) -> Optional[Insight]:
        """Detect emerging patterns using evolved v10 algorithm."""
        event_type = event.get("type", "unknown")
        observations = self.observations
        n = len(observations)
        min_count = 3
        matching = [o for o in observations if o.get("type") == event_type]
        count = len(matching)
        if count < min_count:
            return None
        half = max(n // 2, 1)
        first_half = sum(1 for o in observations[:half] if o.get("type") == event_type)
        second_half = sum(1 for o in observations[half:] if o.get("type") == event_type)
        first_rate = first_half / half
        second_rate = second_half / max(n - half, 1)
        if second_rate > first_rate * 1.1:
            trend = "increasing"
        elif first_rate > second_rate * 1.1:
            trend = "decreasing"
        else:
            trend = "stable"
        proportion = count / max(n, 1)
        confidence = min(0.4 * min(count / 10.0, 1.0) + 0.6 * proportion, 0.95)
        confidence = max(confidence, min(count / 20.0, 0.95))
        confidence = round(min(confidence, 0.95), 3)
        return Insight(
            priority="medium",
            ability="pattern_sight",
            message=f"Pattern: {event_type} ({count}x, {trend})",
            action=f"Consider automating {event_type}",
            confidence=confidence
        )
    
    # ═════════════════════════════════════════════════════════════════
    # CORE: PROACTIVE MONITOR
    # ═════════════════════════════════════════════════════════════════
    
    async def run_proactive_monitor(self):
        """
        Background task that continuously:
        1. Synthesizes all 10 abilities
        2. Detects what needs attention
        3. Generates proactive insights
        """
        self._stop_event = asyncio.Event()
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.config["proactive_interval"])
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass
            
            try:
                context = {"observations": self.observations[-50:]}
                insights = await self._synthesize(context)
                critical = [i for i in insights if i.priority in ["critical", "high"]]
                
                if critical:
                    logger.info("Proactive insights: %d critical/high priority", len(critical))
                    with self._lock:
                        self.insights.extend(critical)
                        if len(self.insights) > 200:
                            self.insights = self.insights[-200:]
                    self._save_data()
                
            except (ValueError, RuntimeError, AttributeError, TypeError, OSError) as e:
                logger.error("Proactive monitor error: %s", type(e).__name__)
    
    async def stop(self):
        """Stop the proactive monitor and clean up resources."""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self.mind:
            await self.mind.close()
        if hasattr(self, 'registry') and hasattr(self.registry, 'close'):
            await self.registry.close()
    
    async def _synthesize(self, context: Dict) -> List[Insight]:
        """
        Synthesize ALL 10 abilities into unified insights.
        
        Instead of calling abilities individually, combine:
        - WHERE you are (X-Ray)
        - WHO is nearby (Life Pulse)
        - WHAT you're doing (Observations)
        - WHEN patterns occur (Pattern Sight)
        - WHAT you promised (Perfect Memory)
        - WHAT will happen (Oracle + Intent)
        - WHAT matters (Omni Sense)
        """
        insights = []
        
        # Get environment snapshot (async)
        try:
            devices = await self.registry._scan_bt()
        except (OSError, RuntimeError, subprocess.SubprocessError, asyncio.TimeoutError):
            devices = []
        try:
            devices_wifi = await self.registry._scan_wifi()
            networks = devices_wifi
        except (OSError, RuntimeError, subprocess.SubprocessError, asyncio.TimeoutError):
            networks = []
        
        # Identify people present
        present = []
        for name, data in self.people.items():
            for d in devices:
                if d["mac"] in data.get("devices", []):
                    present.append(name)
        
        # Recent activity
        recent = self.observations[-10:] if self.observations else []
        activity_types = [o.get("type", "?") for o in recent]
        
        # UNIFIED PICTURE
        # 1. Time-based insight
        hour = datetime.now(timezone.utc).hour
        if hour < 6:
            insights.append(Insight(
                priority="low",
                ability="omni_sense",
                message="Late night/early morning activity",
                action="Consider rest optimization",
                confidence=0.8
            ))
        elif hour >= 9 and hour <= 17:
            insights.append(Insight(
                priority="medium",
                ability="intent_seer",
                message=f"Work hours - activity: {activity_types[-1] if activity_types else 'unknown'}",
                action="Optimize for productivity",
                confidence=0.7
            ))
        
        # 2. People insight
        _SANITIZE_NAME_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]')
        present_sanitized = [_SANITIZE_NAME_RE.sub('', n) for n in present]
        if present:
            insights.append(Insight(
                priority="medium",
                ability="life_pulse",
                message=f"{', '.join(present_sanitized)} is nearby",
                action="Consider interaction opportunities",
                confidence=0.9
            ))
        
        # 3. Pattern insight
        if len(self.observations) > 20:
            by_type = {}
            for o in self.observations:
                t = o.get("type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
            
            top_activity = max(by_type.items(), key=lambda x: x[1])
            insights.append(Insight(
                priority="low",
                ability="pattern_sight",
                message=f"Top activity: {top_activity[0]} ({top_activity[1]} times)",
                action=f"Automate or optimize {top_activity[0]}",
                confidence=0.6
            ))
        
        # 4. Commitment insight
        commitments = [o for o in self.observations if self._is_commitment(o)]
        if commitments:
            insights.append(Insight(
                priority="high",
                ability="perfect_memory",
                message=f"{len(commitments)} pending commitments detected",
                action="Review and track promises",
                confidence=0.8
            ))
        
        # 5. Decision insight
        if self.decisions:
            recent_decisions = []
            for d in self.decisions:
                try:
                    ts = datetime.fromisoformat(d["timestamp"]).replace(tzinfo=timezone.utc)
                    if ts > datetime.now(timezone.utc) - timedelta(days=7):
                        recent_decisions.append(d)
                except (ValueError, TypeError, KeyError):
                    continue
            if recent_decisions:
                avg_sat = round(sum(d.get("satisfaction", 5) for d in recent_decisions) / len(recent_decisions), 1)
                insights.append(Insight(
                    priority="medium",
                    ability="shadow_council",
                    message=f"Recent decision satisfaction: {avg_sat:.1f}/10",
                    action="Learn from past choices",
                    confidence=0.7
                ))
        
        return insights
    
    # ═════════════════════════════════════════════════════════════════
    # CORE: ASK
    # ═════════════════════════════════════════════════════════════════
    
    async def ask(self, question: str, context: Dict = None) -> Dict:
        """
        Unified query interface.
        
        Instead of choosing which Shikigami, ask ANY question
        and let Shadow AI synthesize the best response.
        """
        if context is None:
            context = {"observations": self.observations[-20:]}
        
        # Auto-detect which abilities are relevant
        abilities_needed = self._detect_relevant_abilities(question)
        
        # Gather insights from multiple abilities
        responses = {}
        for ability in abilities_needed:
            try:
                result = await self.registry.execute(ability, question, context)
                responses[ability.value] = {
                    "output": result.output,
                    "data": result.data
                }
            except (ValueError, RuntimeError, AttributeError, TypeError, OSError) as e:
                logger.error("Ability %s failed: %s", ability, type(e).__name__)
        
        # Synthesize unified response
        unified = await self._synthesize_response(question, responses)
        
        return {
            "question": question,
            "abilities_used": [a.value for a in abilities_needed],
            "unified_response": unified,
            "individual_responses": responses,
            "insights": copy.deepcopy(self.insights[-5:]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _detect_relevant_abilities(self, question: str) -> List:
        """Detect which abilities are relevant to the question using evolved v10 algorithm."""
        return detect_relevant_abilities(question)
    
    async def _synthesize_response(self, question: str, responses: Dict) -> str:
        """Combine multiple ability outputs into unified response using evolved v10 algorithm."""
        if not responses:
            return "No relevant abilities found for this query."
        if len(responses) == 1:
            output = list(responses.values())[0].get("output", "")
            return output if output else "No output available."
        combined = "**Shadow AI Analysis**\n\n"
        for ability, response in responses.items():
            output = response.get("output", "")
            if not output:
                continue
            summary = output.split('\n')[:3]
            combined += f"**{ability.title()}**:\n"
            combined += '\n'.join(summary) + '\n\n'
        return combined
    
    # ═════════════════════════════════════════════════════════════════
    # CORE: ACT
    # ═════════════════════════════════════════════════════════════════
    
    async def act(self, action: str) -> Dict:
        """
        Take ACTION using Reality Forge + Intent Seer.
        
        Shadow AI doesn't just suggest - it executes.
        """
        # Predict outcome first (Intent Seer)
        prediction = await self.registry.execute(
            ShikigamiType.INTENT_SEER,
            f"What happens if I {action}",
            {"observations": self.observations[-10:]}
        )
        
        # Execute (Reality Forge)
        result = await self.registry.execute(
            ShikigamiType.REALITY_FORGE,
            action,
            {"observations": self.observations[-10:]}
        )
        
        # Learn from outcome (Shadow Council)
        prediction_output = getattr(prediction, 'output', str(prediction)) or ""
        _SANITIZE_PRED_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        safe_prediction = _SANITIZE_PRED_RE.sub('', prediction_output[:100])
        self.decisions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "situation": f"Action: {action}",
            "choice": f"Execute: {action}",
            "outcome": "executed",
            "prediction": safe_prediction
        })
        self._save_data()
        
        return {
            "action": action,
            "prediction": prediction_output,
            "result": getattr(result, 'output', str(result)) or "",
            "files_created": getattr(result, 'files_created', []),
            "commands_run": getattr(result, 'commands_run', [])
        }
    
    # ═════════════════════════════════════════════════════════════════
    # CORE: LEARN
    # ═════════════════════════════════════════════════════════════════
    
    _MAX_PATTERNS = 500
    _MAX_PEOPLE = 100
    _MAX_DEVICES_PER_PERSON = 50

    _SANITIZE_INPUT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]')
    _SANITIZE_CONTENT_INPUT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def track_person(self, name: str, devices: str, relation: str = ""):
        """Learn to identify a person."""
        name = self._SANITIZE_INPUT_RE.sub('', name.strip())[:200]
        device_list = [self._SANITIZE_INPUT_RE.sub('', d.strip()) for d in devices.replace(",", " ").split() if ":" in d or len(d) > 8]
        relation = self._SANITIZE_CONTENT_INPUT_RE.sub('', relation)[:500]
        with self._lock:
            if name not in self.people:
                if len(self.people) >= self._MAX_PEOPLE:
                    oldest = next(iter(self.people))
                    del self.people[oldest]
                self.people[name] = {"devices": [], "patterns": {}, "relation": relation}
            self.people[name]["devices"].extend(device_list[:self._MAX_DEVICES_PER_PERSON])
            self.people[name]["devices"] = list(set(self.people[name]["devices"]))[:self._MAX_DEVICES_PER_PERSON]
        self._save_data()

        # Bridge to shikigami registry so Life Pulse / Social Radar see the new
        # person without requiring a daemon restart.
        try:
            registry_people = getattr(self.registry, "people", None)
            if isinstance(registry_people, dict):
                registry_people.setdefault(name, {"devices": [], "patterns": {}, "relation": relation})
                registry_people[name]["devices"] = list({*registry_people[name].get("devices", []),
                                                         *self.people[name]["devices"]})
            fp = getattr(self.registry, "device_fingerprinter", None)
            if fp is not None and hasattr(fp, "register_person"):
                fp.register_person(name, self.people[name]["devices"])
        except Exception as e:
            logger.debug("person bridge failed: %s", type(e).__name__)

        return f"✓ Tracking {name} with {len(device_list)} devices"
    
    def track_decision(self, situation: str, choice: str, outcome: str, satisfaction: int):
        """Learn from a decision."""
        situation = self._SANITIZE_CONTENT_INPUT_RE.sub('', situation)[:1000]
        choice = self._SANITIZE_CONTENT_INPUT_RE.sub('', choice)[:1000]
        outcome = self._SANITIZE_CONTENT_INPUT_RE.sub('', outcome)[:1000]
        satisfaction = max(1, min(10, satisfaction))
        with self._lock:
            self.decisions.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "situation": situation,
                "choice": choice,
                "outcome": outcome,
                "satisfaction": satisfaction
            })
            if len(self.decisions) > 500:
                self.decisions = self.decisions[-500:]
        self._save_data()

        # Propagate to the shikigami's DecisionTracker so Shadow Council sees
        # the new decision without requiring a daemon restart.
        try:
            tracker = getattr(self.registry, "decision_tracker", None)
            if tracker is not None and hasattr(tracker, "record_decision"):
                tracker.record_decision(situation, choice, outcome, satisfaction)
        except Exception as e:
            logger.debug("decision bridge failed: %s", type(e).__name__)

        return f"✓ Decision tracked: {choice}"
    
    def learn_pattern(self, cause: str, effect: str):
        """Learn a causation pattern."""
        cause = self._SANITIZE_CONTENT_INPUT_RE.sub('', cause)[:1000]
        effect = self._SANITIZE_CONTENT_INPUT_RE.sub('', effect)[:1000]
        with self._lock:
            if len(self.patterns) >= self._MAX_PATTERNS:
                oldest = next(iter(self.patterns))
                del self.patterns[oldest]
            self.patterns[cause] = effect
        self._save_data()
        return f"✓ Pattern learned: {cause} → {effect}"
    
    # ═════════════════════════════════════════════════════════════════
    # STATUS
    # ═════════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict:
        """Get unified status."""
        with self._lock:
            obs_count = len(self.observations)
            people_count = len(self.people)
            decisions_count = len(self.decisions)
            patterns_count = len(self.patterns)
            pending_insights = len([i for i in self.insights if i.priority in ["critical", "high"]])
        return {
            "observations": obs_count,
            "tracked_people": people_count,
            "decisions_learned": decisions_count,
            "patterns_known": patterns_count,
            "pending_insights": pending_insights,
            "abilities": "10 unified (Oracle, X-Ray, Life Pulse, Perfect Memory, Pattern Sight, Social Radar, Intent Seer, Omni Sense, Shadow Council, Reality Forge)",
            "mode": "proactive optimization"
        }
    
    async def get_unified_snapshot(self) -> Dict:
        """
        Get a unified snapshot using ALL 10 abilities.
        
        This is the "Omni Sense" view - everything at once.
        """
        try:
            devices = await self.registry._scan_bt()
        except (OSError, subprocess.SubprocessError, asyncio.TimeoutError, RuntimeError):
            devices = []
        try:
            networks = await self.registry._scan_wifi()
        except (OSError, subprocess.SubprocessError, asyncio.TimeoutError, RuntimeError):
            networks = []
        
        present = [name for name, data in self.people.items()
                   for d in devices if d["mac"] in data.get("devices", [])]
        
        location = "unknown"
        for n in networks[:3]:
            if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["home", "my"]):
                location = "home"
            elif any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["office", "corp"]):
                location = "office"
        
        with self._lock:
            observations_copy = list(self.observations[-5:]) if self.observations else []
            commitments_count = len([o for o in self.observations if self._is_commitment(o)])
            insights_pending = len([i for i in self.insights if i.priority in ["critical", "high"]])
            people_count = len(self.people)
            patterns_count = len(self.patterns)
        
        return {
            "time": datetime.now(timezone.utc).strftime("%A %I:%M %p"),
            "location": location,
            "network": networks[0]["ssid"] if networks else "none",
            "devices_nearby": len(devices),
            "people_present": list(present),
            "activity": [o.get("type", "?") for o in observations_copy],
            "commitments": commitments_count,
            "patterns": patterns_count,
            "learned_people": people_count,
            "insights_pending": insights_pending
        }