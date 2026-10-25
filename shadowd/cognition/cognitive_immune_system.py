#!/usr/bin/env python3
"""
CognitiveImmuneSystem -- Belief Protection and Cognitive Pathogen Defense.

Protects the cognitive system from cognitive pathogens -- harmful beliefs,
logical fallacies, biases, misinformation. Features detection, antibody
(counter-argument) generation, inoculation, and learned immunity.

Architecture:
  - CognitivePathogen: a detected harmful cognitive pattern
  - CognitiveAntibody: a counter-argument defense mechanism
  - ImmuneMemoryCell: learned immunity from past exposure
  - DefenseResponse: record of a defensive action taken
  - CognitiveImmuneSystem: main engine managing pathogen defense

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import re
import uuid
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.cognitive_immune_system")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_PATHOGENS = 1000
_MAX_ANTIBODIES = 2000
_MAX_MEMORY = 500
_MAX_RESPONSES = 5000
_MAX_EVIDENCE = 50
_MAX_TAGS = 20


class PathogenType(str, Enum):
    MISINFORMATION = "misinformation"
    FALLACY = "fallacy"
    BIAS = "bias"
    DOGMA = "dogma"
    DELUSION = "delusion"
    RUMOR = "rumor"
    PROPAGANDA = "propaganda"
    CONSPIRACY = "conspiracy"


class DefenseType(str, Enum):
    ANTIBODY = "antibody"
    INOCULATION = "inoculation"
    MEMORY_CELL = "memory_cell"
    MACROPHAGE = "macrophage"
    NK_CELL = "nk_cell"


class ThreatLevel(int, Enum):
    BENIGN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResponseAction(str, Enum):
    IGNORE = "ignore"
    MONITOR = "monitor"
    NEUTRALIZE = "neutralize"
    QUARANTINE = "quarantine"
    ERADICATE = "eradicate"


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class CognitivePathogen:
    pathogen_id: str = ""
    name: str = ""
    pathogen_type: PathogenType = PathogenType.BIAS
    content: str = ""
    threat_level: int = 1
    spread_rate: float = 0.5
    evidence_count: int = 0
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pathogen_id:
            self.pathogen_id = "path_%s" % uuid.uuid4().hex[:12]
        self.pathogen_id = _sanitize(self.pathogen_id, 64)
        self.name = _sanitize(self.name, 200)
        if isinstance(self.pathogen_type, str):
            try:
                self.pathogen_type = PathogenType(self.pathogen_type)
            except ValueError:
                self.pathogen_type = PathogenType.BIAS
        self.content = _sanitize_content(self.content, 50000)
        self.threat_level = int(_clamp(self.threat_level, 0, 4))
        self.spread_rate = _clamp(self.spread_rate)
        self.evidence_count = max(0, int(self.evidence_count))

    def to_dict(self) -> Dict:
        return {
            "pathogen_id": self.pathogen_id,
            "name": self.name,
            "pathogen_type": self.pathogen_type.value,
            "content": self.content,
            "threat_level": int(self.threat_level),
            "spread_rate": round(self.spread_rate, 4),
            "evidence_count": self.evidence_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class CognitiveAntibody:
    antibody_id: str = ""
    target_pathogen_id: str = ""
    counter_argument: str = ""
    confidence: float = 0.5
    effectiveness: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.antibody_id:
            self.antibody_id = "ab_%s" % uuid.uuid4().hex[:12]
        self.antibody_id = _sanitize(self.antibody_id, 64)
        self.target_pathogen_id = _sanitize(self.target_pathogen_id, 64)
        self.counter_argument = _sanitize_content(self.counter_argument, 10000)
        self.confidence = _clamp(self.confidence)
        self.effectiveness = _clamp(self.effectiveness)

    def to_dict(self) -> Dict:
        return {
            "antibody_id": self.antibody_id,
            "target_pathogen_id": self.target_pathogen_id,
            "counter_argument": self.counter_argument,
            "confidence": round(self.confidence, 4),
            "effectiveness": round(self.effectiveness, 4),
            "created_at": self.created_at,
        }


@dataclass
class ImmuneMemoryCell:
    memory_id: str = ""
    pathogen_type: PathogenType = PathogenType.BIAS
    pattern: str = ""
    learned_defense: str = ""
    exposure_count: int = 0
    immunity_strength: float = 0.5

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = "mem_%s" % uuid.uuid4().hex[:12]
        self.memory_id = _sanitize(self.memory_id, 64)
        if isinstance(self.pathogen_type, str):
            try:
                self.pathogen_type = PathogenType(self.pathogen_type)
            except ValueError:
                self.pathogen_type = PathogenType.BIAS
        self.pattern = _sanitize_content(self.pattern, 2000)
        self.learned_defense = _sanitize_content(self.learned_defense, 5000)
        self.exposure_count = max(0, int(self.exposure_count))
        self.immunity_strength = _clamp(self.immunity_strength)

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "pathogen_type": self.pathogen_type.value,
            "pattern": self.pattern,
            "learned_defense": self.learned_defense,
            "exposure_count": self.exposure_count,
            "immunity_strength": round(self.immunity_strength, 4),
        }


@dataclass
class DefenseResponse:
    response_id: str = ""
    pathogen_id: str = ""
    defense_type: DefenseType = DefenseType.ANTIBODY
    action: ResponseAction = ResponseAction.MONITOR
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.response_id:
            self.response_id = "dr_%s" % uuid.uuid4().hex[:12]
        self.response_id = _sanitize(self.response_id, 64)
        self.pathogen_id = _sanitize(self.pathogen_id, 64)
        if isinstance(self.defense_type, str):
            try:
                self.defense_type = DefenseType(self.defense_type)
            except ValueError:
                self.defense_type = DefenseType.ANTIBODY
        if isinstance(self.action, str):
            try:
                self.action = ResponseAction(self.action)
            except ValueError:
                self.action = ResponseAction.MONITOR
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "response_id": self.response_id,
            "pathogen_id": self.pathogen_id,
            "defense_type": self.defense_type.value,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
        }


_PATHOGEN_KEYWORDS: Dict[PathogenType, List[str]] = {
    PathogenType.MISINFORMATION: [
        "fake news", "disinformation", "hoax", "unverified", "debunked",
        "fabricated", "false claim", "baseless", "unsubstantiated",
    ],
    PathogenType.FALLACY: [
        "therefore", "obviously", "everyone knows", "slippery slope",
        "straw man", "ad hominem", "circular reasoning", "false dichotomy",
        "appeal to authority", "post hoc", "correlation causation",
        "begging the question", "red herring", "non sequitur",
    ],
    PathogenType.BIAS: [
        "confirmation bias", "anchoring", "availability heuristic",
        "sunk cost", "dunning kruger", "framing effect", "bandwagon",
        "halo effect", "hindsight bias", "survivorship", "recency bias",
        "selection bias", "attribution error",
    ],
    PathogenType.DOGMA: [
        "always been", "never questioned", "immutable truth", "sacred",
        "infallible", "beyond doubt", "unquestionable", "absolute certainty",
        "forbidden to challenge", "eternal truth",
    ],
    PathogenType.DELUSION: [
        "grandiose", "persecution", "paranoid", "omniscient",
        "invulnerable", "chosen one", "reality distortion", "magical thinking",
        "impossible coincidence", "destiny",
    ],
    PathogenType.RUMOR: [
        "heard that", "supposedly", "rumor has it", "word on the street",
        "i heard", "someone said", "whispered", "unconfirmed report",
        "allegedly", "supposedly",
    ],
    PathogenType.PROPAGANDA: [
        "must believe", "enemy of the people", "us versus them",
        "inevitable", "only option", "those people", "true believers",
        "thoughtcrime", "loyalty oath", "with us or against us",
    ],
    PathogenType.CONSPIRACY: [
        "they don't want you to know", "cover up", "secret society",
        "shadow government", "hidden agenda", "controlled by",
        "deep state", "master plan", "orchestrated", "suppressed evidence",
    ],
}

_SPREAD_RATE_BY_TYPE: Dict[PathogenType, float] = {
    PathogenType.MISINFORMATION: 0.7,
    PathogenType.FALLACY: 0.3,
    PathogenType.BIAS: 0.4,
    PathogenType.DOGMA: 0.2,
    PathogenType.DELUSION: 0.1,
    PathogenType.RUMOR: 0.8,
    PathogenType.PROPAGANDA: 0.6,
    PathogenType.CONSPIRACY: 0.5,
}

_THREAT_KEYWORD_WEIGHTS: Dict[str, int] = {
    "dangerous": 2, "harmful": 2, "toxic": 3, "extreme": 2,
    "absolute": 1, "never": 1, "always": 1, "must": 1,
    "proven": 1, "undeniable": 2, "impossible": 1,
}


class CognitiveImmuneSystem:
    """Belief Protection and Cognitive Pathogen Defense.

    Detects cognitive pathogens, generates counter-argument antibodies,
    inoculates against known patterns, and maintains learned immunity.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._pathogens: Dict[str, CognitivePathogen] = {}
        self._pathogen_order: deque = deque(maxlen=_MAX_PATHOGENS)
        self._antibodies: Dict[str, CognitiveAntibody] = {}
        self._antibody_order: deque = deque(maxlen=_MAX_ANTIBODIES)
        self._memory_cells: Dict[str, ImmuneMemoryCell] = {}
        self._memory_order: deque = deque(maxlen=_MAX_MEMORY)
        self._responses: Dict[str, DefenseResponse] = {}
        self._response_order: deque = deque(maxlen=_MAX_RESPONSES)
        self._stats = {
            "pathogens_detected": 0,
            "antibodies_generated": 0,
            "pathogens_neutralized": 0,
            "pathogens_quarantined": 0,
            "pathogens_eradicated": 0,
            "inoculations_performed": 0,
            "immunity_checks": 0,
            "threats_evaluated": 0,
        }

    def detect_pathogen(
        self,
        content: str,
        source: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CognitivePathogen]:
        """Scan content for cognitive pathogens."""
        raw_content = content if isinstance(content, str) else str(content)
        content = _sanitize_content(content, 50000)
        source = _sanitize(source, 200)
        if context is None:
            context = {}
        context = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list(context.items())[:20]
        }

        content_lower = raw_content.lower()
        best_type: Optional[PathogenType] = None
        best_score = 0.0

        for ptype, keywords in _PATHOGEN_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if len(kw.split()) > 1:
                    if kw in content_lower:
                        score += 0.3
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', content_lower):
                        score += 0.2
            if score > best_score:
                best_score = score
                best_type = ptype

        threat_bonus = 0.0
        for kw, weight in _THREAT_KEYWORD_WEIGHTS.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', content_lower):
                threat_bonus += weight * 0.05

        best_score += threat_bonus

        if best_type is None or best_score < 0.1:
            return None

        threat_level = ThreatLevel.LOW
        if best_score >= 1.0:
            threat_level = ThreatLevel.CRITICAL
        elif best_score >= 0.7:
            threat_level = ThreatLevel.HIGH
        elif best_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif best_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.BENIGN

        spread_rate = _SPREAD_RATE_BY_TYPE.get(best_type, 0.3)
        pathogen = CognitivePathogen(
            name=_sanitize(best_type.value, 200),
            pathogen_type=best_type,
            content=content,
            threat_level=int(threat_level),
            spread_rate=spread_rate,
        )

        with self._lock:
            if len(self._pathogens) >= _MAX_PATHOGENS:
                evict_id = self._pathogen_order.popleft()
                self._pathogens.pop(evict_id, None)
                logger.warning("Evicting pathogen %s (cap=%d)", evict_id, _MAX_PATHOGENS)
            self._pathogens[pathogen.pathogen_id] = pathogen
            self._pathogen_order.append(pathogen.pathogen_id)
            with self._stats_lock:
                self._stats["pathogens_detected"] += 1

        return copy.deepcopy(pathogen)

    def generate_antibody(
        self,
        pathogen_id: str,
        counter_argument: str,
        confidence: float = 0.5,
    ) -> Optional[CognitiveAntibody]:
        """Create a counter-argument antibody against a pathogen."""
        pathogen_id = _sanitize(pathogen_id, 64)
        counter_argument = _sanitize_content(counter_argument, 10000)
        confidence = _clamp(confidence)

        with self._lock:
            pathogen = self._pathogens.get(pathogen_id)
            if pathogen is None:
                return None
            effectiveness = _clamp(0.3 + confidence * 0.5)
            if pathogen.threat_level >= ThreatLevel.HIGH:
                effectiveness = _clamp(effectiveness * 0.85)

            antibody = CognitiveAntibody(
                target_pathogen_id=pathogen_id,
                counter_argument=counter_argument,
                confidence=confidence,
                effectiveness=effectiveness,
            )

            if len(self._antibodies) >= _MAX_ANTIBODIES:
                evict_id = self._antibody_order.popleft()
                self._antibodies.pop(evict_id, None)
                logger.warning("Evicting antibody %s (cap=%d)", evict_id, _MAX_ANTIBODIES)
            self._antibodies[antibody.antibody_id] = antibody
            self._antibody_order.append(antibody.antibody_id)
            with self._stats_lock:
                self._stats["antibodies_generated"] += 1

            response = DefenseResponse(
                pathogen_id=pathogen_id,
                defense_type=DefenseType.ANTIBODY,
                action=ResponseAction.NEUTRALIZE,
                confidence=confidence,
            )
            if len(self._responses) >= _MAX_RESPONSES:
                evict_id = self._response_order.popleft()
                self._responses.pop(evict_id, None)
                logger.warning("Evicting response %s (cap=%d)", evict_id, _MAX_RESPONSES)
            self._responses[response.response_id] = response
            self._response_order.append(response.response_id)

        return copy.deepcopy(antibody)

    def neutralize(self, pathogen_id: str) -> Optional[DefenseResponse]:
        """Apply antibodies to neutralize a pathogen."""
        pathogen_id = _sanitize(pathogen_id, 64)

        with self._lock:
            pathogen = self._pathogens.get(pathogen_id)
            if pathogen is None:
                return None

            antibodies = [
                ab for ab in self._antibodies.values()
                if ab.target_pathogen_id == pathogen_id
            ]
            best_ab = max(antibodies, key=lambda a: a.effectiveness) if antibodies else None

            confidence = best_ab.effectiveness if best_ab else 0.1
            action = ResponseAction.NEUTRALIZE
            if pathogen.threat_level >= ThreatLevel.CRITICAL:
                action = ResponseAction.ERADICATE

            response = DefenseResponse(
                pathogen_id=pathogen_id,
                defense_type=DefenseType.ANTIBODY if best_ab else DefenseType.NK_CELL,
                action=action,
                confidence=confidence,
            )

            if len(self._responses) >= _MAX_RESPONSES:
                evict_id = self._response_order.popleft()
                self._responses.pop(evict_id, None)
                logger.warning("Evicting response %s (cap=%d)", evict_id, _MAX_RESPONSES)
            self._responses[response.response_id] = response
            self._response_order.append(response.response_id)
            with self._stats_lock:
                self._stats["pathogens_neutralized"] += 1

        return copy.deepcopy(response)

    def quarantine(self, pathogen_id: str) -> Optional[DefenseResponse]:
        """Isolate a harmful belief by quarantining it."""
        pathogen_id = _sanitize(pathogen_id, 64)

        with self._lock:
            pathogen = self._pathogens.get(pathogen_id)
            if pathogen is None:
                return None

            response = DefenseResponse(
                pathogen_id=pathogen_id,
                defense_type=DefenseType.MACROPHAGE,
                action=ResponseAction.QUARANTINE,
                confidence=_clamp(pathogen.spread_rate * 0.8),
            )

            if len(self._responses) >= _MAX_RESPONSES:
                evict_id = self._response_order.popleft()
                self._responses.pop(evict_id, None)
                logger.warning("Evicting response %s (cap=%d)", evict_id, _MAX_RESPONSES)
            self._responses[response.response_id] = response
            self._response_order.append(response.response_id)

            pathogen.threat_level = max(pathogen.threat_level, int(ThreatLevel.MEDIUM))
            with self._stats_lock:
                self._stats["pathogens_quarantined"] += 1

        return copy.deepcopy(response)

    def inoculate(
        self,
        pathogen_type: PathogenType,
        weak_content: str,
    ) -> ImmuneMemoryCell:
        """Preemptive defense -- vaccination against a pathogen type."""
        if isinstance(pathogen_type, str):
            try:
                pathogen_type = PathogenType(pathogen_type)
            except ValueError:
                pathogen_type = PathogenType.BIAS
        weak_content = _sanitize_content(weak_content, 5000)

        content_lower = weak_content.lower()
        keywords = _PATHOGEN_KEYWORDS.get(pathogen_type, [])
        match_score = 0.0
        for kw in keywords:
            if len(kw.split()) > 1:
                if kw in content_lower:
                    match_score += 0.15
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', content_lower):
                    match_score += 0.1
        immunity_strength = _clamp(0.3 + match_score)

        defense_parts = ["Recognize %s patterns" % pathogen_type.value]
        if keywords:
            top_keywords = sorted(keywords, key=len, reverse=True)[:3]
            defense_parts.append("Watch for: %s" % ", ".join(top_keywords))
        defense_parts.append("Apply critical thinking and verify claims independently")
        learned_defense = "; ".join(defense_parts)

        memory = ImmuneMemoryCell(
            pathogen_type=pathogen_type,
            pattern=_sanitize_content(weak_content, 2000),
            learned_defense=learned_defense,
            exposure_count=1,
            immunity_strength=immunity_strength,
        )

        with self._lock:
            existing = None
            for cell in self._memory_cells.values():
                if cell.pathogen_type == pathogen_type and cell.pattern == memory.pattern:
                    existing = cell
                    break

            if existing is not None:
                existing.exposure_count += 1
                existing.immunity_strength = _clamp(
                    existing.immunity_strength + 0.05
                )
                existing.learned_defense = learned_defense
                return copy.deepcopy(existing)

            if len(self._memory_cells) >= _MAX_MEMORY:
                evict_id = self._memory_order.popleft()
                self._memory_cells.pop(evict_id, None)
                logger.warning("Evicting memory cell %s (cap=%d)", evict_id, _MAX_MEMORY)
            self._memory_cells[memory.memory_id] = memory
            self._memory_order.append(memory.memory_id)
            with self._stats_lock:
                self._stats["inoculations_performed"] += 1

        return copy.deepcopy(memory)

    def check_immunity(
        self,
        content: str,
        pathogen_type: PathogenType,
    ) -> float:
        """Check if system has immunity to a specific pathogen type."""
        content = _sanitize_content(content, 5000)
        if isinstance(pathogen_type, str):
            try:
                pathogen_type = PathogenType(pathogen_type)
            except ValueError:
                return 0.0

        with self._lock:
            best_immunity = 0.0
            for cell in self._memory_cells.values():
                if cell.pathogen_type != pathogen_type:
                    continue
                content_words = set(re.findall(r'[a-zA-Z0-9]+', content.lower()))
                pattern_words = set(re.findall(r'[a-zA-Z0-9]+', cell.pattern.lower()))
                if not content_words or not pattern_words:
                    continue
                overlap = len(content_words & pattern_words) / len(content_words | pattern_words)
                immunity = cell.immunity_strength * (0.5 + overlap * 0.5)
                if immunity > best_immunity:
                    best_immunity = immunity

            with self._stats_lock:
                self._stats["immunity_checks"] += 1

        return round(best_immunity, 4)

    def get_pathogens(
        self,
        threat_level: Optional[int] = None,
    ) -> List[CognitivePathogen]:
        """List detected pathogens, optionally filtered by threat level."""
        with self._lock:
            pathogens = list(self._pathogens.values())
            if threat_level is not None:
                pathogens = [
                    p for p in pathogens
                    if p.threat_level >= threat_level
                ]
        return [copy.deepcopy(p) for p in pathogens]

    def get_antibodies(
        self,
        pathogen_id: Optional[str] = None,
    ) -> List[CognitiveAntibody]:
        """List antibodies, optionally filtered by target pathogen."""
        pathogen_id = _sanitize(pathogen_id, 64) if pathogen_id else None
        with self._lock:
            antibodies = list(self._antibodies.values())
            if pathogen_id:
                antibodies = [
                    a for a in antibodies
                    if a.target_pathogen_id == pathogen_id
                ]
        return [copy.deepcopy(a) for a in antibodies]

    def get_immune_memory(self) -> List[ImmuneMemoryCell]:
        """List learned immunity cells."""
        with self._lock:
            cells = list(self._memory_cells.values())
        return [copy.deepcopy(c) for c in cells]

    def evaluate_threat(self, content: str) -> Dict[str, Any]:
        """Assess threat level of content."""
        content = _sanitize_content(content, 50000)

        content_lower = content.lower()
        type_scores: Dict[str, float] = {}
        for ptype, keywords in _PATHOGEN_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if len(kw.split()) > 1:
                    if kw in content_lower:
                        score += 0.3
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', content_lower):
                        score += 0.2
            type_scores[ptype.value] = round(score, 4)

        threat_bonus = 0.0
        for kw, weight in _THREAT_KEYWORD_WEIGHTS.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', content_lower):
                threat_bonus += weight * 0.05

        total_score = sum(type_scores.values()) + threat_bonus
        if total_score >= 2.0:
            overall_threat = ThreatLevel.CRITICAL
        elif total_score >= 1.0:
            overall_threat = ThreatLevel.HIGH
        elif total_score >= 0.5:
            overall_threat = ThreatLevel.MEDIUM
        elif total_score >= 0.2:
            overall_threat = ThreatLevel.LOW
        else:
            overall_threat = ThreatLevel.BENIGN

        recommended_action = ResponseAction.MONITOR
        if overall_threat >= ThreatLevel.CRITICAL:
            recommended_action = ResponseAction.ERADICATE
        elif overall_threat >= ThreatLevel.HIGH:
            recommended_action = ResponseAction.QUARANTINE
        elif overall_threat >= ThreatLevel.MEDIUM:
            recommended_action = ResponseAction.NEUTRALIZE
        elif overall_threat >= ThreatLevel.LOW:
            recommended_action = ResponseAction.MONITOR

        with self._stats_lock:
            self._stats["threats_evaluated"] += 1

        return {
            "overall_threat": int(overall_threat),
            "overall_threat_name": overall_threat.name.lower(),
            "recommended_action": recommended_action.value,
            "total_score": round(total_score, 4),
            "type_scores": type_scores,
            "threat_keywords_found": round(threat_bonus, 4),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics."""
        with self._lock:
            pathogen_count = len(self._pathogens)
            antibody_count = len(self._antibodies)
            memory_count = len(self._memory_cells)
            response_count = len(self._responses)
            threat_dist = {t.name.lower(): 0 for t in ThreatLevel}
            for p in self._pathogens.values():
                level = ThreatLevel(p.threat_level)
                threat_dist[level.name.lower()] = threat_dist.get(level.name.lower(), 0) + 1
            type_dist = {t.value: 0 for t in PathogenType}
            for p in self._pathogens.values():
                type_dist[p.pathogen_type.value] = type_dist.get(p.pathogen_type.value, 0) + 1
        with self._stats_lock:
            stats = dict(self._stats)
        stats["pathogen_count"] = pathogen_count
        stats["antibody_count"] = antibody_count
        stats["memory_count"] = memory_count
        stats["response_count"] = response_count
        stats["threat_distribution"] = threat_dist
        stats["type_distribution"] = type_dist
        return stats