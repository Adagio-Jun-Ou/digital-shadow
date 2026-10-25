"""
Cross-Module Synthesis Engine — Knowledge Fusion Across 40+ Cognitive Modules.

When PatternSight detects a causation chain, FeedLoop routes signals, and
CognitiveResonance finds hidden connections — SynthesisEngine combines these
disparate insights into unified knowledge. It does not just aggregate; it fuses
concepts across domains, discovers emergent properties, and generates synthetic
insights that no single module could produce alone.

Design principles:
- Synthesis over aggregation: combine, don't just collect
- Temporal awareness: synthesis decays, strengthens, and evolves over time
- Confidence-weighted fusion: strong signals dominate, weak signals fade
- Self-validating: synthesized knowledge is tested against module ground truths
- Actionable output: every synthesis produces concrete recommendations

Inspired by: Self-Rewarding LMs (Yuan et al. 2024), AutoGen multi-agent patterns (Wu et al. 2023),
and Constitutional AI self-critique (Bai et al. 2022).
"""

import copy
import logging
import re
import threading
import uuid
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 1000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]


_MAX_SYNTHESES = 5000
_MAX_INSIGHTS = 2000
_MAX_PATTERNS = 1000
_MAX_RECOMMENDATION_HISTORY = 500
_MAX_CONCEPT_GRAPH_NODES = 5000
_MAX_CONCEPT_GRAPH_EDGES = 10000
_MAX_TOPIC_CLUSTERS = 200
_MAX_VALIDATION_HISTORY = 1000
_MAX_TREND_WINDOW = 100
_MAX_MODULE_SOURCES_PER_SYNTHESIS = 10
_MAX_CONFIDENCE_HISTORY = 50


class SynthesisDomain(Enum):
    CROSS_MODULE = "cross_module"
    TEMPORAL_FUSION = "temporal_fusion"
    CAUSAL_CHAIN = "causal_chain"
    EMERGENT_PROPERTY = "emergent_property"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    PREDICTIVE_FUSION = "predictive_fusion"
    CONCEPTUAL_SYNTHESIS = "conceptual_synthesis"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class ConfidenceTier(Enum):
    SPECULATIVE = 0.3
    PLAUSIBLE = 0.5
    PROBABLE = 0.7
    CONFIRMED = 0.85
    CERTAIN = 0.95


class ModuleCategory(Enum):
    COGNITIVE = "cognitive"
    MEMORY = "memory"
    PERSONALITY = "personality"
    SOCIAL = "social"
    PREDICTIVE = "predictive"
    META = "meta"
    CREATIVE = "creative"
    SECURITY = "security"
    FUSION = "fusion"


MODULE_CATEGORY_MAP: Dict[str, ModuleCategory] = {
    "adaptive_skill_tree": ModuleCategory.COGNITIVE,
    "attention_engine": ModuleCategory.COGNITIVE,
    "cognitive_pipeline": ModuleCategory.COGNITIVE,
    "cognitive_reflection": ModuleCategory.META,
    "cognitive_resonance": ModuleCategory.FUSION,
    "cognitive_trinity": ModuleCategory.COGNITIVE,
    "conflict_resolution": ModuleCategory.SOCIAL,
    "creative_synthesis": ModuleCategory.CREATIVE,
    "creative_dream": ModuleCategory.CREATIVE,
    "daily_briefing": ModuleCategory.COGNITIVE,
    "dark_memory": ModuleCategory.MEMORY,
    "decision_journal": ModuleCategory.META,
    "device_context": ModuleCategory.COGNITIVE,
    "dream_state": ModuleCategory.MEMORY,
    "emotional_intelligence": ModuleCategory.COGNITIVE,
    "empathy_mapping": ModuleCategory.SOCIAL,
    "escalation": ModuleCategory.SECURITY,
    "evolutionary_engine": ModuleCategory.META,
    "evolution_guide": ModuleCategory.META,
    "feed_loop": ModuleCategory.FUSION,
    "goal_planner": ModuleCategory.COGNITIVE,
    "goal_reasoning": ModuleCategory.COGNITIVE,
    "health_dashboard": ModuleCategory.META,
    "metacognition_engine": ModuleCategory.META,
    "moral_reasoning": ModuleCategory.META,
    "narrative_memory": ModuleCategory.MEMORY,
    "neural_mesh": ModuleCategory.SOCIAL,
    "oracle_bridge": ModuleCategory.PREDICTIVE,
    "personality_engine": ModuleCategory.PERSONALITY,
    "presence_detection": ModuleCategory.SOCIAL,
    "procedural_memory": ModuleCategory.MEMORY,
    "recall_synthesis": ModuleCategory.FUSION,
    "self_improvement": ModuleCategory.META,
    "shadow_ai": ModuleCategory.COGNITIVE,
    "shadow_arise": ModuleCategory.COGNITIVE,
    "shadow_crew": ModuleCategory.COGNITIVE,
    "shadow_graph": ModuleCategory.COGNITIVE,
    "shadow_powers": ModuleCategory.COGNITIVE,
    "shadow_summon": ModuleCategory.COGNITIVE,
    "shadow_voice": ModuleCategory.COGNITIVE,
    "social_intelligence": ModuleCategory.SOCIAL,
    "social_graph_propagation": ModuleCategory.SOCIAL,
    "stealth_layer": ModuleCategory.SECURITY,
    "temporal_reasoning": ModuleCategory.PREDICTIVE,
    "temporal_self": ModuleCategory.COGNITIVE,
    "thought_sense": ModuleCategory.COGNITIVE,
}

CATEGORY_CROSS_INFLUENCE: Dict[Tuple[str, str], float] = {
    ("cognitive", "memory"): 0.85,
    ("cognitive", "meta"): 0.80,
    ("cognitive", "predictive"): 0.75,
    ("cognitive", "creative"): 0.70,
    ("memory", "personality"): 0.80,
    ("memory", "social"): 0.70,
    ("personality", "social"): 0.90,
    ("personality", "predictive"): 0.75,
    ("social", "predictive"): 0.65,
    ("meta", "cognitive"): 0.85,
    ("meta", "predictive"): 0.70,
    ("creative", "cognitive"): 0.75,
    ("creative", "memory"): 0.65,
    ("fusion", "cognitive"): 0.80,
    ("fusion", "predictive"): 0.75,
    ("fusion", "memory"): 0.70,
    ("security", "meta"): 0.60,
    ("security", "cognitive"): 0.55,
}


@dataclass
class ModuleSignal:
    source_module: str
    category: str
    signal_type: str
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    domains: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.source_module = _sanitize(self.source_module, 50)
        self.category = _sanitize(self.category, 30)
        self.signal_type = _sanitize(self.signal_type, 50)
        self.content = _sanitize_content(self.content, 5000)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.domains = [_sanitize(d, 30) for d in (self.domains or [])[:20]]
        self.entities = [_sanitize(e, 50) for e in (self.entities or [])[:50]]
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }
        if not self.signal_id:
            self.signal_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "source_module": self.source_module,
            "category": self.category,
            "signal_type": self.signal_type,
            "content": self.content[:200],
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "domains": list(self.domains),
            "entities": list(self.entities),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class SynthesisResult:
    synthesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    domain: str = "cross_module"
    source_modules: List[str] = field(default_factory=list)
    source_signals: List[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    key_finding: str = ""
    confidence: float = 0.0
    action_recommendation: str = ""
    expected_impact: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    concepts_linked: List[str] = field(default_factory=list)
    category_crosses: List[Tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    validation_count: int = 0
    validation_score: float = 0.0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.synthesis_id = _sanitize(self.synthesis_id, 20)
        self.domain = _sanitize(self.domain, 30)
        self.source_modules = [_sanitize(m, 50) for m in (self.source_modules or [])[:20]]
        self.source_signals = [_sanitize(s, 20) for s in (self.source_signals or [])[:50]]
        self.title = _sanitize_content(self.title, 200)
        self.summary = _sanitize_content(self.summary, 5000)
        self.key_finding = _sanitize_content(self.key_finding, 1000)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.action_recommendation = _sanitize_content(self.action_recommendation, 1000)
        self.expected_impact = max(0.0, min(1.0, self.expected_impact))
        self.supporting_evidence = [_sanitize_content(e, 500) for e in (self.supporting_evidence or [])[:20]]
        self.conflicting_evidence = [_sanitize_content(e, 500) for e in (self.conflicting_evidence or [])[:20]]
        self.concepts_linked = [_sanitize(c, 100) for c in (self.concepts_linked or [])[:30]]
        self.category_crosses = [
            (_sanitize(a, 30), _sanitize(b, 30))
            for a, b in (self.category_crosses or [])[:20]
        ]
        self.validation_score = max(0.0, min(1.0, self.validation_score))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthesis_id": self.synthesis_id,
            "domain": self.domain,
            "source_modules": list(self.source_modules),
            "source_signals": list(self.source_signals),
            "title": self.title,
            "summary": self.summary[:500],
            "key_finding": self.key_finding,
            "confidence": round(self.confidence, 3),
            "action_recommendation": self.action_recommendation,
            "expected_impact": round(self.expected_impact, 3),
            "supporting_evidence": list(self.supporting_evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
            "concepts_linked": list(self.concepts_linked),
            "category_crosses": list(self.category_crosses),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "validation_count": self.validation_count,
            "validation_score": round(self.validation_score, 3),
            "is_active": self.is_active,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class TrendSnapshot:
    trend_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    topic: str = ""
    start_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    values: List[float] = field(default_factory=list)
    trend_direction: str = "stable"
    rate_of_change: float = 0.0
    acceleration: float = 0.0
    forecast_next: float = 0.0
    forecast_confidence: float = 0.0

    def __post_init__(self):
        self.topic = _sanitize(self.topic, 100)
        self.trend_direction = _sanitize(self.trend_direction, 20)
        self.values = [max(0.0, min(1.0, float(v))) for v in (self.values or [])[:_MAX_TREND_WINDOW]]
        self.rate_of_change = max(-1.0, min(1.0, self.rate_of_change))
        self.forecast_next = max(0.0, min(1.0, self.forecast_next))
        self.forecast_confidence = max(0.0, min(1.0, self.forecast_confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_id": self.trend_id,
            "topic": self.topic,
            "start_at": self.start_at.isoformat(),
            "values": list(self.values[-10:]),
            "trend_direction": self.trend_direction,
            "rate_of_change": round(self.rate_of_change, 4),
            "acceleration": round(self.acceleration, 4),
            "forecast_next": round(self.forecast_next, 3),
            "forecast_confidence": round(self.forecast_confidence, 3),
        }


class CrossModuleSynthesisEngine:
    """
    Knowledge fusion engine that synthesizes insights across all cognitive modules.

    Core capabilities:
    1. Signal ingestion from any module (observe pattern)
    2. Cross-category synthesis (fuse signals across memory/cognitive/social/etc.)
    3. Contradiction detection and resolution
    4. Emergent property discovery (insights no single module can see)
    5. Trend detection across signal types
    6. Confidence-weighted validation (self-validating synthesis)
    7. Action recommendation generation
    """

    _MAX_SYNTHESES = _MAX_SYNTHESES
    _MAX_INSIGHTS = _MAX_INSIGHTS
    _MAX_PATTERNS = _MAX_PATTERNS

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._signals: deque = deque(maxlen=_MAX_SYNTHESES)
        self._syntheses: Dict[str, SynthesisResult] = {}
        self._synthesis_order: deque = deque(maxlen=_MAX_SYNTHESES)
        self._module_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_PATTERNS))
        self._concept_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._trends: Dict[str, TrendSnapshot] = {}
        self._trend_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_TREND_WINDOW))
        self._cross_category_correlations: Dict[Tuple[str, str], float] = {}
        self._validation_history: deque = deque(maxlen=_MAX_VALIDATION_HISTORY)

        self._stats: Dict[str, int] = {
            "signals_ingested": 0,
            "syntheses_generated": 0,
            "cross_category_crosses": 0,
            "contradictions_detected": 0,
            "contradictions_resolved": 0,
            "emergent_properties": 0,
            "trends_active": 0,
            "validations_performed": 0,
            "recommendations_accepted": 0,
            "concepts_linked": 0,
            "active_syntheses": 0,
        }

    def ingest_signal(self, signal: ModuleSignal) -> str:
        """Ingest a signal from any cognitive module for synthesis."""
        with self._lock:
            self._signals.append(signal)
            self._module_activity[signal.source_module].append(signal)

        with self._stats_lock:
            self._stats["signals_ingested"] += 1

        self._update_trend(signal)
        self._update_concept_graph(signal)

        return signal.signal_id

    def synthesize(
        self,
        modules: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        max_syntheses: int = 5,
    ) -> List[SynthesisResult]:
        """Synthesize insights across modules, discovering cross-category patterns."""
        modules = [_sanitize(m, 50) for m in (modules or [])[:_MAX_MODULE_SOURCES_PER_SYNTHESIS]]
        categories = [_sanitize(c, 30) for c in (categories or [])[:10]]
        min_confidence = max(0.0, min(1.0, min_confidence))
        max_syntheses = max(1, min(20, max_syntheses))

        with self._lock:
            signals_snapshot = list(self._signals)
            syntheses_prev_count = len(self._syntheses)

        filtered = self._filter_signals(signals_snapshot, modules, categories, min_confidence)
        if not filtered:
            return []

        syntheses = []
        category_groups = self._group_by_category(filtered)

        for cat_pair, signals in self._find_category_pairs(category_groups):
            if len(syntheses) >= max_syntheses:
                break
            synthesis = self._generate_synthesis(signals, cat_pair)
            if synthesis:
                syntheses.append(synthesis)

        with self._lock:
            for syn in syntheses:
                if syn.synthesis_id not in self._syntheses:
                    self._syntheses[syn.synthesis_id] = syn
                    self._synthesis_order.append(syn.synthesis_id)
                if len(self._syntheses) > self._MAX_SYNTHESES:
                    old_id = self._synthesis_order.popleft()
                    self._syntheses.pop(old_id, None)

            synthesis_count = len(self._syntheses) - syntheses_prev_count
            if synthesis_count > 0:
                pass

        with self._stats_lock:
            self._stats["syntheses_generated"] += len(syntheses)
            self._stats["active_syntheses"] = len(self._syntheses)

        return syntheses

    def detect_contradictions(self, min_severity: float = 0.3) -> List[Dict[str, Any]]:
        """Detect contradictions between module outputs."""
        min_severity = max(0.0, min(1.0, min_severity))

        with self._lock:
            syntheses_snapshot = list(self._syntheses.values())

        contradictions = []
        syn_list = list(syntheses_snapshot)

        for i in range(len(syn_list)):
            for j in range(i + 1, len(syn_list)):
                sa = syn_list[i]
                sb = syn_list[j]
                severity = self._compute_contradiction_severity(sa, sb)

                if severity >= min_severity:
                    contradiction = {
                        "synthesis_a": sa.synthesis_id,
                        "synthesis_b": sb.synthesis_id,
                        "title_a": sa.title[:100],
                        "title_b": sb.title[:100],
                        "severity": round(severity, 3),
                        "resolution_suggestion": self._suggest_contradiction_resolution(sa, sb),
                    }
                    contradictions.append(contradiction)

        with self._stats_lock:
            self._stats["contradictions_detected"] += len(contradictions)

        return contradictions

    def resolve_contradiction(
        self, synthesis_id_a: str, synthesis_id_b: str, resolution: str
    ) -> Optional[SynthesisResult]:
        """Resolve a contradiction by creating a synthesis of the conflict."""
        synthesis_id_a = _sanitize(synthesis_id_a, 20)
        synthesis_id_b = _sanitize(synthesis_id_b, 20)
        resolution = _sanitize_content(resolution, 5000)

        with self._lock:
            sa = self._syntheses.get(synthesis_id_a)
            sb = self._syntheses.get(synthesis_id_b)

        if not sa or not sb:
            return None

        synthesis = SynthesisResult(
            domain="contradiction_resolution",
            title=f"Resolution: {sa.title[:50]} vs {sb.title[:50]}",
            summary=resolution,
            key_finding=resolution[:500],
            source_modules=list(set(sa.source_modules + sb.source_modules)),
            source_signals=list(set(sa.source_signals + sb.source_signals)),
            confidence=round((sa.confidence + sb.confidence) / 2, 2),
            category_crosses=list(set(sa.category_crosses + sb.category_crosses)),
            action_recommendation=f"Apply resolution: {resolution[:200]}",
            expected_impact=round((sa.expected_impact + sb.expected_impact) / 2, 2),
            metadata={"resolution_type": "synthesis", "contradiction_severity": round(
                self._compute_contradiction_severity(sa, sb), 3
            )},
        )

        with self._lock:
            self._syntheses[synthesis.synthesis_id] = synthesis
            self._synthesis_order.append(synthesis.synthesis_id)

        with self._stats_lock:
            self._stats["contradictions_resolved"] += 1

        return synthesis

    def detect_emergent_properties(self, min_confidence: float = 0.4) -> List[SynthesisResult]:
        """Discover emergent properties that transcend single-module insights."""
        min_confidence = max(0.0, min(1.0, min_confidence))

        with self._lock:
            syntheses_snapshot = list(self._syntheses.values())
            signal_modules = set()
            for s in self._signals:
                signal_modules.add(s.source_module)

        emergents = []
        if len(signal_modules) < 2:
            return emergents

        module_groups = self._cluster_related_modules(list(signal_modules))
        for group in module_groups:
            if len(group) < 3:
                continue
            group_signals = [syn for syn in syntheses_snapshot
                            if any(m in group for m in syn.source_modules)]

            if group_signals:
                emergent = self._generate_emergent_property(group, group_signals)
                if emergent and emergent.confidence >= min_confidence:
                    emergents.append(emergent)

        with self._stats_lock:
            self._stats["emergent_properties"] += len(emergents)

        return emergents

    def get_synthesis_history(
        self, limit: int = 20, domain: Optional[str] = None, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get synthesis history with optional filtering."""
        limit = max(1, min(200, limit))
        domain = _sanitize(domain, 30) if domain else None

        with self._lock:
            syntheses = list(self._syntheses.values())
            if domain:
                syntheses = [s for s in syntheses if s.domain == domain]
            if active_only:
                syntheses = [s for s in syntheses if s.is_active]

            syntheses.sort(key=lambda s: s.created_at, reverse=True)
            return [s.to_dict() for s in syntheses[:limit]]

    def get_synthesis_detail(self, synthesis_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view of a single synthesis."""
        synthesis_id = _sanitize(synthesis_id, 20)

        with self._lock:
            syn = self._syntheses.get(synthesis_id)
            if syn is None:
                return None
            result = syn.to_dict()

            related = []
            for other_id, other in self._syntheses.items():
                if other_id == synthesis_id:
                    continue
                overlap = set(syn.source_modules) & set(other.source_modules)
                if overlap:
                    related.append({
                        "synthesis_id": other.synthesis_id,
                        "title": other.title[:100],
                        "shared_modules": list(overlap),
                        "confidence": round(other.confidence, 3),
                    })
            result["related_syntheses"] = related[:10]
            return result

    def validate_synthesis(
        self, synthesis_id: str, outcome: str, ground_truth_confidence: float = 0.0
    ) -> Optional[SynthesisResult]:
        """Validate a synthesis against ground truth, updating confidence."""
        synthesis_id = _sanitize(synthesis_id, 20)
        outcome = _sanitize(outcome, 30)
        ground_truth_confidence = max(0.0, min(1.0, ground_truth_confidence))

        with self._lock:
            syn = self._syntheses.get(synthesis_id)
            if syn is None:
                return None

            syn.validation_count += 1
            validation_entry = {
                "synthesis_id": synthesis_id,
                "outcome": outcome,
                "ground_truth_confidence": ground_truth_confidence,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._validation_history.append(validation_entry)

            if outcome == "confirmed":
                syn.confidence = min(0.99, syn.confidence + 0.05)
            elif outcome == "partially_confirmed":
                syn.confidence = max(0.3, syn.confidence - 0.02)
                syn.validation_score = round(syn.validation_score * 0.7 + 0.3, 3)
            elif outcome == "refuted":
                syn.confidence = max(0.1, syn.confidence - 0.15)
                syn.validation_score = round(syn.validation_score * 0.5, 3)
                if syn.confidence < 0.2:
                    syn.is_active = False

            result = copy.deepcopy(syn)

        with self._stats_lock:
            self._stats["validations_performed"] += 1

        return result

    def get_trends(self, min_forecast_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """Get active trends with forecasts."""
        min_forecast_confidence = max(0.0, min(1.0, min_forecast_confidence))

        with self._lock:
            trends = [
                t.to_dict() for t in self._trends.values()
                if t.forecast_confidence >= min_forecast_confidence
            ]
        return trends

    def get_topic_clusters(self, max_clusters: int = 10) -> List[Dict[str, Any]]:
        """Cluster signals into topic groups for overview."""
        max_clusters = max(1, min(_MAX_TOPIC_CLUSTERS, max_clusters))

        with self._lock:
            signals_snapshot = list(self._signals)

        if not signals_snapshot:
            return []

        clusters = self._cluster_by_topic(signals_snapshot, max_clusters)
        return [
            {
                "topic": cluster["topic"],
                "signal_count": cluster["count"],
                "avg_confidence": round(cluster["avg_confidence"], 3),
                "top_source_modules": cluster["top_modules"][:5],
                "representative_content": cluster["representative"][:200],
            }
            for cluster in clusters[:max_clusters]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            total_signals = len(self._signals)
            total_syntheses = len(self._syntheses)
            active_syntheses = sum(1 for s in self._syntheses.values() if s.is_active)
            unique_modules = len(set(
                s.source_module for s in self._signals
            ))
            concept_nodes = len(self._concept_graph)
            concept_edges = sum(len(edges) for edges in self._concept_graph.values())

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_signals"] = total_signals
        stats["total_syntheses"] = total_syntheses
        stats["active_syntheses"] = active_syntheses
        stats["unique_modules_seen"] = unique_modules
        stats["concept_graph_nodes"] = concept_nodes
        stats["concept_graph_edges"] = concept_edges
        stats["trends_active"] = len(self._trends)
        return stats

    def _filter_signals(
        self,
        signals: List[ModuleSignal],
        modules: List[str],
        categories: List[str],
        min_confidence: float,
    ) -> List[ModuleSignal]:
        filtered = []
        for s in signals:
            if s.confidence < min_confidence:
                continue
            if modules and s.source_module not in modules:
                continue
            if categories and s.category not in categories:
                continue
            filtered.append(s)
        return filtered

    def _group_by_category(self, signals: List[ModuleSignal]) -> Dict[str, List[ModuleSignal]]:
        groups: Dict[str, List[ModuleSignal]] = defaultdict(list)
        for s in signals:
            groups[s.category].append(s)
        return groups

    def _find_category_pairs(
        self, category_groups: Dict[str, List[ModuleSignal]]
    ) -> List[Tuple[Tuple[str, str], List[ModuleSignal]]]:
        pairs = []
        categories = list(category_groups.keys())

        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat_a = categories[i]
                cat_b = categories[j]
                influence = CATEGORY_CROSS_INFLUENCE.get(
                    (cat_a, cat_b), 0.5
                ) or CATEGORY_CROSS_INFLUENCE.get((cat_b, cat_a), 0.5)

                if influence >= 0.5:
                    combined = category_groups[cat_a] + category_groups[cat_b]
                    pairs.append(((cat_a, cat_b), combined))

        pairs.sort(key=lambda p: CATEGORY_CROSS_INFLUENCE.get(
            (p[0][0], p[0][1]),
            CATEGORY_CROSS_INFLUENCE.get((p[0][1], p[0][0]), 0.5)
        ), reverse=True)

        return pairs[:20]

    def _generate_synthesis(
        self, signals: List[ModuleSignal], cat_pair: Tuple[str, str]
    ) -> Optional[SynthesisResult]:
        if not signals:
            return None

        modules = list(set(s.source_module for s in signals))
        signal_ids = list(set(s.signal_id for s in signals))
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        content_keywords = self._extract_keywords(
            " ".join(s.content[:500] for s in signals)
        )

        category_crosses = [(m1[:30], m2[:30]) for m1 in modules for m2 in modules
                           if m1 < m2
                           and MODULE_CATEGORY_MAP.get(m1) != MODULE_CATEGORY_MAP.get(m2)][:10]

        evidence = []
        for s in signals:
            if s.confidence >= 0.7:
                evidence.append(
                    f"{s.source_module} reports: {s.content[:150]} (confidence: {round(s.confidence, 3)})"
                )
        evidence = evidence[:10]

        title = f"Cross-category insight: {cat_pair[0]} ↔ {cat_pair[1]} via {len(modules)} modules"
        summary = (
            f"Synthesized from {len(signals)} signals across {len(modules)} modules "
            f"in categories {cat_pair[0]} and {cat_pair[1]}. "
            f"Concepts: {', '.join(content_keywords[:8])}. "
            f"Average confidence: {round(avg_confidence, 3)}."
        )

        key_finding = self._derive_key_finding(content_keywords, signals)
        action = self._generate_action_recommendation(content_keywords, cat_pair, modules)

        return SynthesisResult(
            domain="cross_module",
            source_modules=modules,
            source_signals=signal_ids,
            title=title,
            summary=summary,
            key_finding=key_finding,
            confidence=round(avg_confidence, 2),
            action_recommendation=action,
            expected_impact=round(avg_confidence * 0.8, 2),
            supporting_evidence=evidence,
            concepts_linked=content_keywords[:12],
            category_crosses=category_crosses,
        )

    def _generate_emergent_property(
        self, modules: List[str], signals: List[SynthesisResult]
    ) -> Optional[SynthesisResult]:
        if not signals:
            return None

        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        module_names = list(set(modules))

        title = f"Emergent: {len(module_names)} modules reveal hidden property"
        summary = (
            f"Modules {', '.join(module_names[:6])} collectively imply an "
            f"emergent property not visible in any single module. "
            f"Average synthesis confidence: {round(avg_confidence, 3)}."
        )

        return SynthesisResult(
            domain="emergent_property",
            source_modules=module_names,
            source_signals=[s.synthesis_id for s in signals[:20]],
            title=title,
            summary=summary,
            key_finding=f"Collective behavior of {len(module_names)} modules suggests novel insight",
            confidence=round(avg_confidence * 0.8, 2),
            action_recommendation="Investigate emergent property for potential system-level optimization",
            expected_impact=round(avg_confidence * 0.7, 2),
            concepts_linked=module_names[:12],
        )

    def _compute_contradiction_severity(self, sa: SynthesisResult, sb: SynthesisResult) -> float:
        shared_modules = set(sa.source_modules) & set(sb.source_modules)
        if not shared_modules:
            return 0.0

        shared_ratio = len(shared_modules) / max(1, len(set(sa.source_modules) | set(sb.source_modules)))
        confidence_diff = abs(sa.confidence - sb.confidence)
        severity = round(shared_ratio * 0.6 + confidence_diff * 0.4, 3)
        return severity

    def _suggest_contradiction_resolution(self, sa: SynthesisResult, sb: SynthesisResult) -> str:
        if sa.confidence > sb.confidence:
            preference = "A"
            higher_conf = sa
        else:
            preference = "B"
            higher_conf = sb

        return (
            f"Prefer synthesis {preference} (higher confidence: "
            f"{round(higher_conf.confidence, 3)} vs {round(sa.confidence if preference == 'B' else sb.confidence, 3)}). "
            f"Validate against ground truth."
        )

    def _update_trend(self, signal: ModuleSignal) -> None:
        topic = f"{signal.source_module}:{signal.signal_type}"

        with self._lock:
            self._trend_values[topic].append(signal.confidence)

            values = list(self._trend_values[topic])
            if len(values) < 2:
                return

            trend = self._trends.get(topic)
            if trend is None:
                trend = TrendSnapshot(topic=topic, start_at=signal.timestamp)
                self._trends[topic] = trend

            trend.values = values

            recent = values[-10:]
            if len(recent) >= 2:
                trend.rate_of_change = round(
                    (recent[-1] - recent[0]) / max(1, len(recent)), 4
                )

            if trend.rate_of_change > 0.05:
                trend.trend_direction = "increasing"
            elif trend.rate_of_change < -0.05:
                trend.trend_direction = "decreasing"
            else:
                trend.trend_direction = "stable"

            if len(values) >= 3:
                trend.forecast_next = round(
                    max(0.0, min(1.0, values[-1] + trend.rate_of_change * 3)), 3
                )
                trend.forecast_confidence = round(
                    max(0.0, min(1.0, len(values) / 30)), 3
                )

            while len(self._trends) > _MAX_PATTERNS:
                oldest_key = next(iter(self._trends))
                self._trends.pop(oldest_key, None)
                self._trend_values.pop(oldest_key, None)

    def _update_concept_graph(self, signal: ModuleSignal) -> None:
        with self._lock:
            for entity in signal.entities:
                clean_entity = entity.lower()
                if clean_entity not in self._concept_graph:
                    self._concept_graph[clean_entity] = {}

                for other_entity in signal.entities:
                    if other_entity == entity:
                        continue
                    clean_other = other_entity.lower()
                    current_weight = self._concept_graph[clean_entity].get(clean_other, 0.0)
                    new_weight = min(1.0, current_weight + 0.1 * signal.confidence)
                    self._concept_graph[clean_entity][clean_other] = round(new_weight, 4)

            if len(self._concept_graph) > _MAX_CONCEPT_GRAPH_NODES:
                oldest = next(iter(self._concept_graph))
                self._concept_graph.pop(oldest, None)
                for node_data in self._concept_graph.values():
                    node_data.pop(oldest, None)

    def _extract_keywords(self, text: str) -> List[str]:
        text_lower = text.lower()
        words = re.findall(r'[a-z][a-z\'-]*[a-z]', text_lower)
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "have",
            "been", "was", "are", "were", "will", "would", "could", "should",
            "what", "when", "where", "which", "who", "how", "about", "into",
            "over", "after", "before", "between", "through", "during", "above",
            "below", "each", "both", "few", "more", "most", "other", "some",
            "such", "only", "own", "same", "then", "than", "too", "very",
            "just", "also", "now", "here", "there", "been", "being", "does",
        }
        word_counts = Counter(w for w in words if w not in stop_words and len(w) > 2)
        return [w for w, _ in word_counts.most_common(30)]

    def _derive_key_finding(
        self, keywords: List[str], signals: List[ModuleSignal]
    ) -> str:
        modules = list(set(s.source_module for s in signals))
        top_keywords = keywords[:5]
        return (
            f"{len(signals)} signals from {len(modules)} modules "
            f"converge on: {', '.join(top_keywords)}. "
            f"Cross-category pattern suggests systemic insight."
        )

    def _generate_action_recommendation(
        self,
        keywords: List[str],
        cat_pair: Tuple[str, str],
        modules: List[str],
    ) -> str:
        cat_a, cat_b = cat_pair
        return (
            f"Investigate the {cat_a}↔{cat_b} connection. "
            f"Key modules: {', '.join(modules[:5])}. "
            f"Focus areas: {', '.join(keywords[:3])}. "
            f"Validate with ground truth from both categories."
        )

    def _cluster_related_modules(self, modules: List[str]) -> List[List[str]]:
        groups: List[List[str]] = []
        assigned: Set[str] = set()

        for m in modules:
            if m in assigned:
                continue
            cat_m = MODULE_CATEGORY_MAP.get(m, ModuleCategory.COGNITIVE).value
            group = [m]
            assigned.add(m)

            for other in modules:
                if other in assigned:
                    continue
                cat_other = MODULE_CATEGORY_MAP.get(other, ModuleCategory.COGNITIVE).value

                influence = CATEGORY_CROSS_INFLUENCE.get(
                    (cat_m, cat_other),
                    CATEGORY_CROSS_INFLUENCE.get((cat_other, cat_m), 0.0),
                )
                if influence >= 0.7:
                    group.append(other)
                    assigned.add(other)

            groups.append(group)

        return groups

    def _cluster_by_topic(
        self, signals: List[ModuleSignal], max_clusters: int
    ) -> List[Dict[str, Any]]:
        topic_groups: Dict[str, List[ModuleSignal]] = defaultdict(list)

        for s in signals:
            key = s.signal_type
            topic_groups[key].append(s)

        clusters = []
        for topic, topic_signals in topic_groups.items():
            confidence_sum = sum(s.confidence for s in topic_signals)
            module_counts = Counter(s.source_module for s in topic_signals)

            clusters.append({
                "topic": topic,
                "count": len(topic_signals),
                "avg_confidence": confidence_sum / len(topic_signals),
                "top_modules": [m for m, _ in module_counts.most_common(10)],
                "representative": topic_signals[0].content if topic_signals else "",
            })

        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters[:max_clusters]
