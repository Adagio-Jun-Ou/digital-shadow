import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_INSPIRATIONS = 500
_MAX_INCUBATIONS = 500
_MAX_CYCLES = 200
_MAX_INCUBATION_INDEX = 500
_DREAM_NOVELTY_THRESHOLD = 0.6
_INSIGHT_CONFIDENCE_THRESHOLD = 0.3
_PATTERN_FREQUENCY_THRESHOLD = 2

_SANITIZE_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_RE.sub('', text).strip()
    return text[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_CONTENT_RE.sub('', text).strip()
    return text[:max_len]


@dataclass
class DreamInspiration:
    inspiration_id: str
    source_insight_id: str
    source_pattern: str
    concept_seeds: List[str]
    emotional_valence: float
    novelty_potential: float
    created_at: str

    def __post_init__(self):
        if not self.inspiration_id:
            self.inspiration_id = uuid.uuid4().hex[:12]
        self.source_insight_id = _sanitize(self.source_insight_id, max_len=200)
        self.source_pattern = _sanitize_content(self.source_pattern, max_len=500)
        self.concept_seeds = [_sanitize(s, max_len=200) for s in self.concept_seeds[:20]]
        self.emotional_valence = max(-1.0, min(1.0, self.emotional_valence))
        self.novelty_potential = max(0.0, min(1.0, self.novelty_potential))

    def to_dict(self) -> Dict:
        return {
            "inspiration_id": self.inspiration_id,
            "source_insight_id": self.source_insight_id,
            "source_pattern": self.source_pattern,
            "concept_seeds": list(self.concept_seeds),
            "emotional_valence": round(self.emotional_valence, 3),
            "novelty_potential": round(self.novelty_potential, 3),
            "created_at": self.created_at,
        }


@dataclass
class CreativityIncubation:
    incubation_id: str
    source_idea_id: str
    concept: str
    domain: str
    incubation_status: str
    first_dreamed_at: str
    reinforcement_count: int
    last_reinforced: str

    def __post_init__(self):
        if not self.incubation_id:
            self.incubation_id = uuid.uuid4().hex[:12]
        self.source_idea_id = _sanitize(self.source_idea_id, max_len=200)
        self.concept = _sanitize_content(self.concept, max_len=500)
        self.domain = _sanitize(self.domain, max_len=100)
        self.incubation_status = _sanitize(self.incubation_status, max_len=50)
        self.reinforcement_count = max(0, self.reinforcement_count)

    def to_dict(self) -> Dict:
        return {
            "incubation_id": self.incubation_id,
            "source_idea_id": self.source_idea_id,
            "concept": self.concept,
            "domain": self.domain,
            "incubation_status": self.incubation_status,
            "first_dreamed_at": self.first_dreamed_at,
            "reinforcement_count": self.reinforcement_count,
            "last_reinforced": self.last_reinforced,
        }


@dataclass
class CreativeDreamCycle:
    cycle_id: str
    dream_inspirations_count: int
    creative_outputs_count: int
    incubated_count: int
    resonance_score: float
    started_at: str
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.cycle_id:
            self.cycle_id = uuid.uuid4().hex[:12]
        self.resonance_score = max(0.0, min(1.0, self.resonance_score))

    def to_dict(self) -> Dict:
        return {
            "cycle_id": self.cycle_id,
            "dream_inspirations_count": self.dream_inspirations_count,
            "creative_outputs_count": self.creative_outputs_count,
            "incubated_count": self.incubated_count,
            "resonance_score": round(self.resonance_score, 3),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class CreativeDreamIntegration:
    def __init__(
        self,
        dream_state: Optional["DreamState"] = None,
        creative_synthesis: Optional["CreativeSynthesis"] = None,
    ):
        self._dream_state = dream_state
        self._creative_synthesis = creative_synthesis
        self._inspirations: deque = deque(maxlen=_MAX_INSPIRATIONS)
        self._incubations: deque = deque(maxlen=_MAX_INCUBATIONS)
        self._cycles: deque = deque(maxlen=_MAX_CYCLES)
        self._incubation_index: Dict[str, CreativityIncubation] = {}
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "dream_to_create_calls": 0,
            "create_to_dream_calls": 0,
            "cycles_run": 0,
            "total_inspirations": 0,
            "total_incubations": 0,
            "total_reinforcements": 0,
        }

    def set_modules(self, dream_state, creative_synthesis):
        with self._lock:
            self._dream_state = dream_state
            self._creative_synthesis = creative_synthesis

    def dream_to_create(self) -> List[DreamInspiration]:
        inspirations: List[DreamInspiration] = []
        if self._dream_state is None:
            with self._stats_lock:
                self._stats["dream_to_create_calls"] += 1
            return inspirations

        insights = self._dream_state.get_insights()
        patterns = self._dream_state.get_patterns()

        for insight in insights:
            if insight.confidence < _INSIGHT_CONFIDENCE_THRESHOLD:
                continue
            seeds = self._extract_concept_seeds(insight.description)
            if not seeds:
                domains = list(insight.domains_connected) if insight.domains_connected else []
                seeds = domains[:5] if domains else ["general"]
            novelty = min(1.0, insight.confidence + 0.2)
            valence = 0.5 if insight.confidence > 0.6 else -0.2

            if self._creative_synthesis is not None:
                for seed in seeds[:3]:
                    try:
                        self._creative_synthesis.register_concept(seed)
                    except (ValueError, TypeError, AttributeError, RuntimeError):
                        pass
                try:
                    from .creative_synthesis import SynthesisMethod
                    self._creative_synthesis.synthesize(
                        seeds[:3],
                        method=SynthesisMethod.CROSS_POLLINATION,
                    )
                except (ValueError, TypeError, AttributeError, RuntimeError, ImportError):
                    pass
                try:
                    self._creative_synthesis.brainstorm(
                        seeds[0],
                        count=3,
                    )
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    pass

            inspiration = DreamInspiration(
                inspiration_id=uuid.uuid4().hex[:12],
                source_insight_id=insight.insight_id,
                source_pattern=insight.description[:500],
                concept_seeds=seeds,
                emotional_valence=valence,
                novelty_potential=novelty,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            inspirations.append(inspiration)

        for pattern in patterns:
            if pattern.frequency < _PATTERN_FREQUENCY_THRESHOLD:
                continue
            seeds = [pattern.theme] if pattern.theme else []
            if pattern.domains:
                seeds.extend(pattern.domains[:3])
            if not seeds:
                continue
            novelty = min(1.0, pattern.avg_importance + 0.3)
            valence = 0.3

            if self._creative_synthesis is not None:
                for seed in seeds[:3]:
                    try:
                        self._creative_synthesis.register_concept(seed)
                    except (ValueError, TypeError, AttributeError, RuntimeError):
                        pass

            inspiration = DreamInspiration(
                inspiration_id=uuid.uuid4().hex[:12],
                source_insight_id=pattern.pattern_id,
                source_pattern=f"Pattern: {pattern.theme} (freq={pattern.frequency})",
                concept_seeds=seeds,
                emotional_valence=valence,
                novelty_potential=novelty,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            inspirations.append(inspiration)

        with self._lock:
            for insp in inspirations:
                self._inspirations.append(copy.deepcopy(insp))
            with self._stats_lock:
                self._stats["dream_to_create_calls"] += 1
                self._stats["total_inspirations"] += len(inspirations)

        return inspirations

    def create_to_dream(self) -> List[CreativityIncubation]:
        incubations: List[CreativityIncubation] = []
        if self._creative_synthesis is None:
            with self._stats_lock:
                self._stats["create_to_dream_calls"] += 1
            return incubations

        ideas = self._creative_synthesis.get_ideas(min_novelty=_DREAM_NOVELTY_THRESHOLD, limit=50)

        for idea_dict in ideas:
            novelty = idea_dict.get("novelty_score", 0.0)
            if novelty < _DREAM_NOVELTY_THRESHOLD:
                continue
            idea_id = idea_dict.get("idea_id", "")
            title = idea_dict.get("title", "")
            domain = idea_dict.get("domain", "general")
            concept = title if title else idea_dict.get("description", "")[:200]
            tags = idea_dict.get("tags", [])
            now = datetime.now(timezone.utc).isoformat()

            with self._lock:
                existing_id = None
                for inc in self._incubations:
                    if inc.source_idea_id == idea_id:
                        existing_id = inc.incubation_id
                        break

                if existing_id is not None:
                    inc = self._incubation_index.get(existing_id)
                    if inc is not None:
                        inc.reinforcement_count += 1
                        inc.last_reinforced = now
                        inc.domain = domain
                    with self._stats_lock:
                        self._stats["total_reinforcements"] += 1
                    if inc is not None:
                        incubations.append(copy.deepcopy(inc))
                    continue

                if self._dream_state is not None:
                    try:
                        memory_tags = tags[:5] if tags else []
                        self._dream_state.add_memory(
                            content=f"Creative idea: {concept}",
                            domain=domain,
                            importance=min(1.0, novelty),
                            tags=memory_tags,
                        )
                    except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                        logger.warning("Failed to add dream memory: %s", type(e).__name__)

                incubation = CreativityIncubation(
                    incubation_id=uuid.uuid4().hex[:12],
                    source_idea_id=idea_id,
                    concept=concept,
                    domain=domain,
                    incubation_status="seeded",
                    first_dreamed_at=now,
                    reinforcement_count=1,
                    last_reinforced=now,
                )

                self._incubations.append(incubation)
                self._incubation_index[incubation.incubation_id] = incubation
                if len(self._incubation_index) > _MAX_INCUBATION_INDEX:
                    stale_ids = set(self._incubation_index.keys()) - {inc.incubation_id for inc in self._incubations}
                    for stale_id in list(stale_ids)[:len(self._incubation_index) - _MAX_INCUBATION_INDEX]:
                        self._incubation_index.pop(stale_id, None)
                with self._stats_lock:
                    self._stats["create_to_dream_calls"] += 1
                    self._stats["total_incubations"] += 1

                incubations.append(copy.deepcopy(incubation))

        with self._stats_lock:
            self._stats["create_to_dream_calls"] += 1

        return incubations

    def run_cycle(self) -> CreativeDreamCycle:
        started = datetime.now(timezone.utc).isoformat()
        inspirations = self.dream_to_create()
        incubations = self.create_to_dream()

        insp_count = len(inspirations)
        inc_count = len(incubations)
        creative_count = 0
        if self._creative_synthesis is not None:
            creative_count = len(self._creative_synthesis.get_ideas(min_novelty=_DREAM_NOVELTY_THRESHOLD, limit=10))

        if insp_count + inc_count > 0:
            resonance = min(1.0, (insp_count + inc_count) / 10.0)
        else:
            resonance = 0.0

        cycle = CreativeDreamCycle(
            cycle_id=uuid.uuid4().hex[:12],
            dream_inspirations_count=insp_count,
            creative_outputs_count=creative_count,
            incubated_count=inc_count,
            resonance_score=resonance,
            started_at=started,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._cycles.append(cycle)
            with self._stats_lock:
                self._stats["cycles_run"] += 1

        return copy.deepcopy(cycle)

    def _extract_concept_seeds(self, description: str) -> List[str]:
        if not description:
            return []
        words = re.findall(r'\b\w{4,}\b', description.lower())
        return list(dict.fromkeys(words))[:10]

    def get_inspirations(self, limit: int = 20) -> List[Dict]:
        limit = max(1, min(200, limit))
        with self._lock:
            items = list(self._inspirations)
        return [i.to_dict() for i in items[-limit:]]

    def get_incubations(self, limit: int = 20) -> List[Dict]:
        limit = max(1, min(200, limit))
        with self._lock:
            items = list(self._incubations)
        return [i.to_dict() for i in items[-limit:]]

    def get_cycles(self, limit: int = 20) -> List[Dict]:
        limit = max(1, min(200, limit))
        with self._lock:
            items = list(self._cycles)
        return [c.to_dict() for c in items[-limit:]]

    def get_stats(self) -> Dict:
        with self._lock:
            inspirations_count = len(self._inspirations)
            incubations_count = len(self._incubations)
            cycles_count = len(self._cycles)
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update({
            "inspirations_count": inspirations_count,
            "incubations_count": incubations_count,
            "cycles_count": cycles_count,
            "dream_state_wired": self._dream_state is not None,
            "creative_synthesis_wired": self._creative_synthesis is not None,
        })
        return stats