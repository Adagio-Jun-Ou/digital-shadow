#!/usr/bin/env python3
"""
CreativeSynthesis - Creative Idea Generation and Synthesis Engine

Generates novel ideas by combining disparate concepts, evaluates creative
potential, and tracks ideation history for pattern discovery.

Architecture:
  - CreativeIdea: a synthesized idea with sources, novelty score, and metadata
  - ConceptPair: two concepts being combined with association strength
  - IdeationPattern: recurring creative patterns across ideation sessions
  - CreativeSynthesis: main engine for creative idea generation
  - Thread-safe, bounded history, input sanitization
"""

import hashlib
import re
import copy
import uuid
import random
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set


logger = logging.getLogger("shadowd.cognition.creative_synthesis")

_MAX_IDEAS = 5000
_MAX_CONCEPTS = 2000
_MAX_PATTERNS = 500
_MAX_ASSOCIATIONS = 3000
_MAX_ASSOCIATIONS_PER_CONCEPT = 50
_MAX_METADATA_KEYS = 50

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_STRICT_RE.sub('', text).strip()
    return text[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_RE.sub('', text).strip()
    return text[:max_len]


class CreativeDomain(str, Enum):
    GENERAL = "general"
    TECHNOLOGY = "technology"
    ART = "art"
    SCIENCE = "science"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTH = "health"
    DESIGN = "design"
    STORYTELLING = "storytelling"
    PROBLEM_SOLVING = "problem_solving"


class SynthesisMethod(str, Enum):
    COMBINATION = "combination"
    ANALOGY = "analogy"
    INVERSION = "inversion"
    EXAGGERATION = "exaggeration"
    SUBTRACTION = "subtraction"
    REFRAMING = "reframing"
    CROSS_POLLINATION = "cross_pollination"
    CONSTRAINT_RELAXATION = "constraint_relaxation"


class NoveltyLevel(int, Enum):
    INCREMENTAL = 1
    MODERATE = 2
    NOVEL = 3
    BREAKTHROUGH = 4


@dataclass
class ConceptPair:
    concept_a: str
    concept_b: str
    domain: CreativeDomain = CreativeDomain.GENERAL
    association_strength: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        self.concept_a = _sanitize(self.concept_a, max_len=200)
        self.concept_b = _sanitize(self.concept_b, max_len=200)
        self.association_strength = max(0.0, min(1.0, self.association_strength))

    def to_dict(self) -> Dict:
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "domain": self.domain.value,
            "association_strength": round(self.association_strength, 3),
        }


@dataclass
class CreativeIdea:
    idea_id: str = ""
    title: str = ""
    description: str = ""
    source_concepts: List[str] = field(default_factory=list)
    synthesis_method: SynthesisMethod = SynthesisMethod.COMBINATION
    domain: CreativeDomain = CreativeDomain.GENERAL
    novelty_score: float = 0.5
    feasibility_score: float = 0.5
    impact_score: float = 0.5
    novelty_level: NoveltyLevel = NoveltyLevel.MODERATE
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.idea_id:
            self.idea_id = f"idea_{uuid.uuid4().hex[:12]}"
        self.title = _sanitize(self.title, max_len=200)
        self.description = _sanitize_content(self.description, max_len=5000)
        self.novelty_score = max(0.0, min(1.0, self.novelty_score))
        self.feasibility_score = max(0.0, min(1.0, self.feasibility_score))
        self.impact_score = max(0.0, min(1.0, self.impact_score))
        self.source_concepts = [_sanitize(c, max_len=200) for c in self.source_concepts[:10]]
        self.tags = [_sanitize(t, max_len=50) for t in self.tags[:20]]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    @property
    def composite_score(self) -> float:
        return (self.novelty_score * 0.4 + self.feasibility_score * 0.3 + self.impact_score * 0.3)

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "title": self.title,
            "description": self.description[:500],
            "source_concepts": list(self.source_concepts),
            "synthesis_method": self.synthesis_method.value,
            "domain": self.domain.value,
            "novelty_score": round(self.novelty_score, 3),
            "feasibility_score": round(self.feasibility_score, 3),
            "impact_score": round(self.impact_score, 3),
            "composite_score": round(self.composite_score, 3),
            "novelty_level": self.novelty_level.value,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


@dataclass
class IdeationPattern:
    pattern_id: str = ""
    name: str = ""
    description: str = ""
    method: SynthesisMethod = SynthesisMethod.COMBINATION
    concepts: List[str] = field(default_factory=list)
    frequency: int = 0
    avg_novelty: float = 0.5
    avg_impact: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = f"ipat_{uuid.uuid4().hex[:12]}"
        self.name = _sanitize(self.name, max_len=200)
        self.description = _sanitize_content(self.description, max_len=500)
        self.avg_novelty = max(0.0, min(1.0, self.avg_novelty))
        self.avg_impact = max(0.0, min(1.0, self.avg_impact))

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "method": self.method.value,
            "concepts": list(self.concepts[:10]),
            "frequency": self.frequency,
            "avg_novelty": round(self.avg_novelty, 3),
            "avg_impact": round(self.avg_impact, 3),
        }


_COMBINATION_TEMPLATES = [
    "What if {a} and {b} were combined?",
    "How could {a} enhance {b}?",
    "What would {a} look like if it adopted principles from {b}?",
    "Can {a} solve problems traditionally handled by {b}?",
    "What happens when {a} meets {b}?",
]

_INVERSION_TEMPLATES = [
    "What if {a} worked in the opposite way?",
    "How would reversing {a} create new possibilities?",
    "What is the anti-{a} and what can it teach us?",
    "If {a} is the problem, what is the opposite solution?",
]

_ANALOGY_TEMPLATES = [
    "How is {a} like {b}?",
    "What can {a} learn from how {b} works?",
    "If {b} were a metaphor for {a}, what insights emerge?",
    "What structural similarities exist between {a} and {b}?",
]

_SUBTRACTION_TEMPLATES = [
    "What if we removed the core feature of {a}?",
    "What remains when you strip {a} down to its essence?",
    "What opportunities arise from removing {a} entirely?",
]


class CreativeSynthesis:
    """
    Creative Idea Generation and Synthesis Engine.

    Combines concepts, generates novel ideas through various synthesis methods,
    evaluates creative potential, and tracks ideation patterns.
    """

    def __init__(self):
        self._ideas: Dict[str, CreativeIdea] = {}
        self._concepts: Set[str] = set()
        self._concept_order: List[str] = []
        self._associations: Dict[str, List[ConceptPair]] = {}
        self._patterns: Dict[str, IdeationPattern] = {}
        self._lock = threading.Lock()
        self._stats = {
            "ideas_generated": 0,
            "concepts_registered": 0,
            "syntheses_performed": 0,
            "avg_novelty": 0.0,
            "avg_feasibility": 0.0,
            "avg_impact": 0.0,
        }
        self._stats_lock = threading.Lock()

    def register_concept(self, concept: str, domain: CreativeDomain = CreativeDomain.GENERAL) -> bool:
        concept = _sanitize(concept.lower().strip(), max_len=200)
        if not concept or len(concept) < 2:
            return False

        with self._lock:
            return self._register_concept_unlocked(concept)

    def _register_concept_unlocked(self, concept: str) -> bool:
        if concept in self._concepts:
            return False
        self._concepts.add(concept)
        self._concept_order.append(concept)
        self._stats["concepts_registered"] = len(self._concepts)
        if len(self._concepts) > _MAX_CONCEPTS:
            excess = len(self._concepts) - _MAX_CONCEPTS
            to_remove = self._concept_order[:excess]
            self._concept_order = self._concept_order[excess:]
            for c in to_remove:
                logger.warning("Evicting concept '%s' and its associations", c)
                self._concepts.discard(c)
                self._associations.pop(c, None)
                for k in list(self._associations.keys()):
                    self._associations[k] = [p for p in self._associations[k] if p.concept_a != c and p.concept_b != c]
        return True

    def add_association(self, concept_a: str, concept_b: str,
                         domain: CreativeDomain = CreativeDomain.GENERAL,
                         strength: float = 0.5) -> ConceptPair:
        concept_a = _sanitize(concept_a.lower().strip(), max_len=200)
        concept_b = _sanitize(concept_b.lower().strip(), max_len=200)
        strength = max(0.0, min(1.0, strength))

        pair = ConceptPair(
            concept_a=concept_a,
            concept_b=concept_b,
            domain=domain,
            association_strength=strength,
        )

        with self._lock:
            self._register_concept_unlocked(concept_a)
            self._register_concept_unlocked(concept_b)

            for concept in (concept_a, concept_b):
                if concept not in self._associations:
                    self._associations[concept] = []
                existing = self._associations[concept]
                if not any(p.concept_a == pair.concept_a and p.concept_b == pair.concept_b for p in existing):
                    existing.append(pair)
                if len(existing) > _MAX_ASSOCIATIONS_PER_CONCEPT:
                    self._associations[concept] = existing[-_MAX_ASSOCIATIONS_PER_CONCEPT:]

            if len(self._associations) > _MAX_ASSOCIATIONS:
                excess_keys = list(self._associations.keys())[:len(self._associations) - _MAX_ASSOCIATIONS]
                for k in excess_keys:
                    logger.warning("Evicting association key %s", k)
                    del self._associations[k]

        return copy.deepcopy(pair)

    def synthesize(self, concepts: List[str], method: SynthesisMethod = SynthesisMethod.COMBINATION,
                     domain: CreativeDomain = CreativeDomain.GENERAL) -> CreativeIdea:
        concepts = [_sanitize(c.lower().strip(), max_len=200) for c in concepts[:5]]

        novelty, feasibility, impact = self._evaluate_idea(concepts, method)

        if novelty > 0.8:
            novelty_level = NoveltyLevel.BREAKTHROUGH
        elif novelty > 0.6:
            novelty_level = NoveltyLevel.NOVEL
        elif novelty > 0.3:
            novelty_level = NoveltyLevel.MODERATE
        else:
            novelty_level = NoveltyLevel.INCREMENTAL

        title, description = self._generate_description(concepts, method)

        with self._lock:
            for c in concepts:
                self._register_concept_unlocked(c)

        idea = CreativeIdea(
            title=title,
            description=description,
            source_concepts=concepts,
            synthesis_method=method,
            domain=domain,
            novelty_score=novelty,
            feasibility_score=feasibility,
            impact_score=impact,
            novelty_level=novelty_level,
            tags=self._extract_tags(concepts),
        )

        with self._lock:
            self._ideas[idea.idea_id] = idea
            if len(self._ideas) > _MAX_IDEAS:
                oldest = list(self._ideas.keys())[0]
                logger.warning("Evicting idea %s", oldest)
                del self._ideas[oldest]
            with self._stats_lock:
                self._stats["ideas_generated"] += 1
            with self._stats_lock:
                self._stats["syntheses_performed"] += 1
            total = self._stats["ideas_generated"]
            with self._stats_lock:
                self._stats["avg_novelty"] += (novelty - self._stats["avg_novelty"]) / total
            with self._stats_lock:
                self._stats["avg_feasibility"] += (feasibility - self._stats["avg_feasibility"]) / total
            with self._stats_lock:
                self._stats["avg_impact"] += (impact - self._stats["avg_impact"]) / total

        self._track_pattern(concepts, method, novelty, impact)

        return copy.deepcopy(idea)

    def _evaluate_idea(self, concepts: List[str], method: SynthesisMethod) -> Tuple[float, float, float]:
        novelty = 0.3
        feasibility = 0.5
        impact = 0.4

        with self._lock:
            known_count = sum(1 for c in concepts if c in self._concepts)
            associations_snapshot = {c: list(self._associations.get(c, [])) for c in concepts}

        domain_distance = 0.0
        if len(concepts) >= 2:
            unique_domains = set()
            for c, pairs in associations_snapshot.items():
                for pair in pairs:
                    unique_domains.add(pair.domain)
            if len(unique_domains) <= 1:
                domain_distance = 0.2
            elif len(unique_domains) <= 2:
                domain_distance = 0.5
            else:
                domain_distance = 0.8

        method_novelty_bonus = {
            SynthesisMethod.COMBINATION: 0.1,
            SynthesisMethod.ANALOGY: 0.3,
            SynthesisMethod.INVERSION: 0.4,
            SynthesisMethod.EXAGGERATION: 0.2,
            SynthesisMethod.SUBTRACTION: 0.35,
            SynthesisMethod.REFRAMING: 0.25,
            SynthesisMethod.CROSS_POLLINATION: 0.5,
            SynthesisMethod.CONSTRAINT_RELAXATION: 0.15,
        }
        novelty += method_novelty_bonus.get(method, 0.1)
        novelty += domain_distance * 0.3
        novelty += (1 - known_count / max(1, len(concepts))) * 0.2

        if len(concepts) >= 2:
            feasibility -= 0.1 * (len(concepts) - 1)
        feasibility += known_count / max(1, len(concepts)) * 0.2

        impact += domain_distance * 0.2
        impact += novelty * 0.2

        novelty = max(0.0, min(1.0, novelty))
        feasibility = max(0.1, min(1.0, feasibility))
        impact = max(0.1, min(1.0, impact))

        return novelty, feasibility, impact

    def _generate_description(self, concepts: List[str], method: SynthesisMethod) -> Tuple[str, str]:
        if not concepts:
            return "Untitled Idea", "No concepts provided for synthesis."

        if method == SynthesisMethod.COMBINATION:
            templates = _COMBINATION_TEMPLATES
        elif method == SynthesisMethod.INVERSION:
            templates = _INVERSION_TEMPLATES
        elif method == SynthesisMethod.ANALOGY:
            templates = _ANALOGY_TEMPLATES
        elif method == SynthesisMethod.SUBTRACTION:
            templates = _SUBTRACTION_TEMPLATES
        else:
            templates = _COMBINATION_TEMPLATES

        template = templates[int(hashlib.sha256(str(tuple(concepts) + (method.value,)).encode()).hexdigest()[:8], 16) % len(templates)]

        a = concepts[0] if concepts else "concept_a"
        b = concepts[1] if len(concepts) > 1 else "concept_b"

        title = f"{method.value.title()}: {a.title()} × {b.title()}"
        description = template.replace("{b}", b).replace("{a}", a)

        extra = []
        for c in concepts[2:]:
            extra.append(f"Enhanced by {c}")
        if extra:
            description += " " + " ".join(extra)

        return title[:200], description[:5000]

    def _extract_tags(self, concepts: List[str]) -> List[str]:
        tags = list(concepts[:5])
        for c in concepts:
            words = c.split()
            if len(words) > 1:
                tags.extend(words[:3])
        return list(set(tags))[:20]

    def _track_pattern(self, concepts: List[str], method: SynthesisMethod,
                        novelty: float, impact: float) -> None:
        key = f"{method.value}|{'|'.join(sorted(concepts))}"
        with self._lock:
            if key in self._patterns:
                pat = self._patterns[key]
                pat.frequency += 1
                pat.avg_novelty = (pat.avg_novelty * (pat.frequency - 1) + novelty) / pat.frequency
                pat.avg_impact = (pat.avg_impact * (pat.frequency - 1) + impact) / pat.frequency
            else:
                pat = IdeationPattern(
                    name=f"{method.value.title()}: {' + '.join(concepts[:3])}",
                    description=f"Pattern for {method.value} synthesis of {', '.join(concepts[:3])}",
                    method=method,
                    concepts=concepts[:5],
                    frequency=1,
                    avg_novelty=novelty,
                    avg_impact=impact,
                )
                self._patterns[key] = pat
                if len(self._patterns) > _MAX_PATTERNS:
                    oldest = list(self._patterns.keys())[0]
                    logger.warning("Evicting pattern %s", oldest)
                    del self._patterns[oldest]

    def random_synthesis(self, method: SynthesisMethod = SynthesisMethod.COMBINATION,
                           domain: CreativeDomain = CreativeDomain.GENERAL) -> Optional[CreativeIdea]:
        with self._lock:
            concepts_list = list(self._concepts)

        if len(concepts_list) < 2:
            return None

        num_concepts = min(len(concepts_list), random.randint(2, 4))
        selected = random.sample(concepts_list, num_concepts)

        return self.synthesize(selected, method, domain)

    def brainstorm(self, seed_concept: str, method: SynthesisMethod = SynthesisMethod.CROSS_POLLINATION,
                     domain: CreativeDomain = CreativeDomain.GENERAL,
                     count: int = 5) -> List[CreativeIdea]:
        seed_concept = _sanitize(seed_concept.lower().strip(), max_len=200)
        count = max(1, min(10, count))

        with self._lock:
            related = self._associations.get(seed_concept, [])
            related_concepts = list(set(
                p.concept_b if p.concept_a == seed_concept else p.concept_a
                for p in related
            ))
            all_concepts = list(self._concepts)

        if not related_concepts and not all_concepts:
            self.register_concept(seed_concept, domain)
            return []

        ideas = []
        methods = [method]
        if method == SynthesisMethod.CROSS_POLLINATION:
            methods = [SynthesisMethod.COMBINATION, SynthesisMethod.ANALOGY,
                        SynthesisMethod.INVERSION, SynthesisMethod.REFRAMING]

        for i in range(count):
            if related_concepts:
                partner = random.choice(related_concepts)
            elif all_concepts:
                partner = random.choice(all_concepts)
            else:
                continue

            m = methods[i % len(methods)]
            idea = self.synthesize([seed_concept, partner], m, domain)
            ideas.append(idea)

        return ideas

    def get_ideas(self, domain: Optional[CreativeDomain] = None,
                   min_novelty: float = 0.0, limit: int = 20) -> List[Dict]:
        with self._lock:
            ideas = list(self._ideas.values())

        if domain:
            ideas = [i for i in ideas if i.domain == domain]
        if min_novelty > 0:
            ideas = [i for i in ideas if i.novelty_score >= min_novelty]

        ideas.sort(key=lambda i: i.composite_score, reverse=True)
        return [i.to_dict() for i in ideas[:limit]]

    def get_patterns(self, method: Optional[SynthesisMethod] = None, limit: int = 20) -> List[Dict]:
        with self._lock:
            patterns = list(self._patterns.values())

        if method:
            patterns = [p for p in patterns if p.method == method]

        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return [p.to_dict() for p in patterns[:limit]]

    def get_concept_associations(self, concept: str) -> List[Dict]:
        concept = _sanitize(concept.lower().strip(), max_len=200)
        with self._lock:
            pairs = self._associations.get(concept, [])
            return [p.to_dict() for p in pairs[:20]]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            ideas_count = len(self._ideas)
            concepts_count = len(self._concepts)
            associations_count = sum(len(v) for v in self._associations.values())
            patterns_count = len(self._patterns)
        return {
            "stats": stats,
            "ideas_count": ideas_count,
            "concepts_count": concepts_count,
            "associations_count": associations_count,
            "patterns_count": patterns_count,
        }