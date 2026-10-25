"""
RecallSynthesis - Cross-Domain Pattern Synthesis Engine

Detects hidden connections between observations across different life domains
and time periods. Like a personal "Sherlock Holmes" that notices what others miss.

Finds "aha moment" patterns where keywords appear across multiple domains,
enabling insights that single-domain analysis would never surface.
"""

import copy
import re
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


LIFE_DOMAINS = [
    "health", "finance", "career", "growth",
    "home", "admin", "legal", "social",
]

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 0) -> str:
    if text is None:
        return ""
    result = _SANITIZE_RE.sub('', str(text)).strip()
    if max_len > 0:
        result = result[:max_len]
    return result


def _sanitize_strict(text: str, max_len: int = 0) -> str:
    if text is None:
        return ""
    result = _SANITIZE_STRICT_RE.sub('', str(text)).strip()
    if max_len > 0:
        result = result[:max_len]
    return result

DOMAIN_PROXIMITY = {
    ("health", "growth"): 0.8,
    ("growth", "health"): 0.8,
    ("career", "finance"): 0.9,
    ("finance", "career"): 0.9,
    ("home", "admin"): 0.7,
    ("admin", "home"): 0.7,
    ("career", "growth"): 0.6,
    ("growth", "career"): 0.6,
    ("social", "health"): 0.5,
    ("health", "social"): 0.5,
    ("social", "career"): 0.4,
    ("career", "social"): 0.4,
    ("finance", "admin"): 0.6,
    ("admin", "finance"): 0.6,
    ("legal", "finance"): 0.7,
    ("finance", "legal"): 0.7,
    ("legal", "admin"): 0.5,
    ("admin", "legal"): 0.5,
    ("health", "home"): 0.4,
    ("home", "health"): 0.4,
    ("health", "finance"): 0.3,
    ("finance", "health"): 0.3,
    ("health", "career"): 0.3,
    ("career", "health"): 0.3,
    ("health", "admin"): 0.2,
    ("admin", "health"): 0.2,
    ("health", "legal"): 0.3,
    ("legal", "health"): 0.3,
    ("finance", "growth"): 0.3,
    ("growth", "finance"): 0.3,
    ("finance", "home"): 0.2,
    ("home", "finance"): 0.2,
    ("finance", "social"): 0.2,
    ("social", "finance"): 0.2,
    ("career", "home"): 0.2,
    ("home", "career"): 0.2,
    ("career", "admin"): 0.3,
    ("admin", "career"): 0.3,
    ("career", "legal"): 0.3,
    ("legal", "career"): 0.3,
    ("growth", "home"): 0.2,
    ("home", "growth"): 0.2,
    ("growth", "admin"): 0.3,
    ("admin", "growth"): 0.3,
    ("growth", "legal"): 0.2,
    ("legal", "growth"): 0.2,
    ("growth", "social"): 0.4,
    ("social", "growth"): 0.4,
    ("home", "legal"): 0.2,
    ("legal", "home"): 0.2,
    ("home", "social"): 0.2,
    ("social", "home"): 0.2,
    ("admin", "social"): 0.2,
    ("social", "admin"): 0.2,
    ("legal", "social"): 0.2,
    ("social", "legal"): 0.2,
}

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "that", "this", "these", "those", "what", "which", "who", "whom",
    "whose", "where", "when", "how", "why", "if", "then", "else",
    "and", "but", "or", "nor", "not", "so", "yet", "for", "with",
    "from", "into", "onto", "upon", "about", "above", "below",
})


@dataclass
class Observation:
    observation_id: str
    text: str
    domain: Optional[str]
    timestamp: str
    importance: float
    keywords: List[str]

    def to_dict(self):
        d = asdict(self)
        d["keywords"] = list(self.keywords)
        d["importance"] = round(self.importance, 3)
        return d


@dataclass
class CrossDomainPattern:
    pattern_id: str
    keywords: List[str]
    domains_involved: List[str]
    observations_linked: List[str]
    confidence: float
    description: str

    def to_dict(self):
        d = asdict(self)
        d["keywords"] = list(self.keywords)
        d["domains_involved"] = list(self.domains_involved)
        d["observations_linked"] = list(self.observations_linked)
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class SynthesisResult:
    query: str
    direct_observations: List[Dict]
    cross_domain_insights: List[Dict]
    connections: List[Dict]
    confidence: float

    def to_dict(self):
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        return copy.deepcopy(d)


@dataclass
class Insight:
    domain: Optional[str]
    description: str
    confidence: float
    source_observations: List[str]
    created_at: str
    domains: List[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["source_observations"] = list(self.source_observations)
        d["domains"] = list(self.domains)
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class ConnectionResult:
    concept_a: str
    concept_b: str
    chain: List[Dict]
    confidence: float
    hops: int

    def to_dict(self):
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        return copy.deepcopy(d)


@dataclass
class DomainConnection:
    domain_a: str
    domain_b: str
    strength: float
    shared_keywords: List[str]
    connection_count: int

    def to_dict(self):
        d = asdict(self)
        d["shared_keywords"] = list(self.shared_keywords)
        d["strength"] = round(self.strength, 3)
        return d


def _extract_keywords(text: str) -> List[str]:
    words = text.split()
    keywords = []
    for w in words:
        lower = w.lower().strip(".,;:!?\"'()[]{}")
        if len(lower) < 4 and not w[0].isupper():
            continue
        if lower in STOP_WORDS:
            continue
        if lower.replace('.', '').replace(',', '').isdigit():
            continue
        if lower:
            keywords.append(lower)
    return keywords


class RecallSynthesis:
    """
    Cross-Domain Pattern Synthesis Engine

    Detects hidden connections between observations across different
    life domains and time periods. Like a personal "Sherlock Holmes"
    that notices what others miss.
    """

    _MAX_OBSERVATIONS = 5000
    _MAX_PATTERNS = 1000
    _MAX_INSIGHTS = 500
    _MAX_TEXT_LENGTH = 10000
    _MAX_DOMAIN_LENGTH = 50
    _MAX_QUERY_LENGTH = 1000
    _MAX_CONCEPT_LENGTH = 200
    _VALID_DOMAINS = frozenset(LIFE_DOMAINS)

    def __init__(self):
        self._observations: Dict[str, Observation] = {}
        self._patterns: Dict[str, CrossDomainPattern] = {}
        self._insights: Dict[str, Insight] = {}
        self._observation_order: List[str] = []
        self._pattern_order: List[str] = []
        self._insight_order: List[str] = []
        self._lock = threading.Lock()

    def add_observation(
        self,
        text: str,
        domain: str = None,
        timestamp: str = None,
        importance: float = 0.5,
    ) -> str:
        text = _sanitize(text, self._MAX_TEXT_LENGTH)
        if domain:
            domain = _sanitize_strict(str(domain).strip(), self._MAX_DOMAIN_LENGTH)
            if domain.lower() not in self._VALID_DOMAINS:
                domain = None
        importance = max(0.0, min(1.0, importance))
        obs_id = uuid.uuid4().hex[:12]
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        keywords = _extract_keywords(text)

        obs = Observation(
            observation_id=obs_id,
            text=text,
            domain=domain,
            timestamp=ts,
            importance=importance,
            keywords=keywords,
        )

        with self._lock:
            self._observations[obs_id] = obs
            self._observation_order.append(obs_id)
            if len(self._observations) > self._MAX_OBSERVATIONS:
                evict_id = self._observation_order.pop(0)
                self._observations.pop(evict_id, None)

        return obs_id

    def find_cross_domain_patterns(self) -> List[CrossDomainPattern]:
        with self._lock:
            obs_snapshot = dict(self._observations)

        keyword_to_obs: Dict[str, List[str]] = defaultdict(list)
        keyword_to_domains: Dict[str, set] = defaultdict(set)

        for obs_id, obs in obs_snapshot.items():
            for kw in obs.keywords:
                keyword_to_obs[kw].append(obs_id)
                if obs.domain:
                    keyword_to_domains[kw].add(obs.domain)

        candidate_keywords: Dict[str, List[str]] = defaultdict(list)
        for kw, obs_ids in keyword_to_obs.items():
            domains = keyword_to_domains[kw]
            if len(obs_ids) >= 3 and len(domains) >= 2:
                for oid in obs_ids:
                    candidate_keywords[oid].append(kw)

        keyword_groups: Dict[tuple, set] = defaultdict(set)
        for obs_id, kws in candidate_keywords.items():
            if not kws:
                continue
            key = tuple(sorted(kws))
            keyword_groups[key].add(obs_id)

        patterns: List[CrossDomainPattern] = []

        new_patterns: List[CrossDomainPattern] = []
        new_insights: List[Insight] = []

        for key, obs_ids in keyword_groups.items():
            if len(obs_ids) < 3:
                continue
            obs_list = [obs_snapshot[oid] for oid in obs_ids if oid in obs_snapshot]
            domains = set(o.domain for o in obs_list if o.domain)
            if len(domains) < 2:
                continue

            pattern_id = uuid.uuid4().hex[:12]
            avg_importance = sum(o.importance for o in obs_list) / len(obs_list)
            domain_combo = len(domains) / len(LIFE_DOMAINS)
            confidence = min(1.0, (0.5 * avg_importance) + (0.3 * min(len(obs_list) / 10, 1.0)) + (0.2 * domain_combo))

            description = (
                f"Pattern across {', '.join(sorted(domains))}: "
                f"{', '.join(key[:5])} "
                f"(found in {len(obs_list)} observations)"
            )

            pattern = CrossDomainPattern(
                pattern_id=pattern_id,
                keywords=list(key),
                domains_involved=sorted(domains),
                observations_linked=sorted(obs_ids),
                confidence=round(confidence, 3),
                description=description,
            )
            patterns.append(pattern)
            new_patterns.append(pattern)

            insight = Insight(
                domain=sorted(domains)[0],
                description=description,
                confidence=round(confidence, 3),
                source_observations=sorted(obs_ids)[:10],
                created_at=datetime.now(timezone.utc).isoformat(),
                domains=list(sorted(domains)),
            )
            new_insights.append(insight)

        with self._lock:
            for pattern in new_patterns:
                self._patterns[pattern.pattern_id] = pattern
                self._pattern_order.append(pattern.pattern_id)
            while len(self._patterns) > self._MAX_PATTERNS:
                evict_id = self._pattern_order.pop(0)
                self._patterns.pop(evict_id, None)

            for insight in new_insights:
                iid = uuid.uuid4().hex[:12]
                self._insights[iid] = insight
                self._insight_order.append(iid)
            while len(self._insights) > self._MAX_INSIGHTS:
                evict_id = self._insight_order.pop(0)
                self._insights.pop(evict_id, None)

        return patterns

    def synthesize(self, query: str, domain: str = None) -> SynthesisResult:
        query = _sanitize(query, self._MAX_QUERY_LENGTH)
        query_keywords = set(_extract_keywords(query))

        with self._lock:
            obs_snapshot = dict(self._observations)

        scored: List[tuple] = []
        for obs_id, obs in obs_snapshot.items():
            obs_kw_set = set(obs.keywords)
            overlap = len(query_keywords & obs_kw_set) if query_keywords else 0

            domain_prox = 0.0
            if domain and obs.domain and domain != obs.domain:
                domain_prox = DOMAIN_PROXIMITY.get((domain, obs.domain), 0.0)
            elif domain and not obs.domain:
                domain_prox = 0.1

            try:
                obs_time = datetime.fromisoformat(obs.timestamp)
                now = datetime.now(timezone.utc)
                age_hours = max(0, (now - obs_time).total_seconds() / 3600)
                recency = max(0, 1.0 - age_hours / (30 * 24))
            except (ValueError, TypeError):
                recency = 0.5

            is_domain_match = (not domain) or (obs.domain == domain)
            overlap_weight = 2.0 if is_domain_match else 0.5

            score = (
                overlap * overlap_weight
                + domain_prox * 0.5
                + recency * 0.3
                + obs.importance * 0.5
            )
            if score > 0:
                scored.append((obs_id, obs, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        direct_observations = []
        cross_domain_insights = []
        connections = []

        for obs_id, obs, score in scored[:20]:
            entry = {
                "observation_id": obs.observation_id,
                "text": obs.text,
                "domain": obs.domain,
                "score": round(score, 3),
                "keywords": list(obs.keywords),
            }
            if domain and obs.domain and obs.domain != domain:
                cross_domain_insights.append(entry)
            else:
                direct_observations.append(entry)

            if obs.keywords:
                connections.append({
                    "from_keywords": list(set(query_keywords) & set(obs.keywords)),
                    "to_keywords": list(obs.keywords),
                    "observation_id": obs.observation_id,
                })

        avg_conf = 0.0
        if scored:
            avg_conf = min(1.0, sum(s for _, _, s in scored) / len(scored) / 5.0)

        return SynthesisResult(
            query=query,
            direct_observations=direct_observations,
            cross_domain_insights=cross_domain_insights,
            connections=connections,
            confidence=round(avg_conf, 3),
        )

    def get_insights(self, domain: str = None, min_confidence: float = 0.3) -> List[Insight]:
        with self._lock:
            insights_snapshot = dict(self._insights)

        results = []
        for ins in insights_snapshot.values():
            if domain:
                if ins.domain != domain and domain not in ins.domains:
                    continue
            if ins.confidence < min_confidence:
                continue
            results.append(copy.deepcopy(ins))
        return results

    def connect_concepts(self, concept_a: str, concept_b: str) -> ConnectionResult:
        concept_a = _sanitize(concept_a, self._MAX_CONCEPT_LENGTH)
        concept_b = _sanitize(concept_b, self._MAX_CONCEPT_LENGTH)
        kw_a = set(_extract_keywords(concept_a))
        kw_b = set(_extract_keywords(concept_b))

        with self._lock:
            obs_snapshot = dict(self._observations)

        chain = []
        obs_a_matches = []
        obs_b_matches = []

        for obs in obs_snapshot.values():
            obs_kws = set(obs.keywords)
            if kw_a & obs_kws:
                obs_a_matches.append(obs)
            if kw_b & obs_kws:
                obs_b_matches.append(obs)

        found = False
        best_conf = 0.0
        best_chain = []

        for obs_a in obs_a_matches:
            for obs_b in obs_b_matches:
                shared = set(obs_a.keywords) & set(obs_b.keywords) & (kw_a | kw_b)
                a_only = set(obs_a.keywords) - kw_a - kw_b
                b_only = set(obs_b.keywords) - kw_a - kw_b
                bridge = a_only & b_only

                if shared or bridge:
                    overlap = len(shared) + len(bridge)
                    total = len(kw_a | kw_b) or 1
                    conf = min(1.0, overlap / total + 0.3)
                    if conf > best_conf:
                        best_conf = conf
                        best_chain = [obs_a.to_dict(), obs_b.to_dict()]
                        found = True

        if found:
            chain = best_chain

        if not found and obs_a_matches and obs_b_matches:
            bridge_obs = []
            kw_b_remaining = set(kw_b)
            for obs_a in obs_a_matches[:5]:
                obs_a_kws = set(obs_a.keywords)
                for candidate in obs_snapshot.values():
                    if candidate.observation_id == obs_a.observation_id:
                        continue
                    cand_kws = set(candidate.keywords)
                    if obs_a_kws & cand_kws and cand_kws & kw_b_remaining:
                        bridge_obs.append(candidate)
                        kw_b_remaining -= (cand_kws & kw_b_remaining)
                        break

            if bridge_obs:
                chain = [obs_a_matches[0].to_dict(), bridge_obs[0].to_dict(), obs_b_matches[0].to_dict()]
                best_conf = 0.2
            else:
                best_conf = 0.0

        return ConnectionResult(
            concept_a=concept_a,
            concept_b=concept_b,
            chain=chain,
            confidence=round(best_conf, 3),
            hops=len(chain) - 1 if chain else 0,
        )

    def get_domain_connections(self, domain: str) -> List[DomainConnection]:
        with self._lock:
            obs_snapshot = dict(self._observations)

        domain_keywords: Dict[str, set] = defaultdict(set)
        domain_obs_count: Dict[str, int] = defaultdict(int)

        for obs in obs_snapshot.values():
            if obs.domain:
                domain_keywords[obs.domain].update(obs.keywords)
                domain_obs_count[obs.domain] += 1

        if domain not in domain_keywords:
            return []

        target_kws = domain_keywords[domain]
        connections = []

        for other_domain, other_kws in domain_keywords.items():
            if other_domain == domain:
                continue
            shared = sorted(target_kws & other_kws)
            if not shared:
                continue
            proximity = DOMAIN_PROXIMITY.get((domain, other_domain), 0.1)
            keyword_overlap = len(shared) / max(len(target_kws), 1)
            shared_obs_count = sum(
                1 for obs in obs_snapshot.values()
                if obs.domain == other_domain and set(obs.keywords) & target_kws
            )
            obs_scale = min(domain_obs_count[domain] + shared_obs_count, 50) / 50
            strength = min(1.0, proximity * 0.4 + keyword_overlap * 0.4 + obs_scale * 0.2)
            connections.append(DomainConnection(
                domain_a=domain,
                domain_b=other_domain,
                strength=round(strength, 3),
                shared_keywords=shared[:20],
                connection_count=shared_obs_count,
            ))

        connections.sort(key=lambda c: c.strength, reverse=True)
        return connections

    def get_stats(self) -> Dict:
        with self._lock:
            obs_snapshot = dict(self._observations)
            pattern_snapshot = dict(self._patterns)
            insight_snapshot = dict(self._insights)

        domain_dist = defaultdict(int)
        importance_sum = 0.0
        for obs in obs_snapshot.values():
            if obs.domain:
                domain_dist[obs.domain] += 1
            importance_sum += obs.importance

        avg_importance = importance_sum / len(obs_snapshot) if obs_snapshot else 0.0

        return {
            "observation_count": len(obs_snapshot),
            "pattern_count": len(pattern_snapshot),
            "insight_count": len(insight_snapshot),
            "domain_distribution": dict(domain_dist),
            "avg_importance": round(avg_importance, 3),
            "max_observations": self._MAX_OBSERVATIONS,
            "max_patterns": self._MAX_PATTERNS,
            "max_insights": self._MAX_INSIGHTS,
        }