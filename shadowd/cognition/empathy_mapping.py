#!/usr/bin/env python3
"""
EmpathyMapping - Social Context and Relationship Dynamics Tracker

Models emotional dynamics and interaction patterns between people:
- Person relationship profiles (communication style, emotional patterns)
- Interaction history with sentiment tracking
- Empathy scoring (how well we understand each person)
- Communication recommendations (how to talk to each person)
"""

import copy
import logging
import re
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger("shadowd.cognition.empathy_mapping")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


class RelationshipType(str, Enum):
    FAMILY = "family"
    FRIEND = "friend"
    COLLEAGUE = "colleague"
    MANAGER = "manager"
    REPORT = "report"
    ACQUAINTANCE = "acquaintance"
    PROFESSIONAL = "professional"


class InteractionType(str, Enum):
    CALL = "call"
    TEXT = "text"
    EMAIL = "email"
    MEETING = "meeting"
    IN_PERSON = "in_person"
    SOCIAL_MEDIA = "social_media"


@dataclass
class PersonProfile:
    person_id: str
    name: str
    relationship: RelationshipType
    created_at: str
    last_interaction: Optional[str]
    interaction_count: int
    avg_sentiment: float
    preferred_channels: List[str]
    emotional_triggers: List[str]
    formality_preference: str
    response_patterns: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["relationship"] = self.relationship.value
        d["response_patterns"] = copy.deepcopy(self.response_patterns)
        d["metadata"] = copy.deepcopy(self.metadata)
        d["avg_sentiment"] = round(self.avg_sentiment, 3)
        return d


@dataclass
class CommunicationStyle:
    person_id: str
    preferred_channel: str
    formality_level: str
    response_time_hours: float
    emotional_triggers: List[str]
    dos: List[str]
    donts: List[str]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["emotional_triggers"] = list(self.emotional_triggers)
        d["dos"] = list(self.dos)
        d["donts"] = list(self.donts)
        d["response_time_hours"] = round(self.response_time_hours, 2)
        return d


@dataclass
class ReactionPrediction:
    person_id: str
    topic: str
    approach: str
    predicted_sentiment: float
    predicted_receptiveness: float
    confidence: float
    alternative_approaches: List[str]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["alternative_approaches"] = list(self.alternative_approaches)
        d["predicted_sentiment"] = round(self.predicted_sentiment, 3)
        d["predicted_receptiveness"] = round(self.predicted_receptiveness, 3)
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class RelationshipHealth:
    person_id: str
    health_score: float
    trend: str
    last_interaction: Optional[str]
    recommendation: str
    interaction_frequency: str

    def to_dict(self) -> Dict:
        d = copy.deepcopy(asdict(self))
        d["health_score"] = round(self.health_score, 2)
        return d


@dataclass
class _Interaction:
    interaction_id: str
    person_id: str
    interaction_type: InteractionType
    sentiment: float
    notes: str
    metadata: Dict[str, Any]
    timestamp: str


_FORMALITY_MAP = {
    RelationshipType.MANAGER: "formal",
    RelationshipType.REPORT: "mixed",
    RelationshipType.COLLEAGUE: "mixed",
    RelationshipType.PROFESSIONAL: "formal",
    RelationshipType.FRIEND: "casual",
    RelationshipType.FAMILY: "casual",
    RelationshipType.ACQUAINTANCE: "mixed",
}


class EmpathyMapping:

    _MAX_PERSONS = 500
    _MAX_INTERACTIONS_PER_PERSON = 500
    _MAX_TOPIC_LENGTH = 500
    _MAX_APPROACH_LENGTH = 200
    _MAX_METADATA_KEYS = 50

    def __init__(self):
        self._lock = threading.Lock()
        self._persons: Dict[str, PersonProfile] = {}
        self._interactions: deque = deque(maxlen=10000)
        self._person_interactions: Dict[str, List[_Interaction]] = {}
        logger.info("EmpathyMapping initialized")

    def add_person(self, name: str, relationship: str = "acquaintance", metadata: Dict = None) -> str:
        name = re.sub(r'[<>\x00-\x1f\x7f]', '', name).strip()[:200]
        if not name:
            raise ValueError("name must not be empty")
        try:
            rel = RelationshipType(relationship)
        except ValueError:
            raise ValueError(f"invalid relationship type: {relationship}")

        with self._lock:
            if len(self._persons) >= self._MAX_PERSONS:
                oldest_id = min(self._persons, key=lambda pid: self._persons[pid].created_at)
                logger.warning("Evicting person %s to stay within max_persons=%d", oldest_id, self._MAX_PERSONS)
                self._remove_person(oldest_id)

            if metadata and len(metadata) > self._MAX_METADATA_KEYS:
                metadata = dict(list(metadata.items())[:self._MAX_METADATA_KEYS])
            if metadata:
                metadata = {str(k): str(v)[:500] for k, v in metadata.items()}

            person_id = uuid.uuid4().hex[:12]
            profile = PersonProfile(
                person_id=person_id,
                name=name,
                relationship=rel,
                created_at=datetime.now(timezone.utc).isoformat(),
                last_interaction=None,
                interaction_count=0,
                avg_sentiment=0.0,
                preferred_channels=[],
                emotional_triggers=[],
                formality_preference=_FORMALITY_MAP.get(rel, "mixed"),
                response_patterns={},
                metadata=metadata or {},
            )
            self._persons[person_id] = profile
            self._person_interactions[person_id] = []
            logger.info("Added person %s (%s, %s)", name, person_id, rel.value)
            return person_id

    def record_interaction(
        self,
        person_id: str,
        interaction_type: str,
        sentiment: float,
        notes: str = "",
        metadata: Dict = None,
    ) -> str:
        try:
            itype = InteractionType(interaction_type)
        except ValueError:
            raise ValueError(f"invalid interaction type: {interaction_type}")

        sentiment = max(-1.0, min(1.0, sentiment))
        notes = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', notes).strip()[:2000]
        if metadata and len(metadata) > self._MAX_METADATA_KEYS:
            metadata = dict(list(metadata.items())[:self._MAX_METADATA_KEYS])
        if metadata:
            metadata = {str(k): str(v)[:500] for k, v in metadata.items()}

        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")

            interaction_id = uuid.uuid4().hex[:12]
            now = datetime.now(timezone.utc).isoformat()
            interaction = _Interaction(
                interaction_id=interaction_id,
                person_id=person_id,
                interaction_type=itype,
                sentiment=sentiment,
                notes=notes,
                metadata=metadata or {},
                timestamp=now,
            )

            self._interactions.append(interaction)
            self._person_interactions[person_id].append(interaction)
            if len(self._person_interactions[person_id]) > self._MAX_INTERACTIONS_PER_PERSON:
                self._person_interactions[person_id] = self._person_interactions[person_id][-self._MAX_INTERACTIONS_PER_PERSON:]

            profile = self._persons[person_id]
            profile.last_interaction = now
            profile.interaction_count += 1

            sentiments = [i.sentiment for i in self._person_interactions[person_id]]
            profile.avg_sentiment = sum(sentiments) / len(sentiments)

            channel_counts = Counter(
                i.interaction_type.value for i in self._person_interactions[person_id]
            )
            profile.preferred_channels = [
                ch for ch, _ in channel_counts.most_common(5)
            ]

            profile.emotional_triggers = self._compute_emotional_triggers(person_id)

            if profile.interaction_count >= 2:
                if profile.avg_sentiment > 0.3:
                    profile.formality_preference = "casual"
                else:
                    profile.formality_preference = "mixed"

            profile.response_patterns = self._compute_response_patterns(person_id)
            logger.info("Recorded interaction %s for %s", interaction_id, person_id)
            return interaction_id

    def get_person_profile(self, person_id: str) -> PersonProfile:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")
            return copy.deepcopy(self._persons[person_id])

    def get_communication_style(self, person_id: str) -> CommunicationStyle:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")
            profile = self._persons[person_id]
            interactions = self._person_interactions.get(person_id, [])

            preferred_channel = profile.preferred_channels[0] if profile.preferred_channels else "text"

            formality_level = profile.formality_preference

            response_time_hours = profile.response_patterns.get("avg_response_hours", 24.0)

            emotional_triggers = list(profile.emotional_triggers)

            dos = self._generate_dos(profile, interactions)
            donts = self._generate_donts(profile, interactions)

            return CommunicationStyle(
                person_id=person_id,
                preferred_channel=preferred_channel,
                formality_level=formality_level,
                response_time_hours=response_time_hours,
                emotional_triggers=emotional_triggers,
                dos=dos,
                donts=donts,
            )

    def get_empathy_score(self, person_id: str) -> float:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")
            profile = self._persons[person_id]
            interactions = self._person_interactions.get(person_id, [])

            if profile.interaction_count == 0:
                return 0.0

            interaction_score = min(profile.interaction_count / 20.0, 1.0) * 40

            sentiments = [i.sentiment for i in interactions]
            if len(sentiments) >= 2:
                mean_s = sum(sentiments) / len(sentiments)
                variance = sum((s - mean_s) ** 2 for s in sentiments) / len(sentiments)
                consistency = max(0.0, 1.0 - variance) * 30
            else:
                consistency = 10.0

            channel_diversity = min(len(profile.preferred_channels) / 3.0, 1.0) * 15

            notes_richness = 0.0
            for i in interactions:
                if i.notes:
                    notes_richness += 1.0
            notes_score = min(notes_richness / max(profile.interaction_count, 1), 1.0) * 15

            return min(100.0, interaction_score + consistency + channel_diversity + notes_score)

    def predict_reaction(
        self, person_id: str, topic: str, approach: str
    ) -> ReactionPrediction:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")
            profile = self._persons[person_id]
            interactions = self._person_interactions.get(person_id, [])

            topic = _SANITIZE_RE.sub('', topic).strip()[:self._MAX_TOPIC_LENGTH]
            approach = _SANITIZE_RE.sub('', approach).strip()[:self._MAX_APPROACH_LENGTH]
            predicted_sentiment = profile.avg_sentiment

            topic_lower = topic.lower()
            for trigger in profile.emotional_triggers:
                if re.search(r'\b' + re.escape(trigger.lower()) + r'\b', topic_lower):
                    predicted_sentiment -= 0.2
                    break

            approach_lower = approach.lower()
            if approach_lower in ("direct", "gentle", "supportive", "cautious"):
                if profile.formality_preference == "casual" and approach_lower in ("gentle", "supportive"):
                    predicted_sentiment += 0.1
                elif profile.formality_preference == "formal" and approach_lower in ("direct", "cautious"):
                    predicted_sentiment += 0.1
                elif profile.formality_preference == "mixed":
                    predicted_sentiment += 0.05
            else:
                approach = "direct"

            predicted_sentiment = max(-1.0, min(1.0, predicted_sentiment))

            receptiveness = 0.5
            if profile.interaction_count > 0:
                receptiveness = max(0.0, min(1.0, 0.5 + profile.avg_sentiment * 0.3))

            confidence = min(1.0, profile.interaction_count / 20.0)
            if len(interactions) >= 5:
                confidence = min(1.0, confidence + 0.2)

            alternatives = self._suggest_alternative_approaches(profile, topic)

            return ReactionPrediction(
                person_id=person_id,
                topic=topic,
                approach=approach,
                predicted_sentiment=predicted_sentiment,
                predicted_receptiveness=receptiveness,
                confidence=confidence,
                alternative_approaches=alternatives,
            )

    def get_relationship_health(self, person_id: str) -> RelationshipHealth:
        with self._lock:
            if person_id not in self._persons:
                raise KeyError(f"person {person_id} not found")
            profile = self._persons[person_id]
            interactions = self._person_interactions.get(person_id, [])

            sentiment_score = max(0, (profile.avg_sentiment + 1.0) / 2.0) * 40

            frequency_score = 0.0
            if profile.interaction_count > 0:
                frequency_score = min(profile.interaction_count / 10.0, 1.0) * 30

            recency_score = 0.0
            if profile.last_interaction:
                try:
                    last_dt = datetime.fromisoformat(profile.last_interaction)
                    hours_ago = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                    recency_score = max(0, 30 - hours_ago * 0.5)
                except (ValueError, TypeError):
                    pass

            health_score = min(100.0, sentiment_score + frequency_score + recency_score)

            trend = "stable"
            if len(interactions) >= 3:
                recent = interactions[-3:]
                recent_avg = sum(i.sentiment for i in recent) / len(recent)
                if len(interactions) > 5:
                    older = interactions[-6:-3]
                    older_avg = sum(i.sentiment for i in older) / len(older)
                    if recent_avg > older_avg + 0.1:
                        trend = "improving"
                    elif recent_avg < older_avg - 0.1:
                        trend = "declining"
                else:
                    if recent_avg > 0.2:
                        trend = "improving"
                    elif recent_avg < -0.2:
                        trend = "declining"

            freq = "rare"
            if profile.interaction_count >= 20:
                freq = "daily"
            elif profile.interaction_count >= 8:
                freq = "weekly"
            elif profile.interaction_count >= 3:
                freq = "monthly"

            recommendation = self._make_recommendation(profile, health_score, trend)

            return RelationshipHealth(
                person_id=person_id,
                health_score=health_score,
                trend=trend,
                last_interaction=profile.last_interaction,
                recommendation=recommendation,
                interaction_frequency=freq,
            )

    def get_social_graph(self) -> Dict:
        with self._lock:
            nodes = []
            edges = []
            person_list = list(self._persons.values())

            for profile in person_list:
                empathy = self._compute_empathy_score_unlocked(profile.person_id)
                nodes.append({
                    "id": profile.person_id,
                    "name": profile.name,
                    "relationship": profile.relationship.value,
                    "empathy_score": empathy,
                    "interaction_count": profile.interaction_count,
                    "avg_sentiment": profile.avg_sentiment,
                })

            rel_affinity = {
                RelationshipType.FAMILY: 0.9,
                RelationshipType.FRIEND: 0.8,
                RelationshipType.COLLEAGUE: 0.5,
                RelationshipType.MANAGER: 0.4,
                RelationshipType.REPORT: 0.4,
                RelationshipType.PROFESSIONAL: 0.3,
                RelationshipType.ACQUAINTANCE: 0.2,
            }

            for i, p1 in enumerate(person_list):
                for p2 in person_list[i + 1:]:
                    affinity = rel_affinity.get(p1.relationship, 0.2) * rel_affinity.get(p2.relationship, 0.2)
                    if affinity > 0.1:
                        edges.append({
                            "source": p1.person_id,
                            "target": p2.person_id,
                            "affinity": round(affinity, 2),
                        })

            return {
                "nodes": nodes,
                "edges": edges,
                "total_persons": len(person_list),
            }

    def get_stats(self) -> Dict:
        with self._lock:
            total_interactions = len(self._interactions)
            sentiments = [i.sentiment for i in self._interactions]
            avg = sum(sentiments) / len(sentiments) if sentiments else 0.0

            channel_counts = Counter(i.interaction_type.value for i in self._interactions)
            relationship_counts = Counter(
                p.relationship.value for p in self._persons.values()
            )

            empathy_scores = []
            for pid in self._persons:
                empathy_scores.append(self._compute_empathy_score_unlocked(pid))
            avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0

            return {
                "total_persons": len(self._persons),
                "total_interactions": total_interactions,
                "avg_sentiment": round(avg, 3),
                "channel_distribution": dict(channel_counts),
                "relationship_distribution": dict(relationship_counts),
                "avg_empathy_score": round(avg_empathy, 1),
                "max_persons": self._MAX_PERSONS,
                "max_interactions": 10000,
            }

    def _remove_person(self, person_id: str):
        if person_id in self._persons:
            del self._persons[person_id]
        if person_id in self._person_interactions:
            del self._person_interactions[person_id]

    def _compute_emotional_triggers(self, person_id: str) -> List[str]:
        interactions = self._person_interactions.get(person_id, [])
        if len(interactions) < 2:
            return []

        trigger_words = []
        for interaction in interactions:
            if abs(interaction.sentiment) > 0.5 and interaction.notes:
                words = interaction.notes.lower().split()
                for w in words:
                    w_clean = w.strip(".,!?;:")
                    if len(w_clean) > 3:
                        trigger_words.append(w_clean)

        counter = Counter(trigger_words)
        return [word for word, _ in counter.most_common(10)]

    def _compute_response_patterns(self, person_id: str) -> Dict[str, Any]:
        interactions = self._person_interactions.get(person_id, [])
        if len(interactions) < 2:
            return {}

        gaps = []
        for i in range(1, len(interactions)):
            try:
                t1 = datetime.fromisoformat(interactions[i - 1].timestamp)
                t2 = datetime.fromisoformat(interactions[i].timestamp)
                gap_hours = abs((t2 - t1).total_seconds()) / 3600
                gaps.append(gap_hours)
            except (ValueError, TypeError):
                continue

        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        return {
            "avg_response_hours": round(avg_gap, 2),
            "min_response_hours": round(min(gaps), 2) if gaps else 0.0,
            "max_response_hours": round(max(gaps), 2) if gaps else 0.0,
            "total_interactions": len(interactions),
        }

    def _generate_dos(self, profile: PersonProfile, interactions: List[_Interaction]) -> List[str]:
        dos = []
        if profile.preferred_channels:
            dos.append(f"Use {profile.preferred_channels[0]} as the primary channel")

        if profile.formality_preference == "formal":
            dos.append("Address formally, use professional language")
        elif profile.formality_preference == "casual":
            dos.append("Be warm and casual, use first names")
        else:
            dos.append("Adapt formality based on context")

        if profile.avg_sentiment > 0.2:
            dos.append("Positive relationship — maintain rapport")
        elif profile.avg_sentiment < -0.2:
            dos.append("Relationship is strained — be extra considerate")

        if relationship_triggers := profile.emotional_triggers[:3]:
            topics = ", ".join(relationship_triggers)
            dos.append(f"Be aware of sensitive topics: {topics}")

        return dos[:5]

    def _generate_donts(self, profile: PersonProfile, interactions: List[_Interaction]) -> List[str]:
        donts = []
        neg_interactions = [i for i in interactions if i.sentiment < -0.3]
        neg_channels = Counter(i.interaction_type.value for i in neg_interactions)

        for ch, _ in neg_channels.most_common(2):
            donts.append(f"Avoid having difficult conversations via {ch}")

        if profile.formality_preference == "formal":
            donts.append("Avoid overly casual language")
        elif profile.formality_preference == "casual":
            donts.append("Avoid being overly stiff or formal")

        if profile.avg_sentiment < -0.2:
            donts.append("Avoid pushing boundaries or being confrontational")

        if profile.emotional_triggers:
            donts.append(f"Be cautious with: {', '.join(profile.emotional_triggers[:3])}")

        return donts[:5]

    def _suggest_alternative_approaches(self, profile: PersonProfile, topic: str) -> List[str]:
        alternatives = []

        current = profile.formality_preference
        if current == "formal":
            alternatives.extend(["gentle", "supportive"])
        elif current == "casual":
            alternatives.extend(["direct", "supportive"])
        else:
            alternatives.extend(["gentle", "direct", "supportive"])

        for trigger in profile.emotional_triggers[:2]:
            if re.search(r'\b' + re.escape(trigger.lower()) + r'\b', topic.lower()):
                alternatives.append("cautious")
                break

        return list(dict.fromkeys(alternatives))[:4]

    def _make_recommendation(self, profile: PersonProfile, health_score: float, trend: str) -> str:
        if health_score < 30:
            return f"Relationship with {profile.name} needs attention — schedule a meaningful interaction soon"
        if trend == "declining":
            return f"Relationship with {profile.name} is declining — consider reaching out with a positive gesture"
        if trend == "improving":
            return f"Relationship with {profile.name} is improving — keep up current approach"
        if health_score < 60:
            return f"Invest more time in interactions with {profile.name} to strengthen the bond"
        return f"Relationship with {profile.name} is healthy — maintain regular contact"

    def _compute_empathy_score_unlocked(self, person_id: str) -> float:
        profile = self._persons[person_id]
        interactions = self._person_interactions.get(person_id, [])

        if profile.interaction_count == 0:
            return 0.0

        interaction_score = min(profile.interaction_count / 20.0, 1.0) * 40

        sentiments = [i.sentiment for i in interactions]
        if len(sentiments) >= 2:
            mean_s = sum(sentiments) / len(sentiments)
            variance = sum((s - mean_s) ** 2 for s in sentiments) / len(sentiments)
            consistency = max(0.0, 1.0 - variance) * 30
        else:
            consistency = 10.0

        channel_diversity = min(len(profile.preferred_channels) / 3.0, 1.0) * 15

        notes_richness = sum(1.0 for i in interactions if i.notes)
        notes_score = min(notes_richness / max(profile.interaction_count, 1), 1.0) * 15

        return round(min(100.0, interaction_score + consistency + channel_diversity + notes_score), 1)