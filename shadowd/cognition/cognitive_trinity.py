#!/usr/bin/env python3
"""
CognitiveTrinity — Three-Mind Decision Ensemble

Makes decisions by running three distinct reasoning modes
in parallel and aggregating their votes:
1. Intuition — fast pattern matching, heuristics, emotional resonance
2. Logic — analytical, evidence-based, risk-calculating
3. Creative — lateral, generative, alternative-seeking

The final decision weighs each mind's output based on confidence
and domain appropriateness.
"""

import copy
import re
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from collections import deque

logger = logging.getLogger("shadowd.cognition.cognitive_trinity")


class MindType(str, Enum):
    INTUITION = "intuition"
    LOGIC = "logic"
    CREATIVITY = "creativity"


class ConsensusType(str, Enum):
    CONSENSUS = "consensus"
    SPLIT = "split"
    DIVERGENCE = "divergence"


@dataclass
class MindResult:
    mind_type: MindType
    recommendation: str
    confidence: float
    reasoning: str
    key_factors: List[str]
    risk_level: str

    def to_dict(self) -> Dict:
        return {
            "mind_type": self.mind_type.value,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "key_factors": list(self.key_factors),
            "risk_level": self.risk_level,
        }


@dataclass
class SynthesisResult:
    final_recommendation: str
    confidence: float
    consensus: ConsensusType
    mind_results: List[MindResult]
    dominant_mind: MindType
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            "final_recommendation": self.final_recommendation,
            "confidence": round(self.confidence, 3),
            "consensus": self.consensus.value,
            "mind_results": [mr.to_dict() for mr in self.mind_results],
            "dominant_mind": self.dominant_mind.value,
            "reasoning": self.reasoning,
        }


@dataclass
class TrinityAnalysis:
    analysis_id: str
    situation: str
    intuition: MindResult
    logic: MindResult
    creative: MindResult
    synthesis: SynthesisResult
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "analysis_id": self.analysis_id,
            "situation": self.situation[:200],
            "intuition": self.intuition.to_dict(),
            "logic": self.logic.to_dict(),
            "creative": self.creative.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "timestamp": self.timestamp,
        }


_INTUITION_PATTERNS = [
    (r"\b(deadline|urgent|asap|rush|hurry)\b", "urgency", "Act quickly — time pressure detected", 0.8, "medium"),
    (r"\b(trust|believe|faith|gut|instinct)\b", "trust", "Go with established trust patterns", 0.7, "low"),
    (r"\b(risk|danger|harm|threat|warning)\b", "caution", "Proceed with caution — risk signals detected", 0.75, "high"),
    (r"\b(opportunity|chance|potential|possibility)\b", "opportunity", "Opportunity detected — consider proactive approach", 0.7, "medium"),
    (r"\b(relationship|team|together|collaborat)\b", "social", "Prioritize relationship and collaboration aspects", 0.65, "low"),
    (r"\b(money|cost|budget|expense|price)\b", "financial", "Financial factors are primary — trust the numbers", 0.7, "medium"),
    (r"\b(health|sick|doctor|medical|pain)\b", "health", "Health considerations take priority", 0.85, "high"),
    (r"\b(learn|study|grow|develop|improve)\b", "growth", "Growth opportunity — lean toward action", 0.6, "low"),
]

_LOGIC_PATTERNS = [
    (r"\b(if|then|because|therefore|thus|hence)\b", "causal", "Causal reasoning applies — follow the logic chain", 0.75, "medium"),
    (r"\b(data|evidence|fact|proven|statistics)\b", "evidence", "Evidence-based decision — trust the data", 0.8, "low"),
    (r"\b(risk|uncertain|maybe|might|could)\b", "uncertainty", "Uncertainty detected — gather more information first", 0.65, "high"),
    (r"\b(compare|versus|alternative|option|choice)\b", "comparison", "Multiple options identified — evaluate trade-offs", 0.7, "medium"),
    (r"\b(plan|strategy|goal|objective|target)\b", "planning", "Strategic context — align with long-term goals", 0.75, "low"),
    (r"\b(rule|policy|regulation|law|compliance)\b", "compliance", "Compliance context — follow established rules", 0.8, "medium"),
    (r"\b(impact|consequence|result|outcome)\b", "impact", "Impact analysis needed — trace consequences", 0.7, "medium"),
]

_CREATIVE_PATTERNS = [
    (r"\b(stuck|blocked|impossible|can't|no way)\b", "inversion", "Invert the problem — what's the opposite approach?", 0.65, "medium"),
    (r"\b(different|creative|innovat|novel|uniqu)\b", "divergence", "Divergent thinking needed — explore unusual angles", 0.7, "low"),
    (r"\b(simpl|easy|straightforward|obvious)\b", "simplification", "Challenge assumptions — is there a simpler way?", 0.6, "low"),
    (r"\b(complex|complicat|overwhelm|difficult)\b", "decomposition", "Break it down — what are the smallest solvable parts?", 0.7, "medium"),
    (r"\b(connection|relat|similar|like|analogy)\b", "analogy", "Find analogies — what other domain has solved similar problems?", 0.65, "low"),
]


class CognitiveTrinity:
    _MAX_SITUATION_LENGTH = 5000
    _MAX_HISTORY = 2000
    _MAX_DOMAINS_PER_MIND = 100

    def __init__(self):
        self._lock = threading.Lock()
        self._analysis_history: deque = deque(maxlen=self._MAX_HISTORY)
        self._mind_accuracy: Dict[str, Dict[str, float]] = {
            "intuition": {},
            "logic": {},
            "creativity": {},
        }
        self._total_analyses = 0

    def _sanitize(self, text: str) -> str:
        text = re.sub(r"[<>\x00-\x1f\x7f]", "", text)
        return text.strip()[: self._MAX_SITUATION_LENGTH]

    @staticmethod
    def _sanitize_content(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()[: CognitiveTrinity._MAX_SITUATION_LENGTH]

    def intuition_mind(self, situation: str, context: Dict = None) -> MindResult:
        situation = self._sanitize_content(situation)
        context = context or {}
        matched_patterns = []
        key_factors = []
        total_confidence = 0.0
        max_risk = "low"

        for pattern, name, recommendation, conf, risk in _INTUITION_PATTERNS:
            if re.search(pattern, situation, re.IGNORECASE):
                matched_patterns.append(name)
                key_factors.append(name)
                total_confidence += conf
                if risk == "high":
                    max_risk = "high"
                elif risk == "medium" and max_risk != "high":
                    max_risk = "medium"

        positive_words = len(re.findall(r"\b(good|great|excellent|love|happy|excited|wonderful)\b", situation, re.IGNORECASE))
        negative_words = len(re.findall(r"\b(bad|terrible|hate|angry|sad|worst|awful)\b", situation, re.IGNORECASE))

        emotional_tone = "neutral"
        if positive_words > negative_words + 1:
            emotional_tone = "positive"
            total_confidence += 0.1
        elif negative_words > positive_words + 1:
            emotional_tone = "negative"
            total_confidence -= 0.1

        if not matched_patterns:
            recommendation = "No strong intuitive signal — proceed with general awareness"
            confidence = max(0.3, 0.5 + total_confidence)
        else:
            dominant = matched_patterns[0]
            rec_map = {p: r for _, p, r, _, _ in _INTUITION_PATTERNS}
            recommendation = rec_map.get(dominant, "Trust your instincts")
            confidence = min(0.95, max(0.3, total_confidence / max(len(matched_patterns), 1)))

        reasoning = f"Emotional tone: {emotional_tone}. "
        if matched_patterns:
            reasoning += f"Patterns detected: {', '.join(matched_patterns)}. "
        reasoning += f"Risk assessment: {max_risk}."

        return MindResult(
            mind_type=MindType.INTUITION,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors[:5],
            risk_level=max_risk,
        )

    def logic_mind(self, situation: str, context: Dict = None) -> MindResult:
        situation = self._sanitize_content(situation)
        context = context or {}
        matched_patterns = []
        key_factors = []
        total_confidence = 0.0
        max_risk = "low"

        for pattern, name, recommendation, conf, risk in _LOGIC_PATTERNS:
            if re.search(pattern, situation, re.IGNORECASE):
                matched_patterns.append(name)
                key_factors.append(name)
                total_confidence += conf
                if risk == "high":
                    max_risk = "high"
                elif risk == "medium" and max_risk != "high":
                    max_risk = "medium"

        evidence_words = len(re.findall(r"\b(data|fact|evidence|proven|shown|demonstrated)\b", situation, re.IGNORECASE))
        uncertainty_words = len(re.findall(r"\b(maybe|might|could|perhaps|possibly|uncertain)\b", situation, re.IGNORECASE))

        if evidence_words > 0:
            total_confidence += 0.15
            key_factors.append("evidence_available")
        if uncertainty_words > 1:
            total_confidence -= 0.15
            max_risk = "high" if max_risk == "high" else "medium"
            key_factors.append("high_uncertainty")

        if not matched_patterns:
            recommendation = "Insufficient logical structure — gather more information"
            confidence = max(0.3, 0.4 + total_confidence)
        else:
            dominant = matched_patterns[0]
            rec_map = {p: r for _, p, r, _, _ in _LOGIC_PATTERNS}
            recommendation = rec_map.get(dominant, "Apply analytical reasoning")
            confidence = min(0.95, max(0.3, total_confidence / max(len(matched_patterns), 1)))

        reasoning = f"Logical patterns: {', '.join(matched_patterns) if matched_patterns else 'none'}. "
        reasoning += f"Evidence markers: {evidence_words}. Uncertainty markers: {uncertainty_words}. "
        reasoning += f"Risk level: {max_risk}."

        if confidence < 0.3:
            recommendation = "Gather more information before deciding"
            confidence = 0.3
        elif confidence < 0.7:
            recommendation = f"Proceed with caution — {recommendation.lower()}"

        return MindResult(
            mind_type=MindType.LOGIC,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors[:5],
            risk_level=max_risk,
        )

    def creative_mind(self, situation: str, context: Dict = None) -> MindResult:
        situation = self._sanitize_content(situation)
        context = context or {}
        matched_patterns = []
        key_factors = []
        total_confidence = 0.0
        max_risk = "low"

        for pattern, name, recommendation, conf, risk in _CREATIVE_PATTERNS:
            if re.search(pattern, situation, re.IGNORECASE):
                matched_patterns.append(name)
                key_factors.append(name)
                total_confidence += conf
                if risk == "high":
                    max_risk = "high"
                elif risk == "medium" and max_risk != "high":
                    max_risk = "medium"

        alternatives = []

        if re.search(r"\b(not|can't|cannot|don't|won't|isn't|aren't)\b", situation, re.IGNORECASE):
            alternatives.append("Consider reframing the constraint as an opportunity")
        if re.search(r"\b(change|new|different)\b", situation, re.IGNORECASE):
            alternatives.append("Explore how this change could create unexpected benefits")
        if re.search(r"\b(problem|issue|challenge)\b", situation, re.IGNORECASE):
            alternatives.append("What would happen if we did the exact opposite?")

        if alternatives:
            key_factors.extend(alternatives[:2])

        if not matched_patterns:
            recommendation = "Look for unconventional approaches — what hasn't been tried?"
            confidence = max(0.3, 0.4 + total_confidence)
        else:
            dominant = matched_patterns[0]
            rec_map = {p: r for _, p, r, _, _ in _CREATIVE_PATTERNS}
            recommendation = rec_map.get(dominant, "Think outside the box")
            confidence = min(0.9, max(0.3, total_confidence / max(len(matched_patterns), 1)))

        reasoning = f"Creative patterns: {', '.join(matched_patterns) if matched_patterns else 'none'}. "
        if alternatives:
            reasoning += f"Alternative approaches: {'; '.join(alternatives)}. "
        reasoning += f"Risk tolerance: {max_risk}."

        return MindResult(
            mind_type=MindType.CREATIVITY,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors[:5],
            risk_level=max_risk,
        )

    def synthesize(
        self,
        intuition: MindResult,
        logic: MindResult,
        creative: MindResult,
    ) -> SynthesisResult:
        results = [intuition, logic, creative]
        recs = [r.recommendation for r in results]

        if len(set(recs)) == 1 or (
            intuition.recommendation == logic.recommendation == creative.recommendation
        ):
            consensus = ConsensusType.CONSENSUS
        elif (
            intuition.recommendation == logic.recommendation
            or intuition.recommendation == creative.recommendation
            or logic.recommendation == creative.recommendation
        ):
            consensus = ConsensusType.SPLIT
        else:
            consensus = ConsensusType.DIVERGENCE

        risk_weights = {"low": 1.0, "medium": 0.7, "high": 0.4}
        weighted = []
        for r in results:
            weight = r.confidence * risk_weights.get(r.risk_level, 0.7)
            weighted.append((r, weight))

        best = max(weighted, key=lambda x: x[1])
        dominant = best[0].mind_type

        if consensus == ConsensusType.CONSENSUS:
            final_conf = min(0.95, max(r.confidence for r in results) * 0.8 + 0.2)
            reasoning = "All three minds agree — high confidence in this recommendation."
        elif consensus == ConsensusType.SPLIT:
            final_conf = min(0.85, best[1] * 0.9 + 0.1)
            reasoning = f"Two of three minds align (dominant: {dominant.value}) — moderate confidence."
        else:
            final_conf = max(0.3, best[1] * 0.6)
            reasoning = f"Minds diverge — proceeding with {dominant.value} perspective. Consider gathering more info."

        final_rec = best[0].recommendation

        return SynthesisResult(
            final_recommendation=final_rec,
            confidence=round(final_conf, 3),
            consensus=consensus,
            mind_results=results,
            dominant_mind=dominant,
            reasoning=reasoning,
        )

    def analyze(self, situation: str, context: Dict = None) -> TrinityAnalysis:
        situation = self._sanitize_content(situation)
        context = context or {}

        intuition = self.intuition_mind(situation, context)
        logic = self.logic_mind(situation, context)
        creative = self.creative_mind(situation, context)
        synthesis = self.synthesize(intuition, logic, creative)

        analysis = TrinityAnalysis(
            analysis_id=uuid.uuid4().hex[:12],
            situation=situation,
            intuition=intuition,
            logic=logic,
            creative=creative,
            synthesis=synthesis,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            if len(self._analysis_history) >= self._MAX_HISTORY:
                evicted = self._analysis_history[0]
                logger.warning("Evicted analysis %s from history (cap=%d)", evicted.analysis_id, self._MAX_HISTORY)
            self._analysis_history.append(analysis)
            self._total_analyses += 1

        return copy.deepcopy(analysis)

    def get_dominant_mind(self, domain: str = "general") -> str:
        with self._lock:
            accuracies = self._mind_accuracy
            best_mind = "intuition"
            best_score = 0.0
            for mind, domains in accuracies.items():
                score = domains.get(domain, domains.get("general", 0.5))
                if score > best_score:
                    best_score = score
                    best_mind = mind
            return best_mind

    def record_outcome(self, analysis_id: str, actual_outcome: str, satisfaction: float) -> None:
        analysis_id = str(analysis_id).strip()[:100]
        satisfaction = max(0.0, min(1.0, satisfaction))
        found = False
        with self._lock:
            for analysis in self._analysis_history:
                if analysis.analysis_id == analysis_id:
                    found = True
                    dominant = analysis.synthesis.dominant_mind.value
                    adjustment = satisfaction * 0.1 - 0.05
                    for mind in ["intuition", "logic", "creativity"]:
                        current = self._mind_accuracy[mind].get("general", 0.5)
                        self._mind_accuracy[mind]["general"] = round(
                            max(0.1, min(1.0, current + (adjustment if mind == dominant else adjustment * 0.3))), 3
                        )
                        if len(self._mind_accuracy[mind]) > self._MAX_DOMAINS_PER_MIND:
                            keys_to_remove = [k for k in self._mind_accuracy[mind] if k not in ("general",)]
                            for k in keys_to_remove[:len(self._mind_accuracy[mind]) - self._MAX_DOMAINS_PER_MIND]:
                                del self._mind_accuracy[mind][k]
                    break
        if not found:
            logger.warning("record_outcome: analysis_id %s not found", analysis_id)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_analyses": self._total_analyses,
                "history_size": len(self._analysis_history),
                "mind_accuracy": {k: dict(v) for k, v in self._mind_accuracy.items()},
            }