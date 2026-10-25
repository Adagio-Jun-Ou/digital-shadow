#!/usr/bin/env python3
"""
Enhanced Decision Learning System - Based on Research

Implements patterns from:
- AgentGym-RL: Progressive horizon, outcome tracking
- pyDecision: MCDA methods (AHP, TOPSIS, ELECTRE, PROMETHEE)
- graphthulhu: Decision lifecycle, status tracking
- commit-story: Automatic journal generation
"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone

def _dl_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import re as _re
import uuid
import logging
import math
import os
import threading

logger = logging.getLogger("shadowd.decision_learning")

_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text


# Decision Lifecycle (from graphthulhu pattern)

class DecisionStatus:
    """Decision status lifecycle."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"
    ABANDONED = "abandoned"
    FAILED = "failed"


@dataclass
class DecisionV2:
    """
    Enhanced decision tracking (multiple research sources).
    
    From AgentGym-RL:
    - Progressive horizon (immediate, short-term, long-term)
    - Multi-step outcomes
    
    From graphthulhu:
    - Lifecycle status
    - Priority
    - Dependencies
    """
    # Core fields
    id: str
    context: str
    situation: str
    options: List[str]
    chosen: str
    reasoning: str
    confidence: float = 0.0
    
    # Status (graphthulhu pattern)
    status: str = DecisionStatus.PENDING
    priority: str = "medium"  # low, medium, high, critical
    dependencies: List[str] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Outcome tracking (AgentGym-RL)
    outcome: Optional[str] = None
    satisfaction: int = 5  # 1-10 scale
    outcome_steps: List[Dict] = field(default_factory=list)  # Multi-step outcomes
    horizon: str = "immediate"  # immediate, short_term, long_term
    
    # Pattern matching
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    similar_decisions: List[str] = field(default_factory=list)
    
    # Learning
    lessons_learned: List[str] = field(default_factory=list)
    pattern_detected: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "context": self.context,
            "situation": self.situation,
            "options": list(self.options),
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "created_at": self.created_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "outcome": self.outcome,
            "satisfaction": self.satisfaction,
            "outcome_steps": copy.deepcopy(self.outcome_steps),
            "horizon": self.horizon,
            "tags": list(self.tags),
            "category": self.category,
            "similar_decisions": list(self.similar_decisions),
            "lessons_learned": list(self.lessons_learned),
            "pattern_detected": self.pattern_detected
        }


# MCDA Methods (from pyDecision)

class MCDAMethods:
    """
    Multi-Criteria Decision Analysis methods.
    
    From pyDecision research:
    - AHP (Analytic Hierarchy Process)
    - TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    - ELECTRE (Elimination and Choice Translating Reality)
    - PROMETHEE (Preference Ranking Organization METHod for Enrichment Evaluations)
    """
    
    @staticmethod
    def ahp(criteria: List[str], alternatives: List[str], 
            criteria_weights: Dict[str, float],
            alternative_scores: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """
        Analytic Hierarchy Process.
        
        Returns ranked alternatives with scores.
        """
        # Normalize weights
        total_weight = sum(criteria_weights.values())
        normalized_weights = {k: v/total_weight for k, v in criteria_weights.items()}
        
        # Calculate scores
        scores = {}
        for alt in alternatives:
            score = sum(
                alternative_scores.get(alt, {}).get(criterion, 0) * normalized_weights[criterion]
                for criterion in criteria
            )
            scores[alt] = score
        
        # Rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    @staticmethod
    def topsis(criteria: List[str], alternatives: List[str],
               values: Dict[str, List[float]],
               weights: List[float]) -> List[Tuple[str, float]]:
        """
        TOPSIS method.
        
        Returns alternatives ranked by distance to ideal solution.
        """
        n_alternatives = len(alternatives)
        n_criteria = len(criteria)
        
        # Normalize matrix
        normalized = []
        for j in range(n_criteria):
            col = [values[alt][j] for alt in alternatives]
            norm = math.sqrt(sum(x**2 for x in col))
            normalized.append([x/norm if norm > 0 else 0 for x in col])
        
        # Weighted matrix
        weighted = [
            [normalized[j][i] * weights[j] for j in range(n_criteria)]
            for i in range(n_alternatives)
        ]
        
        # Ideal and anti-ideal solutions
        ideal = [max(weighted[i][j] for i in range(n_alternatives)) for j in range(n_criteria)]
        anti_ideal = [min(weighted[i][j] for i in range(n_alternatives)) for j in range(n_criteria)]
        
        # Calculate distances
        scores = []
        for i in range(n_alternatives):
            dist_ideal = math.sqrt(sum((weighted[i][j] - ideal[j])**2 for j in range(n_criteria)))
            dist_anti = math.sqrt(sum((weighted[i][j] - anti_ideal[j])**2 for j in range(n_criteria)))
            
            # Relative closeness
            score = dist_anti / (dist_ideal + dist_anti) if (dist_ideal + dist_anti) > 0 else 0
            scores.append((alternatives[i], score))
        
        # Rank by score
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return ranked
    
    @staticmethod
    def electre_outranking(alternatives: List[str], criteria: List[str],
                          values: Dict[str, List[float]],
                          weights: List[float],
                          concordance_threshold: float = 0.7) -> Dict[str, Set[str]]:
        """
        ELECTRE outranking method.
        
        Returns which alternatives outrank others.
        """
        n = len(alternatives)
        
        # Concordance matrix
        concordance = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Calculate concordance
                concordance_sum = 0.0
                for k, criterion in enumerate(criteria):
                    if values[alternatives[i]][k] >= values[alternatives[j]][k]:
                        concordance_sum += weights[k]
                
                concordance[i][j] = concordance_sum
        
        # Determine outranking
        outranks = defaultdict(set)
        for i in range(n):
            for j in range(n):
                if i != j and concordance[i][j] >= concordance_threshold:
                    outranks[alternatives[i]].add(alternatives[j])
        
        return dict(outranks)


@dataclass
class DecisionPattern:
    """A learned decision pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    examples: List[Dict] = field(default_factory=list)
    success_rate: float = 0.0
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    occurrences: int = 0

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "examples": copy.deepcopy(self.examples),
            "success_rate": round(self.success_rate, 3),
            "confidence": round(self.confidence, 3),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrences": self.occurrences
        }


class DecisionLearningV2:
    """
    Enhanced decision learning system.
    
    Features:
    - Decision lifecycle (pending → resolved/failed)
    - MCDA methods (AHP, TOPSIS, ELECTRE)
    - Pattern learning and matching
    - Outcome correlation
    - Journal generation
    """
    
    _MAX_DECISIONS = 10000
    _MAX_PATTERNS = 500
    _MAX_HISTORY = 10000

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "decisions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Decision tracking
        self.decisions: Dict[str, DecisionV2] = {}
        self.decision_history: List[str] = []
        
        # Pattern learning
        self.patterns: Dict[str, DecisionPattern] = {}
        
        # Category tracking
        self.categories: Dict[str, List[str]] = defaultdict(list)
        
        # MCDA
        self.mcda = MCDAMethods()
        
        # Load persisted data
        self._load()
    
    def create_decision(
        self,
        situation: str,
        options: List[str],
        context: str = "",
        priority: str = "medium",
        deadline: Optional[datetime] = None,
        tags: List[str] = None
    ) -> DecisionV2:
        situation = _sanitize_content(situation)
        context = _sanitize_content(context)
        options = [_sanitize_content(o) for o in options]
        priority = _sanitize(priority) if priority in ("low", "medium", "high", "critical") else "medium"
        tags = [_sanitize(t) for t in (tags or [])]
        with self._lock:
            decision_id = f"dec_{uuid.uuid4().hex[:12]}"
            category = self._detect_category(situation, tags or [])
            similar = self._find_similar(situation, limit=5)
        
            decision = DecisionV2(
                id=decision_id,
                context=context,
                situation=situation,
                options=options,
                chosen="",
                reasoning="",
                status=DecisionStatus.PENDING,
                priority=priority,
                deadline=deadline,
                tags=tags or [],
                category=category,
                similar_decisions=[d.id for d in similar]
            )
        
            self.decisions[decision_id] = decision
            self.decision_history.append(decision_id)
            self.categories[category].append(decision_id)
            
            if len(self.decisions) > self._MAX_DECISIONS:
                oldest = list(self.decisions.keys())[:len(self.decisions) - self._MAX_DECISIONS]
                for old_id in oldest:
                    del self.decisions[old_id]
            if len(self.decision_history) > self._MAX_HISTORY:
                self.decision_history = self.decision_history[-self._MAX_HISTORY:]
            for cat in self.categories:
                if len(self.categories[cat]) > self._MAX_DECISIONS:
                    self.categories[cat] = self.categories[cat][-self._MAX_DECISIONS:]
            
            self._save()
            recommendation = self._mcda_analyze(decision, similar)
            
            return decision, recommendation
    
    def _detect_category(self, situation: str, tags: List[str]) -> str:
        """Detect decision category."""
        text = (situation + " " + " ".join(tags)).lower()
        
        categories = {
            "career": [r"\bjob\b", r"\bwork\b", r"\bcareer\b", r"\bposition\b", r"\bcompany\b", r"\bsalary\b"],
            "health": [r"\bhealth\b", r"\bmedical\b", r"\bdoctor\b", r"\bexercise\b", r"\bdiet\b", r"\btreatment\b"],
            "finance": [r"\bmoney\b", r"\binvest\b", r"\bbudget\b", r"\bexpense\b", r"\bpurchase\b", r"\bfinancial\b"],
            "relationship": [r"\brelationship\b", r"\bpartner\b", r"\bfriend\b", r"\bfamily\b", r"\bmarriage\b"],
            "education": [r"\bcourse\b", r"\blearn\b", r"\bstudy\b", r"\bdegree\b", r"\bschool\b"],
            "lifestyle": [r"\bmove\b", r"\bhouse\b", r"\bcar\b", r"\blifestyle\b", r"\bhobby\b"]
        }
        
        for category, patterns in categories.items():
            if any(_re.search(p, text) for p in patterns):
                return category
        
        return "general"
    
    def _find_similar(self, situation: str, limit: int = 5) -> List[DecisionV2]:
        """Find similar past decisions using keyword matching."""
        situation_words = set(situation.lower().split())
        
        # Score decisions by similarity
        scored = []
        for decision in self.decisions.values():
            if decision.status == DecisionStatus.PENDING:
                continue
            
            decision_words = set(decision.situation.lower().split())
            similarity = len(situation_words & decision_words) / max(len(situation_words), 1)
            
            # Boost by success
            if decision.satisfaction >= 7:
                similarity *= 1.2
            
            scored.append((similarity, decision))
        
        # Sort and return top
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for score, d in scored[:limit]]
    
    def _mcda_analyze(self, decision: DecisionV2, similar: List[DecisionV2]) -> Dict:
        """Use MCDA to analyze decision options."""
        if not similar:
            return {"recommendation": None, "confidence": 0.0}
        
        # Extract what worked in similar situations
        successful_choices = {}
        for sim in similar:
            if sim.satisfaction >= 7:
                choice_type = self._classify_choice(sim.chosen)
                successful_choices[choice_type] = successful_choices.get(choice_type, 0) + 1
        
        # Rank options by similarity to successful choices
        if successful_choices:
            best_type = max(successful_choices.items(), key=lambda x: x[1])[0]
            
            # Find option closest to best type
            for option in decision.options:
                if self._classify_choice(option) == best_type:
                    return {
                        "recommendation": option,
                        "confidence": successful_choices[best_type] / sum(successful_choices.values()),
                        "based_on": [s.id for s in similar[:3]],
                        "method": "pattern_matching"
                    }
        
        return {"recommendation": None, "confidence": 0.0}
    
    def _classify_choice(self, choice: str) -> str:
        """Classify choice type for pattern matching."""
        choice_lower = choice.lower()
        
        if any(_re.search(r'\b' + _re.escape(w) + r'\b', choice_lower) for w in ["yes", "accept", "buy", "start", "begin"]):
            return "positive"
        elif any(_re.search(r'\b' + _re.escape(w) + r'\b', choice_lower) for w in ["no", "reject", "skip", "stop", "end"]):
            return "negative"
        else:
            return "neutral"
    
    def decide(
        self,
        decision_id: str,
        chosen: str,
        reasoning: str,
        confidence: float = 0.8
    ) -> DecisionV2:
        """Record decision choice."""
        chosen = _sanitize_content(chosen)
        reasoning = _sanitize_content(reasoning)
        confidence = max(0.0, min(1.0, confidence))
        with self._lock:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision {decision_id} not found")
            
            decision = self.decisions[decision_id]
            decision.chosen = chosen
            decision.reasoning = reasoning
            decision.confidence = max(0.0, min(1.0, confidence))
            decision.status = DecisionStatus.IN_PROGRESS
            decision.decided_at = datetime.now(timezone.utc)
            
            self._save()
            return decision
    
    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        satisfaction: int,
        steps: List[Dict] = None
    ) -> DecisionV2:
        """Record outcome of a decision."""
        outcome = _sanitize_content(outcome)
        satisfaction = max(1, min(10, satisfaction))
        steps = steps or []
        with self._lock:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision {decision_id} not found")
            
            satisfaction = max(1, min(10, satisfaction))
            
            decision = self.decisions[decision_id]
            decision.outcome = outcome
            decision.satisfaction = satisfaction
            decision.outcome_steps = steps or []
            decision.resolved_at = datetime.now(timezone.utc)
            decision.status = DecisionStatus.RESOLVED if satisfaction >= 5 else DecisionStatus.FAILED
            
            decision.lessons_learned = self._extract_lessons(decision)
            self._detect_pattern(decision)
            
            self._save()
            return decision
    
    def _extract_lessons(self, decision: DecisionV2) -> List[str]:
        """Extract lessons from decision."""
        lessons = []
        
        if decision.satisfaction >= 8:
            lessons.append(f"Choosing '{decision.chosen}' worked well in {decision.category} situations")
            lessons.append(f"Reasoning '{decision.reasoning[:50]}...' was effective")
        elif decision.satisfaction <= 3:
            lessons.append(f"Consider avoiding '{decision.chosen}' in similar {decision.category} situations")
            lessons.append("The reasoning may need adjustment")
        
        return lessons
    
    def _detect_pattern(self, decision: DecisionV2):
        """Detect decision pattern."""
        # Cap total patterns
        if len(self.patterns) >= self._MAX_PATTERNS:
            oldest_key = min(self.patterns, key=lambda k: self.patterns[k].first_seen)
            del self.patterns[oldest_key]
        # Create pattern key from situation keywords
        pattern_key = " ".join(sorted(decision.tags[:3])) if decision.tags else decision.category
        
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = DecisionPattern(
                pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
                pattern_type="contextual",
                description=f"Decisions about {pattern_key}",
                examples=[decision.to_dict()],
                occurrences=1
            )
        else:
            pattern = self.patterns[pattern_key]
            pattern.examples.append(decision.to_dict())
            if len(pattern.examples) > 50:
                pattern.examples = pattern.examples[-50:]
            pattern.occurrences += 1
            pattern.last_seen = datetime.now(timezone.utc)
            
            # Calculate success rate
            successful = sum(1 for ex in pattern.examples if ex.get("satisfaction", 5) >= 7)
            pattern.success_rate = successful / len(pattern.examples)
            pattern.confidence = min(pattern.occurrences / 10.0, 1.0)
    
    def get_recommendation(self, situation: str, options: List[str]) -> Dict:
        """Get recommendation for a new decision."""
        similar = self._find_similar(situation, limit=10)
        
        if not similar:
            return {
                "recommendation": None,
                "confidence": 0.0,
                "method": "no_data"
            }
        
        # Use MCDA analysis
        mcda_result = self._mcda_analyze(
            DecisionV2(
                id="temp",
                context="",
                situation=situation,
                options=options,
                chosen="",
                reasoning=""
            ),
            similar
        )
        
        return {
            **mcda_result,
            "similar_cases": [s.id for s in similar[:3]],
            "pattern": similar[0].category if similar else None
        }
    
    def get_pending(self) -> List[DecisionV2]:
        """Get all pending decisions."""
        with self._lock:
            return [copy.deepcopy(d) for d in self.decisions.values() if d.status == DecisionStatus.PENDING]
    
    def get_by_category(self, category: str) -> List[DecisionV2]:
        """Get decisions by category."""
        with self._lock:
            return [copy.deepcopy(self.decisions[did]) for did in self.categories.get(category, []) if did in self.decisions]
    
    def get_patterns(self, min_confidence: float = 0.7) -> List[DecisionPattern]:
        """Get learned patterns above confidence threshold."""
        with self._lock:
            return [copy.deepcopy(p) for p in self.patterns.values() if p.confidence >= min_confidence]
    
    def generate_journal_entry(self, decision_id: str) -> str:
        """Generate journal entry for a decision (commit-story pattern)."""
        decision = self.decisions.get(decision_id)
        if not decision:
            return "Decision not found"
        
        entry = f"""## Decision: {decision.situation[:80]}

**Date**: {decision.created_at.strftime('%Y-%m-%d %H:%M')}
**Category**: {decision.category}
**Status**: {decision.status}

### Context
{decision.context}

### Options Considered
{chr(10).join(f'- {opt}' for opt in decision.options)}

### Choice Made
**{decision.chosen}**

Reasoning: {decision.reasoning}

### Outcome
{decision.outcome or 'Pending'}

**Satisfaction**: {decision.satisfaction}/10

### Lessons Learned
{chr(10).join(f'- {lesson}' for lesson in decision.lessons_learned)}

---
*Pattern: {decision.category}*
"""
        return entry
    
    def _save(self):
        """Persist decisions to disk atomically."""
        import tempfile
        try:
            data = {
                "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
                "patterns": {k: {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "examples": p.examples[:10],
                    "success_rate": round(p.success_rate, 3),
                    "confidence": round(p.confidence, 3),
                    "occurrences": p.occurrences
                } for k, p in self.patterns.items()},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            data_file = self.data_dir / "decisions.json"
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.data_dir), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(data_file))
            except (OSError, json.JSONDecodeError, ValueError):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to save decisions: %s", type(e).__name__)
    
    def _load(self):
        """Load persisted decisions."""
        try:
            data_file = self.data_dir / "decisions.json"
            if not data_file.exists():
                return
            
            data = json.loads(data_file.read_text())
            
            # Load decisions
            for dec_id, dec_data in data.get("decisions", {}).items():
                decision = DecisionV2(
                    id=dec_data["id"],
                    context=dec_data["context"],
                    situation=dec_data["situation"],
                    options=dec_data["options"],
                    chosen=dec_data["chosen"],
                    reasoning=dec_data["reasoning"],
                    confidence=dec_data["confidence"],
                    status=dec_data["status"],
                    priority=dec_data["priority"],
                    deadline=_dl_ts(dec_data, "deadline"),
                    created_at=_dl_ts(dec_data, "created_at", datetime.now(timezone.utc)),
                    decided_at=_dl_ts(dec_data, "decided_at"),
                    resolved_at=_dl_ts(dec_data, "resolved_at"),
                    outcome=dec_data.get("outcome"),
                    satisfaction=dec_data.get("satisfaction", 5),
                    tags=dec_data.get("tags", []),
                    category=dec_data.get("category", "general")
                )
                self.decisions[dec_id] = decision
            
            # Load patterns
            for pattern_id, pattern_data in data.get("patterns", {}).items():
                pattern = DecisionPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    description=pattern_data["description"],
                    examples=pattern_data["examples"],
                    success_rate=pattern_data["success_rate"],
                    confidence=pattern_data["confidence"],
                    occurrences=pattern_data["occurrences"]
                )
                self.patterns[pattern_id] = pattern
            
            logger.info("Loaded %d decisions, %d patterns", len(self.decisions), len(self.patterns))
        
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Failed to load decisions: %s", type(e).__name__)
    
    def get_stats(self) -> Dict:
        """Get decision learning statistics."""
        with self._lock:
            resolved = [d for d in self.decisions.values() if d.status == DecisionStatus.RESOLVED]
            successful = [d for d in resolved if d.satisfaction >= 7]
            pending_count = len([d for d in self.decisions.values() if d.status == DecisionStatus.PENDING])
            total_count = len(self.decisions)
            patterns_count = len(self.patterns)
            high_conf = len([p for p in self.patterns.values() if p.confidence >= 0.7])
            categories = {cat: len(decs) for cat, decs in self.categories.items()}
        
        return {
            "total_decisions": total_count,
            "pending": pending_count,
            "resolved": len(resolved),
            "successful": len(successful),
            "success_rate": round(len(successful) / len(resolved), 3) if resolved else 0,
            "patterns_learned": patterns_count,
            "high_confidence_patterns": high_conf,
            "categories": categories,
        }