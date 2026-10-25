#!/usr/bin/env python3
"""
Learning System - Decision tracking and outcome learning

Implements:
- Decision tracking (Shadow Council ability)
- Outcome feedback
- Pattern learning from decisions
- Recommendation improvement
"""

import asyncio
import copy
import hashlib
import os
import re as _re
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone

def _ls_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import logging
import math

logger = logging.getLogger("shadowd.learning_system")

_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 10000) -> str:
    if not text:
        return text
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not text:
        return text
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


@dataclass
class Decision:
    """A tracked decision."""
    id: str
    timestamp: datetime
    context: str              # Situation description
    options: List[str]        # Available choices
    chosen: str               # What was chosen
    reasoning: str            # Why this choice
    confidence: float = 0.0
    outcome: Optional[str] = None
    satisfaction: int = 5     # 1-10 scale
    outcome_timestamp: Optional[datetime] = None
    lessons: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Feedback:
    """Feedback on an ability's response."""
    ability: str
    query: str
    response_preview: str
    rating: int               # 1-5
    feedback_type: str       # helpful, accurate, specific, actionable
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


@dataclass
class LearnedPattern:
    """A learned pattern from decisions."""
    pattern_id: str
    pattern_type: str         # decision, preference, outcome
    description: str
    examples: List[str]
    confidence: float
    first_seen: datetime
    last_used: datetime
    success_rate: float = 0.0


class LearningSystem:
    """
    Learning system for Shadow AI.
    
    Tracks:
    - Decisions made (for Shadow Council ability)
    - Feedback on ability responses
    - Patterns from successful decisions
    - Domain-specific expertise
    
    Learns:
    - What decisions work well
    - User preferences
    - Optimal approaches for situations
    """
    
    MAX_DECISIONS = 10000
    MAX_FEEDBACK = 10000
    MAX_PATTERNS = 500
    MAX_DOMAINS = 100
    MAX_CONTEXT_KEYS = 1000
    MAX_EXAMPLES_PER_PATTERN = 50
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Decision tracking (Shadow Council)
        self.decisions: List[Decision] = []
        
        # Feedback tracking
        self.feedback: List[Feedback] = []
        
        # Learned patterns
        self.patterns: Dict[str, LearnedPattern] = {}
        
        # Domain expertise
        self.domain_expertise: Dict[str, Dict] = defaultdict(dict)
        
        # Context->Success tracking
        self.context_success: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
        
        # Load persisted data
        self._load_state()
    
    def _load_state(self):
        """Load persisted learning data."""
        state_file = self.data_dir / "learning_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                
                # Load decisions
                self.decisions = [
                    Decision(
                        id=d["id"],
                        timestamp=_ls_ts(d, "timestamp"),
                        context=d["context"],
                        options=d["options"],
                        chosen=d["chosen"],
                        reasoning=d["reasoning"],
                        confidence=d.get("confidence", 0.0),
                        outcome=d.get("outcome"),
                        satisfaction=d.get("satisfaction", 5),
                        outcome_timestamp=_ls_ts(d, "outcome_timestamp"),
                        lessons=d.get("lessons", []),
                        tags=d.get("tags", [])
                    )
                    for d in data.get("decisions", [])
                ]
                
                # Load patterns
                self.patterns = {
                    p_id: LearnedPattern(
                        pattern_id=p_id,
                        pattern_type=p["pattern_type"],
                        description=p["description"],
                        examples=p["examples"],
                        confidence=p["confidence"],
                        first_seen=_ls_ts(p, "first_seen"),
                        last_used=_ls_ts(p, "last_used"),
                        success_rate=p.get("success_rate", 0.0)
                    )
                    for p_id, p in data.get("patterns", {}).items()
                }
                
                # Load context success
                self.context_success = defaultdict(lambda: {"success": 0, "failure": 0})
                for ctx, stats in data.get("context_success", {}).items():
                    self.context_success[ctx] = stats
                
                # Load feedback
                self.feedback = [
                    Feedback(
                        ability=f["ability"],
                        query=f["query"],
                        response_preview=f["response_preview"],
                        rating=f["rating"],
                        feedback_type=f["feedback_type"],
                        timestamp=_ls_ts(f, "timestamp"),
                        metadata=f.get("metadata", {})
                    )
                    for f in data.get("feedback", [])
                ]
                
                logger.info(f"Loaded {len(self.decisions)} decisions, {len(self.patterns)} patterns")
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
                logger.error(f"Failed to load learning state: {type(e).__name__}")
    
    def _save_state(self):
        """Persist learning data."""
        with self._lock:
            self._save_state_unlocked()
    
    def _save_state_unlocked(self):
        """Persist learning data (caller must hold self._lock)."""
        state_file = self.data_dir / "learning_state.json"
        try:
            data = {
                "decisions": [
                    {
                        "id": d.id,
                        "timestamp": d.timestamp.isoformat(),
                        "context": d.context,
                        "options": d.options,
                        "chosen": d.chosen,
                        "reasoning": d.reasoning,
                        "confidence": d.confidence,
                        "outcome": d.outcome,
                        "satisfaction": d.satisfaction,
                        "outcome_timestamp": d.outcome_timestamp.isoformat() if d.outcome_timestamp else None,
                        "lessons": d.lessons,
                        "tags": d.tags
                    }
                    for d in self.decisions
                ],
                "patterns": {
                    p_id: {
                        "pattern_type": p.pattern_type,
                        "description": p.description,
                        "examples": p.examples,
                        "confidence": p.confidence,
                        "first_seen": p.first_seen.isoformat(),
                        "last_used": p.last_used.isoformat(),
                        "success_rate": p.success_rate
                    }
                    for p_id, p in self.patterns.items()
                },
                "context_success": dict(self.context_success),
                "feedback": [
                    {
                        "ability": f.ability,
                        "query": f.query,
                        "response_preview": f.response_preview,
                        "rating": f.rating,
                        "feedback_type": f.feedback_type,
                        "timestamp": f.timestamp.isoformat(),
                        "metadata": f.metadata
                    }
                    for f in self.feedback
                ],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            state_file = self.data_dir / "learning_state.json"
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
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save learning state: {type(e).__name__}")
    
    def track_decision(
        self,
        context: str,
        options: List[str],
        chosen: str,
        reasoning: str,
        confidence: float = 0.0,
        tags: List[str] = None
    ) -> Decision:
        """
        Track a decision for future learning (Shadow Council ability).
        
        Args:
            context: The situation/decision context
            options: Available choices
            chosen: What was chosen
            reasoning: Why this choice
            confidence: Confidence in the decision (0-1)
            tags: Optional tags for categorization
        
        Returns:
            Decision object
        """
        context = _sanitize_content(context)
        options = [_sanitize_content(o) for o in options]
        chosen = _sanitize_content(chosen)
        reasoning = _sanitize_content(reasoning)
        tags = [_sanitize(t) for t in (tags or [])]
        with self._lock:
            decision = Decision(
                id=f"dec_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(timezone.utc),
                context=context,
                options=options,
                chosen=chosen,
                reasoning=reasoning,
                confidence=confidence,
                tags=tags
            )
            decision.confidence = max(0.0, min(1.0, decision.confidence))
            
            self.decisions.append(decision)
            if len(self.decisions) > self.MAX_DECISIONS:
                self.decisions = self.decisions[-self.MAX_DECISIONS:]
            
            if len(self.domain_expertise) > self.MAX_DOMAINS:
                oldest = list(self.domain_expertise.keys())[:len(self.domain_expertise) - self.MAX_DOMAINS]
                for k in oldest:
                    del self.domain_expertise[k]
            
            if len(self.context_success) > self.MAX_CONTEXT_KEYS:
                oldest = list(self.context_success.keys())[:len(self.context_success) - self.MAX_CONTEXT_KEYS]
                for k in oldest:
                    del self.context_success[k]
            
            # Extract context pattern
            self._extract_decision_pattern(decision)
            
            # Save periodically
            if len(self.decisions) % 10 == 0:
                self._save_state_unlocked()
            
            logger.info("Tracked decision: %s", chosen[:50])
            return decision
    
    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        satisfaction: int
    ) -> bool:
        """
        Record outcome of a decision.
        
        Args:
            decision_id: ID of tracked decision
            outcome: What happened
            satisfaction: User satisfaction (1-10)
        
        Returns:
            True if decision found and updated
        """
        with self._lock:
            for decision in self.decisions:
                if decision.id == decision_id:
                    decision.outcome = outcome
                    decision.outcome_timestamp = datetime.now(timezone.utc)
                    decision.satisfaction = max(1, min(10, satisfaction))
                    
                    lessons = self._extract_lessons(decision)
                    decision.lessons = lessons
                    
                    context_key = self._get_context_key(decision.context)
                    if satisfaction >= 6:
                        self.context_success[context_key]["success"] += 1
                    else:
                        self.context_success[context_key]["failure"] += 1
                    
                    self._learn_from_outcome(decision)
                    
                    self._save_state_unlocked()
                    logger.info(f"Recorded outcome for {decision_id}: satisfaction={satisfaction}")
                    return True
        
        return False
    
    def record_feedback(
        self,
        ability: str,
        query: str,
        response_preview: str,
        rating: int,
        feedback_type: str = "helpful"
    ):
        _VALID_FEEDBACK_TYPES = {"helpful", "accurate", "specific", "actionable"}
        ability = _sanitize(ability)
        query = _sanitize_content(query)
        response_preview = _sanitize_content(response_preview)
        rating = max(1, min(5, rating))
        if feedback_type not in _VALID_FEEDBACK_TYPES:
            feedback_type = "helpful"
        feedback = Feedback(
            ability=ability,
            query=query,
            response_preview=response_preview,
            rating=rating,
            feedback_type=feedback_type
        )
        
        with self._lock:
            self.feedback.append(feedback)
            if len(self.feedback) > self.MAX_FEEDBACK:
                self.feedback = self.feedback[-self.MAX_FEEDBACK:]
            
            # Learn from feedback
            self._learn_from_feedback(feedback)
            
            if len(self.feedback) % 10 == 0:
                self._save_state_unlocked()
    
    def query_similar_decisions(self, context: str, limit: int = 5) -> List[Decision]:
        """
        Find similar past decisions.
        
        Args:
            context: Current situation
            limit: Max results
        
        Returns:
            Similar decisions with outcomes
        """
        with self._lock:
            limit = max(1, min(limit, 100))
            context_keywords = set(context.lower().split())
            
            scored_decisions = []
            
            for decision in self.decisions:
                if not decision.outcome:
                    continue
                
                # Calculate similarity
                decision_keywords = set(decision.context.lower().split())
                similarity = len(context_keywords & decision_keywords) / max(len(context_keywords), 1)
                
                # Factor in success
                success_factor = decision.satisfaction / 10.0
                
                # Combined score
                score = (similarity * 0.6) + (success_factor * 0.4)
                
                scored_decisions.append((score, decision))
            
            # Sort by score
            scored_decisions.sort(key=lambda x: x[0], reverse=True)
            
            return [copy.deepcopy(decision) for score, decision in scored_decisions[:limit]]
    
    def get_domain_expertise(self, domain: str) -> Dict:
        """
        Get accumulated expertise for a domain.
        
        Returns patterns and learnings relevant to domain.
        """
        with self._lock:
            if domain not in self.domain_expertise:
                domain_decisions = [
                    d for d in self.decisions
                    if domain.lower() in d.context.lower() or domain.lower() in ' '.join(d.tags).lower()
                ]
                
                expertise = {
                    "total_decisions": len(domain_decisions),
                    "avg_satisfaction": 0.0,
                    "common_choices": {},
                    "successful_patterns": [],
                    "lessons_learned": []
                }
                
                if domain_decisions:
                    expertise["avg_satisfaction"] = sum(d.satisfaction for d in domain_decisions) / len(domain_decisions)
                    
                    choice_counts = defaultdict(int)
                    for d in domain_decisions:
                        choice_counts[d.chosen] += 1
                    expertise["common_choices"] = dict(choice_counts)
                    
                    successful = [d for d in domain_decisions if d.satisfaction >= 7]
                    for d in successful[:5]:
                        expertise["successful_patterns"].append({
                            "context": d.context[:100],
                            "choice": d.chosen,
                            "reasoning": d.reasoning[:100]
                        })
                    
                    for d in domain_decisions:
                        expertise["lessons_learned"].extend(d.lessons[:2])
                    
                    expertise["lessons_learned"] = list(set(expertise["lessons_learned"]))[:10]
                
                self.domain_expertise[domain] = expertise
            
            return copy.deepcopy(self.domain_expertise[domain])
    
    def get_recommendation(self, context: str, options: List[str] = None) -> Dict:
        similar = self.query_similar_decisions(context, limit=10)
        
        if not similar:
            return {
                "recommendation": None,
                "confidence": 0.0,
                "reasoning": "No similar past decisions found"
            }
        
        option_success = defaultdict(lambda: {"count": 0, "success": 0})
        
        for decision in similar:
            if decision.chosen:
                option_success[decision.chosen]["count"] += 1
                if decision.satisfaction >= 6:
                    option_success[decision.chosen]["success"] += 1
        
        best_option = None
        best_score = 0
        
        for option, stats in option_success.items():
            if stats["count"] > 0:
                score = stats["success"] / stats["count"]
                if score > best_score:
                    best_score = score
                    best_option = option
        
        if options:
            valid_options = [o for o in options if o in option_success]
            if valid_options:
                best_in_options = max(
                    valid_options,
                    key=lambda o: option_success[o]["success"] / max(option_success[o]["count"], 1)
                )
                best_option = best_in_options
        
        reasoning_parts = []
        for d in similar[:3]:
            if d.satisfaction >= 7:
                reasoning_parts.append(f"Previously chose '{d.chosen}' and was satisfied ({d.satisfaction}/10)")
        
        return {
            "recommendation": best_option,
            "confidence": best_score,
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Based on past outcomes",
            "similar_cases": len(similar),
            "options_considered": list(option_success.keys())
        }
    
    def _extract_decision_pattern(self, decision: Decision):
        """Extract pattern from a decision."""
        pattern_id = f"decision_{hashlib.sha256(decision.context.encode()).hexdigest()[:16]}"
        
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                pattern_type="decision",
                description=f"When {decision.context[:50]}, chose {decision.chosen}",
                examples=[f"{decision.context[:50]} -> {decision.chosen}"],
                confidence=0.0,
                first_seen=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc)
            )
        else:
            self.patterns[pattern_id].examples.append(
                f"{decision.context[:50]} -> {decision.chosen}"
            )
            if len(self.patterns[pattern_id].examples) > self.MAX_EXAMPLES_PER_PATTERN:
                self.patterns[pattern_id].examples = self.patterns[pattern_id].examples[-self.MAX_EXAMPLES_PER_PATTERN:]
            self.patterns[pattern_id].last_used = datetime.now(timezone.utc)
        
        if len(self.patterns) > self.MAX_PATTERNS:
            oldest_key = min(self.patterns, key=lambda k: self.patterns[k].last_used)
            del self.patterns[oldest_key]
    
    def _extract_lessons(self, decision: Decision) -> List[str]:
        """Extract lessons from a decision outcome."""
        lessons = []
        
        if decision.satisfaction >= 8:
            lessons.append(f"Choosing '{decision.chosen}' worked well in similar situations")
        elif decision.satisfaction <= 4:
            lessons.append(f"Consider alternatives to '{decision.chosen}' in similar situations")
        
        # Extract key differentiators using word-boundary regex
        if _re.search(r'\bbecause\b', decision.reasoning.lower()):
            reason = decision.reasoning.split("because")[1][:100]
            if decision.satisfaction >= 7:
                lessons.append(f"Good reasoning: {reason}")
            else:
                lessons.append(f"This reasoning may not apply: {reason}")
        
        return lessons
    
    def _get_context_key(self, context: str) -> str:
        """Get a simplified context key for tracking."""
        # Use first 3 significant words
        words = [w for w in context.lower().split() if len(w) > 3][:3]
        return "_".join(words)
    
    def _learn_from_outcome(self, decision: Decision):
        """Learn from a decision outcome."""
        pattern_id = f"decision_{hashlib.sha256(decision.context.encode()).hexdigest()[:16]}"
        
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Use keywords for matching instead of raw snippet
            context_words = decision.context.lower().split()[:5]
            scored = []
            for d in self.decisions:
                if d.outcome:
                    d_words = set(d.context.lower().split())
                    overlap = len(set(context_words) & d_words) / max(len(set(context_words)), 1)
                    scored.append((overlap, d))
            
            relevant_decisions = [
                d for score, d in sorted(scored, key=lambda x: x[0], reverse=True)[:20]
                if score > 0.1
            ]
            
            if relevant_decisions:
                successes = sum(1 for d in relevant_decisions if d.satisfaction >= 6)
                pattern.success_rate = successes / len(relevant_decisions)
    
    def _learn_from_feedback(self, feedback: Feedback):
        """Learn from ability feedback."""
        # Track ability performance
        ability_key = f"ability_{feedback.ability}"
        
        # Build patterns from feedback
        positive = feedback.rating >= 4
        negative = feedback.rating <= 2
        
        if positive:
            if ability_key not in self.patterns:
                self.patterns[ability_key] = LearnedPattern(
                    pattern_id=ability_key,
                    pattern_type="ability_preference",
                    description=f"{feedback.ability} performs well on queries like '{feedback.query[:30]}'",
                    examples=[feedback.query[:50]],
                    confidence=0.5,
                    first_seen=datetime.now(timezone.utc),
                    last_used=datetime.now(timezone.utc)
                )
            else:
                self.patterns[ability_key].examples.append(feedback.query[:50])
                if len(self.patterns[ability_key].examples) > self.MAX_EXAMPLES_PER_PATTERN:
                    self.patterns[ability_key].examples = self.patterns[ability_key].examples[-self.MAX_EXAMPLES_PER_PATTERN:]
                self.patterns[ability_key].confidence = min(1.0, self.patterns[ability_key].confidence + 0.1)
        
        elif negative:
            # Reduce confidence
            if ability_key in self.patterns:
                self.patterns[ability_key].confidence = max(0.0, self.patterns[ability_key].confidence - 0.2)
    
    def get_status(self) -> Dict:
        """Get learning system status."""
        with self._lock:
            decisions_count = len(self.decisions)
            decisions_with_outcomes = len([d for d in self.decisions if d.outcome])
            avg_satisfaction = sum(d.satisfaction for d in self.decisions) / max(decisions_count, 1)
            feedback_count = len(self.feedback)
            avg_rating = sum(f.rating for f in self.feedback) / max(feedback_count, 1)
            patterns_count = len(self.patterns)
            high_confidence = len([p for p in self.patterns.values() if p.confidence > 0.7])
            domains_count = len(self.domain_expertise)
        return {
            "decisions": {
                "total": decisions_count,
                "with_outcomes": decisions_with_outcomes,
                "avg_satisfaction": round(avg_satisfaction, 3),
            },
            "feedback": {
                "total": feedback_count,
                "avg_rating": round(avg_rating, 3),
            },
            "patterns": {
                "total": patterns_count,
                "high_confidence": high_confidence,
            },
            "domains": domains_count,
        }