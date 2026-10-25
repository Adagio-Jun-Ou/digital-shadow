#!/usr/bin/env python3
"""
Proactive Insights - LifeBuddy-style proactive intelligence

Generates insights without being asked:
- Commitment reminders
- Pattern-based predictions
- Contextual suggestions
- Anomaly alerts
"""

import asyncio
import copy
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import logging

from .ambient_monitor import Commitment, Pattern, Anomaly, Insight

logger = logging.getLogger("shadowd.proactive_insights")

_SANITIZE_STRICT_RE = __import__('re').compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = __import__('re').compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 10000) -> str:
    if not text:
        return text
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not text:
        return text
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


@dataclass
class ProactiveRule:
    """A rule for generating proactive insights."""
    name: str
    condition: Callable
    message_generator: Callable
    priority: str  # low, medium, high, critical
    cooldown_minutes: int = 60
    last_triggered: datetime = None


class ProactiveInsightEngine:
    _MAX_INSIGHTS = 500
    _MAX_RULE_COOLDOWNS = 100

    """
    Generates proactive insights (LifeBuddy-style).
    
    Monitors:
    - Commitments (deadlines, promises)
    - Patterns (recurring events)
    - Anomalies (unusual activity)
    - Context (time, location, people)
    
    Generates:
    - Time-based reminders
    - Pattern-based predictions
    - Contextual suggestions
    - Cross-domain insights
    """
    
    def __init__(self):
        self.insights_queue: List[Insight] = []
        self.rules: List[ProactiveRule] = []
        self.rule_cooldowns: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register proactive insight rules."""
        
        # Morning review
        self.rules.append(ProactiveRule(
            name="morning_review",
            condition=lambda ctx: ctx.get("hour", 0) == 9 and ctx.get("pending_commitments", 0) > 0,
            message_generator=lambda ctx: f"Morning review: {ctx['pending_commitments']} pending commitments",
            priority="medium",
            cooldown_minutes=1440  # Once per day
        ))
        
        # Approaching deadline
        self.rules.append(ProactiveRule(
            name="deadline_approaching",
            condition=lambda ctx: ctx.get("urgent_deadlines", 0) > 0,
            message_generator=lambda ctx: f"⏰ {ctx['urgent_deadlines']} deadline(s) in next 24 hours",
            priority="high",
            cooldown_minutes=60
        ))
        
        # Pattern reminder
        self.rules.append(ProactiveRule(
            name="pattern_reminder",
            condition=lambda ctx: ctx.get("strong_pattern", None) is not None,
            message_generator=lambda ctx: f"Pattern: {ctx['strong_pattern']}",
            priority="low",
            cooldown_minutes=120
        ))
        
        # Anomaly alert
        self.rules.append(ProactiveRule(
            name="anomaly_detected",
            condition=lambda ctx: ctx.get("anomaly_severity", "low") in ["high", "critical"],
            message_generator=lambda ctx: f"⚠️ Unusual activity: {ctx.get('anomaly_description', 'See details')}",
            priority="high",
            cooldown_minutes=30
        ))
        
        # Evening summary
        self.rules.append(ProactiveRule(
            name="evening_summary",
            condition=lambda ctx: ctx.get("hour", 0) == 21,
            message_generator=lambda ctx: f"Evening summary: {ctx.get('events_today', 0)} events tracked today",
            priority="low",
            cooldown_minutes=1440
        ))
        
        # Weekly review
        self.rules.append(ProactiveRule(
            name="weekly_review",
            condition=lambda ctx: ctx.get("day_of_week") == "Friday" and ctx.get("hour", 0) == 17,
            message_generator=lambda ctx: "Weekly review time: Check goals and commitments",
            priority="medium",
            cooldown_minutes=10080  # Once per week
        ))
    
    def evaluate_rules(self, context: Dict) -> List[Insight]:
        """
        Evaluate all rules against current context.
        
        Args:
            context: Current state (time, commitments, patterns, etc.)
        
        Returns:
            List of triggered insights
        """
        triggered = []
        now = datetime.now(timezone.utc)
        
        with self._lock:
            rules_snapshot = list(self.rules)
            cooldowns = dict(self.rule_cooldowns)
        
        for rule in rules_snapshot:
            # Check cooldown
            if rule.name in cooldowns:
                last = cooldowns[rule.name]
                if (now - last).total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Evaluate condition
            try:
                if rule.condition(context):
                    message = rule.message_generator(context)
                    
                    insight = Insight(
                        insight_type="proactive",
                        priority=rule.priority,
                        title=rule.name.replace("_", " ").title(),
                        message=message,
                        action=self._suggest_action(rule.name, context),
                        confidence=0.8,
                        expires_at=now + timedelta(hours=2)
                    )
                    
                    triggered.append(insight)
            
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error("Error evaluating rule %s: %s", rule.name, type(e).__name__)
        
        # Add to queue
        with self._lock:
            self.insights_queue.extend(triggered)
            
            # Keep queue size manageable
            if len(self.insights_queue) > self._MAX_INSIGHTS:
                self.insights_queue = self.insights_queue[-self._MAX_INSIGHTS:]
            
            # Cap rule_cooldowns
            if len(self.rule_cooldowns) > self._MAX_RULE_COOLDOWNS:
                sorted_cooldowns = sorted(self.rule_cooldowns.items(), key=lambda x: x[1])
                self.rule_cooldowns = dict(sorted_cooldowns[-self._MAX_RULE_COOLDOWNS:])
            
            for rule in rules_snapshot:
                if any(t.insight_type == rule.name for t in triggered):
                    self.rule_cooldowns[rule.name] = now
        
        return triggered
    
    def _suggest_action(self, rule_name: str, context: Dict) -> Optional[str]:
        """Suggest action for a rule."""
        action_map = {
            "morning_review": "Review commitments and plan your day",
            "deadline_approaching": "Focus on urgent tasks",
            "pattern_reminder": "Consider if this pattern helps or hinders",
            "anomaly_detected": "Investigate unusual activity",
            "evening_summary": "Review day and plan tomorrow",
            "weekly_review": "Assess weekly progress and adjust goals"
        }
        
        return action_map.get(rule_name)
    
    def get_active_insights(self) -> List[Insight]:
        """Get non-expired insights."""
        now = datetime.now(timezone.utc)
        with self._lock:
            active = [
                i for i in self.insights_queue
                if i.expires_at is None or i.expires_at >= now
            ]
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active.sort(key=lambda i: priority_order.get(i.priority, 3))
        
        return [copy.deepcopy(i) for i in active]
    
    def check_commitments(self, commitments: List[Commitment]) -> List[Insight]:
        """Check for urgent commitments."""
        insights = []
        now = datetime.now(timezone.utc)
        
        for commitment in commitments:
            if commitment.status != "pending":
                continue
            
            if commitment.deadline:
                deadline = commitment.deadline
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                time_until = deadline - now
                
                if time_until <= timedelta(0):
                    # Already overdue
                    hours_overdue = abs(time_until).total_seconds() / 3600
                    insights.append(Insight(
                        insight_type="commitment",
                        priority="critical",
                        title=f"OVERDUE: {commitment.content[:30]}",
                        message=f"Overdue by {int(hours_overdue * 60)} minutes" if hours_overdue < 1 else f"Overdue by {int(hours_overdue)} hours",
                        action="Complete or reschedule immediately!",
                        confidence=1.0,
                        expires_at=now + timedelta(hours=2)
                    ))
                # Critical: Less than 1 hour
                elif time_until < timedelta(hours=1):
                    insights.append(Insight(
                        insight_type="commitment",
                        priority="critical",
                        title=f"Urgent: {commitment.content[:30]}",
                        message=f"Due in {int(time_until.total_seconds()) // 60} minutes",
                        action="Complete or reschedule immediately!",
                        confidence=1.0,
                        expires_at=commitment.deadline
                    ))
                
                # High: Less than 24 hours
                elif time_until < timedelta(hours=24):
                    insights.append(Insight(
                        insight_type="commitment",
                        priority="high",
                        title=f"Upcoming: {commitment.content[:30]}",
                        message=f"Due {commitment.deadline.strftime('%A at %H:%M')}",
                        action="Prepare and complete",
                        confidence=0.9,
                        expires_at=commitment.deadline - timedelta(hours=1)
                    ))
                
                # Medium: Less than 48 hours
                elif time_until < timedelta(hours=48):
                    insights.append(Insight(
                        insight_type="commitment",
                        priority="medium",
                        title=f"Reminder: {commitment.content[:30]}",
                        message=f"Due in {int(time_until.total_seconds()) // 3600} hours",
                        action="Plan ahead",
                        confidence=0.7,
                        expires_at=now + timedelta(hours=12)
                    ))
        
        return insights
    
    def check_patterns(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate insights from patterns."""
        insights = []
        
        for pattern in patterns:
            # Only high-confidence patterns
            if pattern.confidence < 0.8:
                continue
            
            # Time-based patterns
            if pattern.pattern_type == "frequency":
                insights.append(Insight(
                    insight_type="pattern",
                    priority="low",
                    title="Recurring Pattern",
                    message=pattern.description,
                    action="Be aware of this pattern",
                    confidence=pattern.confidence,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
                ))
            
            # Sequence patterns
            elif pattern.pattern_type == "sequence":
                insights.append(Insight(
                    insight_type="pattern",
                    priority="medium",
                    title="Sequence Pattern",
                    message=pattern.description,
                    action="Use this sequence to your advantage",
                    confidence=pattern.confidence,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=12)
                ))
            
            # Social patterns
            elif pattern.pattern_type == "social":
                insights.append(Insight(
                    insight_type="pattern",
                    priority="medium",
                    title="Social Pattern",
                    message=pattern.description,
                    action="Plan interactions accordingly",
                    confidence=pattern.confidence,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=6)
                ))
        
        return insights
    
    def check_anomalies(self, anomalies: List[Anomaly]) -> List[Insight]:
        """Generate insights from anomalies."""
        insights = []
        
        for anomaly in anomalies:
            severity_priority = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "critical": "critical"
            }
            
            insights.append(Insight(
                insight_type="anomaly",
                priority=severity_priority.get(anomaly.severity, "medium"),
                title="Unusual Activity Detected",
                message=anomaly.description,
                action="Investigate if unexpected",
                confidence=0.7,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=4)
            ))
        
        return insights
    
    def generate_cross_domain_insights(
        self,
        domain_data: Dict[str, Dict]
    ) -> List[Insight]:
        """Find insights across domains."""
        insights = []
        
        # Example: Health + Finance
        health_data = domain_data.get("health", {})
        finance_data = domain_data.get("finance", {})
        
        if health_data.get("stress_level") == "high" and finance_data.get("month_spending", 0) > finance_data.get("budget", float('inf')):
            insights.append(Insight(
                insight_type="cross_domain",
                priority="medium",
                title="Health-Finance Connection",
                message="High stress coinciding with budget overage",
                action="Consider if financial stress affects health",
                confidence=0.65,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=12)
            ))
        
        # Example: Career + Growth
        career_data = domain_data.get("career", {})
        growth_data = domain_data.get("growth", {})
        
        if career_data.get("workload") == "high" and growth_data.get("learning_hours", 0) < 5:
            insights.append(Insight(
                insight_type="cross_domain",
                priority="low",
                title="Work-Life Balance",
                message="High workload with low learning time",
                action="Schedule time for personal development",
                confidence=0.70,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
            ))
        
        return insights
    
    def get_summary(self) -> Dict:
        """Get proactive insights summary."""
        with self._lock:
            total = len(self.insights_queue)
        active = self.get_active_insights()
        
        return {
            "total_insights": total,
            "active_insights": len(active),
            "by_priority": {
                "critical": len([i for i in active if i.priority == "critical"]),
                "high": len([i for i in active if i.priority == "high"]),
                "medium": len([i for i in active if i.priority == "medium"]),
                "low": len([i for i in active if i.priority == "low"])
            },
            "by_type": {
                insight_type: len([i for i in active if i.insight_type == insight_type])
                for insight_type in ["commitment", "pattern", "anomaly", "proactive", "cross_domain"]
            }
        }
    
    async def run_proactive_monitor(self, interval: int = 300):
        """Run continuous proactive monitoring."""
        logger.info("Starting proactive insights monitor (interval: %ds)", interval)
        
        while True:
            try:
                await asyncio.sleep(interval)
                context = {
                    "hour": datetime.now(timezone.utc).hour,
                    "day_of_week": datetime.now(timezone.utc).strftime("%A"),
                }
                self.evaluate_rules(context)
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                logger.error("Error in proactive monitor: %s", type(e).__name__)
                await asyncio.sleep(5)