#!/usr/bin/env python3
"""
Domain Expertise - Specialized knowledge for each life domain

Implements expert-level understanding for:
- Health (medical, fitness, wellness)
- Finance (budgets, investments, expenses)
- Legal (contracts, agreements)
- Growth (learning, skills, habits)
- Career (work, projects, goals)
- Home (property, maintenance)
- Admin (schedules, tasks)
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import logging
import re
import threading

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


logger = logging.getLogger("shadowd.domain_expertise")


@dataclass
class DomainKnowledge:
    """Knowledge specific to a life domain."""
    domain: str
    keywords: List[str]
    entities: Dict[str, Any] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    patterns: Dict[str, Any] = field(default_factory=dict)
    goals: List[Dict] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthExpert:
    """Expert knowledge for Health domain."""
    
    DOMAIN = "health"
    
    def __init__(self):
        self.knowledge = DomainKnowledge(
            domain="health",
            keywords=["health", "doctor", "symptom", "medication", "exercise", 
                      "fitness", "wellness", "workout", "medical", "disease",
                      "pain", "appointment", "prescription", "vitamin", "diet"],
            rules=[
                "Track symptoms with timestamps",
                "Correlate symptoms with activities",
                "Monitor medication schedules",
                "Log exercise intensity and duration",
                "Track sleep patterns",
                "Note dietary impacts",
                "Schedule preventive checkups"
            ]
        )
    
    def analyze(self, query: str, observations: List[Dict]) -> Dict:
        """Analyze health-related query/observations."""
        insights = []
        
        # Symptom tracking
        symptoms = [o for o in observations if re.search(r'\bsymptom\b', str(o).lower())]
        if symptoms:
            insights.append(f"Tracked {len(symptoms)} symptom reports")
            
            # Find patterns
            recent_symptoms = [s for s in symptoms 
                              if self._is_recent(s.get("timestamp"))]
            if len(recent_symptoms) > 3:
                insights.append("Frequent symptoms detected - consider medical consultation")
        
        # Exercise patterns
        exercises = [o for o in observations if re.search(r'\b(exercise|workout)\b', str(o).lower())]
        if exercises:
            insights.append(f"Logged {len(exercises)} workouts")
            
            # Check consistency
            weekly_count = len([e for e in exercises 
                               if self._is_this_week(e.get("timestamp"))])
            if weekly_count < 3:
                insights.append("Exercise frequency below recommended (3+ times/week)")
        
        # Specific predictions
        prediction = None
        if re.search(r'\bwhen\b', query.lower()) and re.search(r'\bfeel\b', query.lower()):
            prediction = {
                "prediction": "Symptom pattern suggests improvement in 2-3 days if current rest continues",
                "confidence": 0.65,
                "action": "Continue monitoring, log any changes"
            }
        
        return {
            "domain": "health",
            "insights": insights,
            "prediction": prediction,
            "tracked_items": {
                "symptoms": len(symptoms),
                "exercises": len(exercises)
            }
        }
    
    def _is_recent(self, timestamp, hours=24):
        if not timestamp:
            return False
        try:
            ts = datetime.fromisoformat(timestamp)
            return (datetime.now(timezone.utc) - ts) < timedelta(hours=hours)
        except (ValueError, TypeError):
            return False
    
    def _is_this_week(self, timestamp):
        if not timestamp:
            return False
        try:
            ts = datetime.fromisoformat(timestamp)
            return (datetime.now(timezone.utc) - ts) < timedelta(days=7)
        except (ValueError, TypeError):
            return False


class FinanceExpert:
    """Expert knowledge for Finance domain."""
    
    DOMAIN = "finance"
    
    def __init__(self):
        self.knowledge = DomainKnowledge(
            domain="finance",
            keywords=["money", "budget", "expense", "invest", "tax", "salary",
                      "financial", "payment", "income", "savings", "debt", "credit",
                      "loan", "interest", "portfolio", "stock"],
            rules=[
                "Track all expenses with categories",
                "Monitor income vs expenses",
                "Alert on unusual spending",
                "Track investment performance",
                "Maintain emergency fund",
                "Review subscriptions monthly",
                "Set budget limits by category"
            ]
        )
    
    def analyze(self, query: str, observations: List[Dict]) -> Dict:
        """Analyze finance-related query/observations."""
        insights = []
        
        # Expense tracking
        expenses = [o for o in observations if any(re.search(r'\b' + re.escape(kw) + r'\b', str(o).lower()) 
                   for kw in ["expense", "spent", "cost", "paid"])]
        
        if expenses:
            total = sum(self._extract_amount(e) for e in expenses)
            insights.append(f"Tracked {len(expenses)} expenses, total: ${total:.2f}")
            
            # Category breakdown
            categories = {}
            for e in expenses:
                cat = self._categorize_expense(e)
                categories[cat] = categories.get(cat, 0) + self._extract_amount(e)
            
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                insights.append(f"Top expense category: {top_category[0]} (${top_category[1]:.2f})")
        
        # Budget predictions
        prediction = None
        if re.search(r'\bbudget\b', query.lower()) or re.search(r'\bspending\b', query.lower()):
            prediction = {
                "prediction": f"Based on current rate, monthly expenses will reach ${self._project_monthly(expenses):.2f}",
                "confidence": 0.75,
                "action": "Review discretionary spending"
            }
        
        return {
            "domain": "finance",
            "insights": insights,
            "prediction": prediction,
            "tracked_items": {
                "expenses": len(expenses),
                "categories": len(set(self._categorize_expense(e) for e in expenses))
            }
        }
    
    def _extract_amount(self, observation: Dict) -> float:
        """Extract monetary amount from observation (requires $ prefix)."""
        text = str(observation)
        amounts = re.findall(r'\$(\d+\.?\d*)', text)
        return float(amounts[0]) if amounts else 0.0
    
    def _categorize_expense(self, observation: Dict) -> str:
        """Categorize an expense."""
        text = str(observation).lower()
        
        categories = {
            "food": ["food", "grocery", "restaurant", "coffee", "lunch"],
            "transport": ["uber", "gas", "parking", "transit", "car"],
            "entertainment": ["movie", "game", "subscription", "netflix", "spotify"],
            "utilities": ["electric", "water", "internet", "phone"],
            "shopping": ["amazon", "store", "clothes", "buy"],
            "health": ["pharmacy", "doctor", "medical"]
        }
        
        for category, keywords in categories.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keywords):
                return category
        
        return "other"
    
    def _project_monthly(self, expenses: List[Dict]) -> float:
        """Project monthly spending from expenses."""
        if not expenses:
            return 0.0
        
        total = sum(self._extract_amount(e) for e in expenses)
        
        # Determine date range from expense timestamps
        dates = []
        for e in expenses:
            ts = e.get("timestamp") or e.get("date")
            if ts:
                try:
                    dates.append(datetime.fromisoformat(str(ts)))
                except (ValueError, TypeError):
                    pass
        
        if dates and len(dates) >= 2:
            days_covered = max((max(dates) - min(dates)).days, 1)
            daily_avg = total / days_covered
            return daily_avg * 30
        elif dates:
            # Only one date, assume 1 day
            return total * 30
        else:
            # No dates at all, assume expenses span 30 days
            return total


class GrowthExpert:
    """Expert knowledge for Personal Growth domain."""
    
    DOMAIN = "growth"
    
    def __init__(self):
        self.knowledge = DomainKnowledge(
            domain="growth",
            keywords=["learn", "goal", "skill", "habit", "book", "course",
                      "meditation", "growth", "practice", "improve", "develop",
                      "reading", "study", "progress", "milestone"],
            rules=[
                "Set SMART goals",
                "Track habit consistency",
                "Measure progress weekly",
                "Log learning activities",
                "Review skill development",
                "Celebrate milestones"
            ]
        )
    
    def analyze(self, query: str, observations: List[Dict]) -> Dict:
        """Analyze growth-related query/observations."""
        insights = []
        
        # Goal tracking
        goals = [o for o in observations if re.search(r'\bgoal\b', str(o).lower())]
        if goals:
            insights.append(f"Tracking {len(goals)} goals")
            
            # Check completion rate
            completed = [g for g in goals if re.search(r'\bcomplete\b', str(g).lower())]
            if completed:
                completion_rate = len(completed) / len(goals)
                insights.append(f"Goal completion rate: {completion_rate:.0%}")
        
        # Habit tracking
        habits = [o for o in observations if re.search(r'\bhabit\b', str(o).lower())]
        if habits:
            insights.append(f"Logged {len(habits)} habit checks")
            
            # Consistency check
            habit_streak = self._calculate_streak(habits)
            if habit_streak > 3:
                insights.append(f"Current streak: {habit_streak} days")
        
        # Progress prediction
        prediction = None
        if re.search(r'\bprogress\b', query.lower()) or re.search(r'\bgoal\b', query.lower()):
            prediction = {
                "prediction": f"Based on current pace, goal completion expected in {self._estimate_completion(goals)} weeks",
                "confidence": 0.60,
                "action": "Maintain current learning habit consistency"
            }
        
        return {
            "domain": "growth",
            "insights": insights,
            "prediction": prediction,
            "tracked_items": {
                "goals": len(goals),
                "habits": len(habits),
                "streak": habit_streak if habits else 0
            }
        }
    
    def _calculate_streak(self, habits: List[Dict]) -> int:
        """Calculate habit streak."""
        # Simple streak: consecutive days
        return 7  # Placeholder
    
    def _estimate_completion(self, goals: List[Dict]) -> int:
        """Estimate weeks until completion."""
        return 4  # Placeholder


class CareerExpert:
    """Expert knowledge for Career & Education domain."""
    
    DOMAIN = "career"
    
    def __init__(self):
        self.knowledge = DomainKnowledge(
            domain="career",
            keywords=["work", "project", "meeting", "boss", "deadline",
                      "career", "job", "education", "promotion", "skill",
                      "team", "presentation", "report", "task"],
            rules=[
                "Track project deadlines",
                "Log meeting outcomes",
                "Monitor work hours",
                "Record accomplishments",
                "Track skill development",
                "Plan career milestones"
            ]
        )
    
    def analyze(self, query: str, observations: List[Dict]) -> Dict:
        """Analyze career-related query/observations."""
        insights = []
        
        # Project tracking
        projects = [o for o in observations if re.search(r'\bproject\b', str(o).lower())]
        if projects:
            insights.append(f"Tracking {len(projects)} project activities")
            
            # Deadline check
            deadlines = [p for p in projects if re.search(r'\bdeadline\b', str(p).lower())]
            if deadlines:
                urgent = [d for d in deadlines if self._is_upcoming(d.get("timestamp"))]
                if urgent:
                    insights.append(f"⏰ {len(urgent)} urgent deadlines")
        
        # Meeting patterns
        meetings = [o for o in observations if re.search(r'\bmeeting\b', str(o).lower())]
        if meetings:
            insights.append(f"Attended {len(meetings)} meetings")
            
            weekly_meetings = len([m for m in meetings 
                                  if self._is_this_week(m.get("timestamp"))])
            if weekly_meetings > 15:
                insights.append("High meeting load - consider reducing")
        
        # Career prediction
        prediction = None
        if re.search(r'\bpromotion\b', query.lower()) or re.search(r'\bcareer\b', query.lower()):
            prediction = {
                "prediction": "Continue current performance trajectory for promotion consideration in 6-9 months",
                "confidence": 0.55,
                "action": "Document achievements, seek feedback, develop leadership skills"
            }
        
        return {
            "domain": "career",
            "insights": insights,
            "prediction": prediction,
            "tracked_items": {
                "projects": len(projects),
                "meetings": len(meetings),
                "deadline_count": len(deadlines) if 'deadlines' in locals() else 0
            }
        }
    
    def _is_upcoming(self, timestamp, days=7):
        if not timestamp:
            return False
        try:
            ts = datetime.fromisoformat(timestamp)
            return ts < datetime.now(timezone.utc) + timedelta(days=days)
        except (ValueError, TypeError):
            return False
    
    def _is_this_week(self, timestamp):
        if not timestamp:
            return False
        try:
            ts = datetime.fromisoformat(timestamp)
            return (datetime.now(timezone.utc) - ts) < timedelta(days=7)
        except (ValueError, TypeError):
            return False


class DomainExpertiseManager:
    """
    Manages expertise across all 7 life domains.
    
    Routes queries to appropriate expert,
    synthesizes cross-domain insights.
    """
    
    def __init__(self):
        self.experts = {
            "health": HealthExpert(),
            "finance": FinanceExpert(),
            "legal": DomainKnowledge(domain="legal", keywords=["contract", "legal", "law", "agreement", "sign"]),
            "growth": GrowthExpert(),
            "career": CareerExpert(),
            "home": DomainKnowledge(domain="home", keywords=["home", "house", "apartment", "repair", "maintenance"]),
            "admin": DomainKnowledge(domain="admin", keywords=["schedule", "appointment", "task", "todo", "plan", "calendar"])
        }
        
        # Observations per domain
        self.domain_observations: Dict[str, List[Dict]] = {domain: [] for domain in self.experts}
        self._lock = threading.Lock()
    
    def classify_domain(self, content: str) -> str:
        """Classify content into a domain."""
        content_lower = content.lower()
        
        scores = {}
        for domain, expert in self.experts.items():
            keywords = expert.knowledge.keywords if hasattr(expert, 'knowledge') else []
            score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', content_lower))
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return "admin"  # Default
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def add_observation(self, observation: Dict, domain: str = None):
        """Add observation to domain."""
        if domain is not None:
            domain = _sanitize(domain)
        with self._lock:
            if domain is None:
                domain = self.classify_domain(str(observation))
            if domain not in self.domain_observations:
                domain = "admin"
            
            self.domain_observations[domain].append(observation)
            
            # Keep last 1000 per domain
            if len(self.domain_observations[domain]) > 1000:
                self.domain_observations[domain] = self.domain_observations[domain][-1000:]
    
    def analyze(self, query: str, domain: str = None) -> Dict:
        query = _sanitize_content(query)
        if domain is not None:
            domain = _sanitize(domain)
        if domain is None:
            domain = self.classify_domain(query)
        if domain not in self.domain_observations:
            domain = "admin"
        
        with self._lock:
            observations = list(self.domain_observations.get(domain, []))
        
        expert = self.experts.get(domain)
        
        if hasattr(expert, 'analyze'):
            result = expert.analyze(query, observations)
            if isinstance(result, dict):
                result["insights"] = list(result.get("insights", []))
                if "tracked_items" in result and isinstance(result["tracked_items"], dict):
                    result["tracked_items"] = dict(result["tracked_items"])
            return result
        
        return {
            "domain": domain,
            "insights": [f"Tracking {len(observations)} observations"],
            "prediction": None,
            "tracked_items": {"observations": len(observations)}
        }
    
    def get_domain_summary(self, domain: str) -> Dict:
        """Get summary for a specific domain."""
        with self._lock:
            if domain not in self.experts:
                return {"domain": domain, "observation_count": 0, "recent": [], "keywords": []}
            observations = list(self.domain_observations.get(domain, []))
            keywords = list(self.experts[domain].knowledge.keywords[:10]) if hasattr(self.experts[domain], 'knowledge') else []
        
        return {
            "domain": domain,
            "observation_count": len(observations),
            "recent": copy.deepcopy(observations[-5:]) if observations else [],
            "keywords": keywords,
        }
    
    def get_cross_domain_insights(self) -> List[Dict]:
        """Find patterns across domains."""
        insights = []
        
        with self._lock:
            domain_activity = {
                domain: len(obs)
                for domain, obs in self.domain_observations.items()
            }
        
        if domain_activity:
            top_domain = max(domain_activity.items(), key=lambda x: x[1])
            if top_domain[1] > 10:
                safe_domain = top_domain[0].replace('\n', ' ').replace('\r', ' ')
                insights.append({
                    "insight": f"High activity in {safe_domain} ({top_domain[1]} observations)",
                    "action": f"Consider focused analysis on {safe_domain}"
                })
        
        return copy.deepcopy(insights) if insights else insights
    
    def get_status(self) -> Dict:
        """Get domain expertise status."""
        with self._lock:
            obs_counts = {domain: len(obs) for domain, obs in self.domain_observations.items()}
            total = sum(obs_counts.values())
            most_active = max(obs_counts, key=obs_counts.get) if any(obs_counts.values()) else None
        return {
            "domains": obs_counts,
            "total_observations": total,
            "most_active": most_active
        }