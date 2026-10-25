#!/usr/bin/env python3
"""
Daily Briefing System - Morning Intelligence Report

Generates personalized daily briefings that combine:
- Overnight observations and events
- Pending tasks and commitments
- Predictions and warnings
- Personality-tuned communication style
- Domain-specific insights across 7 life areas

Every morning, your shadow produces a briefing based on what happened
while you slept, what needs attention today, and what it handled on your behalf.

Inspired by: Presidential daily briefings, Strava year-in-review,
and the concept of "digital twin morning report".
"""

import asyncio
import copy
import re
import threading
import uuid
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("shadowd.cognition.daily_briefing")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


class BriefingSection(str, Enum):
    OVERNIGHT = "overnight"
    TODAY = "today"
    PREDICTIONS = "predictions"
    ACTIONS_TAKEN = "actions_taken"
    INSIGHTS = "insights"
    WELLNESS = "wellness"
    REMINDERS = "reminders"


class UrgencyLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BriefingItem:
    section: BriefingSection
    title: str
    content: str
    urgency: UrgencyLevel = UrgencyLevel.INFO
    domain: Optional[str] = None
    action_required: bool = False
    action_url: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.title = _SANITIZE_STRICT_RE.sub('', self.title)
        self.content = _SANITIZE_RE.sub('', self.content)
        if self.domain:
            self.domain = _SANITIZE_STRICT_RE.sub('', self.domain).strip()[:100]
        if self.action_url:
            self.action_url = _SANITIZE_STRICT_RE.sub('', self.action_url).strip()[:500]
        if len(self.metadata) > 20:
            self.metadata = dict(list(self.metadata.items())[:20])

    def to_dict(self) -> Dict:
        return {
            "section": self.section.value,
            "title": self.title,
            "content": self.content,
            "urgency": self.urgency.value,
            "domain": self.domain,
            "action_required": self.action_required,
            "action_url": self.action_url,
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class DailyBriefing:
    briefing_id: str = ""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    items: List[BriefingItem] = field(default_factory=list)
    summary: str = ""
    overnight_count: int = 0
    predictions_count: int = 0
    actions_on_behalf: int = 0
    wellness_score: float = 0.5
    personality_style: str = "casual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.briefing_id:
            self.briefing_id = f"brief_{self.date}_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        sections = {}
        for item in self.items:
            if item.section.value not in sections:
                sections[item.section.value] = []
            sections[item.section.value].append(item.to_dict())

        return {
            "briefing_id": self.briefing_id,
            "date": self.date,
            "generated_at": self.generated_at,
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary,
            "overnight_count": self.overnight_count,
            "predictions_count": self.predictions_count,
            "actions_on_behalf": self.actions_on_behalf,
            "wellness_score": round(self.wellness_score, 2),
            "personality_style": self.personality_style,
            "metadata": copy.deepcopy(self.metadata),
            "sections": sections,
            "total_items": len(self.items),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Shadow Briefing - {self.date}",
            "",
            f"**Summary:** {self.summary}",
            "",
            f"- Overnight events: {self.overnight_count}",
            f"- Predictions: {self.predictions_count}",
            f"- Actions taken on your behalf: {self.actions_on_behalf}",
            f"- Wellness score: {self.wellness_score:.0%}",
            "",
        ]

        sections = {}
        for item in self.items:
            if item.section.value not in sections:
                sections[item.section.value] = []
            sections[item.section.value].append(item)

        section_icons = {
            "overnight": "🌙",
            "today": "📅",
            "predictions": "🔮",
            "actions_taken": "⚡",
            "insights": "💡",
            "wellness": "❤️",
            "reminders": "🔔",
        }

        for section_name, items in sections.items():
            icon = section_icons.get(section_name, "•")
            lines.append(f"## {icon} {section_name.replace('_', ' ').title()}")
            lines.append("")
            for item in items:
                urgency_marker = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "ℹ️"}
                marker = urgency_marker.get(item.urgency.value, "•")
                action = " **[Action Required]**" if item.action_required else ""
                lines.append(f"- {marker} **{item.title}**{action}: {item.content}")
            lines.append("")

        return "\n".join(lines)


class DailyBriefingEngine:
    """
    Generates personalized daily briefings by combining data from
    all ShadowOS modules: DarkMemory, PersonalityEngine, ShadowCrew, etc.
    """

    _OVERNIGHT_HOURS = 8
    _RECENT_MEMORY_LIMIT = 50
    _MAX_BRIEFING_HISTORY = 30

    def __init__(self, memory=None, personality=None, mesh=None, evolution_guide=None):
        self.memory = memory
        self.personality = personality
        self.mesh = mesh
        self.evolution_guide = evolution_guide
        self._briefing_history: List[DailyBriefing] = []
        self._wellness_history: deque = deque(maxlen=30)
        self._history_lock = threading.Lock()
        self._schedule_task: Optional[asyncio.Task] = None
        self._schedule_hour: int = 7
        self._schedule_timezone: str = "America/New_York"
        self._wake_event: asyncio.Event = asyncio.Event()

    async def generate_briefing(self, device_context=None, evolution_guide=None) -> DailyBriefing:
        briefing = DailyBriefing()

        await self._add_overnight_events(briefing)
        await self._add_today_outlook(briefing)
        await self._add_predictions(briefing)
        await self._add_actions_taken(briefing)
        await self._add_insights(briefing, evolution_guide=evolution_guide)
        await self._add_wellness(briefing, device_context=device_context)
        await self._add_reminders(briefing)

        briefing.summary = self._generate_summary(briefing)
        briefing.wellness_score = self._calculate_wellness(briefing, device_context=device_context)

        wellness_metrics = self.get_wellness_metrics(device_context=device_context)
        briefing.metadata["wellness_metrics"] = wellness_metrics

        if self.personality:
            profile = self.personality.get_profile()
            briefing.personality_style = profile.get("communication_style", "casual")

        briefing.items.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(x.urgency.value, 5))

        with self._history_lock:
            self._briefing_history.append(briefing)
            if len(self._briefing_history) > self._MAX_BRIEFING_HISTORY:
                self._briefing_history = self._briefing_history[-self._MAX_BRIEFING_HISTORY:]

        return briefing

    async def _add_overnight_events(self, briefing: DailyBriefing):
        if not self.memory:
            briefing.items.append(BriefingItem(
                section=BriefingSection.OVERNIGHT,
                title="No overnight data",
                content="Memory system not connected",
                urgency=UrgencyLevel.INFO,
            ))
            return

        recent = await self.memory.get_recent(limit=self._RECENT_MEMORY_LIMIT)
        now = datetime.now(timezone.utc)
        overnight_cutoff = now - timedelta(hours=self._OVERNIGHT_HOURS)

        overnight = []
        for atom in recent:
            try:
                atom_time = datetime.fromisoformat(atom.created_at)
                if atom_time.tzinfo is None:
                    atom_time = atom_time.replace(tzinfo=timezone.utc)
                if atom_time > overnight_cutoff:
                    overnight.append(atom)
            except (ValueError, TypeError, AttributeError):
                continue

        briefing.overnight_count = len(overnight)

        if overnight:
            by_type = {}
            for atom in overnight:
                by_type[atom.atom_type.value] = by_type.get(atom.atom_type.value, 0) + 1

            for atype, count in by_type.items():
                briefing.items.append(BriefingItem(
                    section=BriefingSection.OVERNIGHT,
                    title=f"{atype.title()} events",
                    content=f"{count} {atype} events recorded overnight",
                    urgency=UrgencyLevel.INFO,
                ))
        else:
            briefing.items.append(BriefingItem(
                section=BriefingSection.OVERNIGHT,
                title="Quiet night",
                content="No significant events overnight",
                urgency=UrgencyLevel.INFO,
            ))

    async def _add_today_outlook(self, briefing: DailyBriefing):
        briefing.items.append(BriefingItem(
            section=BriefingSection.TODAY,
            title="Daily planning",
            content="Your shadow is ready to assist. Key items below.",
            urgency=UrgencyLevel.MEDIUM,
            action_required=True,
        ))

    async def _add_predictions(self, briefing: DailyBriefing):
        if not self.personality:
            return

        result = self.personality.predict_decision([
            {"name": "productive_day", "reward": 0.8, "risk_level": 0.2, "novelty": 0.3},
            {"name": "exploratory_day", "reward": 0.6, "risk_level": 0.4, "novelty": 0.7},
        ])

        predicted = "productive" if result.get("predicted_choice") == 0 else "exploratory"
        confidence = result.get("confidence", 0.5)

        briefing.predictions_count = 1
        briefing.items.append(BriefingItem(
            section=BriefingSection.PREDICTIONS,
            title="Today's trajectory",
            content=f"Based on your patterns, today is likely to be a {predicted} day ({confidence:.0%} confidence)",
            urgency=UrgencyLevel.INFO,
            domain="prediction",
        ))

    async def _add_actions_taken(self, briefing: DailyBriefing):
        briefing.actions_on_behalf = 0
        briefing.items.append(BriefingItem(
            section=BriefingSection.ACTIONS_TAKEN,
            title="Autonomous actions",
            content="No actions taken on your behalf overnight",
            urgency=UrgencyLevel.INFO,
        ))

    async def _add_insights(self, briefing: DailyBriefing, evolution_guide=None):
        briefing.items.append(BriefingItem(
            section=BriefingSection.INSIGHTS,
            title="Pattern insight",
            content="Your shadow is still learning your patterns. More observations will yield better insights.",
            urgency=UrgencyLevel.INFO,
        ))
        if evolution_guide is not None:
            try:
                evo_items = evolution_guide.get_briefing_evolution_items(limit=10)
                for item in evo_items:
                    item_type = item.get("type", "insight")
                    domain = item.get("domain", "work")
                    description = item.get("description", "")
                    rationale = item.get("rationale", "")
                    if not description and not rationale:
                        continue
                    if item_type == "friction":
                        severity = item.get("severity", "medium")
                        urgency = UrgencyLevel.HIGH if severity in ("high", "critical") else UrgencyLevel.MEDIUM if severity == "medium" else UrgencyLevel.LOW
                        briefing.items.append(BriefingItem(
                            section=BriefingSection.INSIGHTS,
                            title=f"Friction detected in {domain}",
                            content=description,
                            urgency=urgency,
                            domain=domain,
                        ))
                    elif item_type == "suggestion":
                        content = description if description else (rationale if rationale else "Optimization suggestion available")
                        if rationale and description:
                            content = f"{description} — {rationale}"
                        briefing.items.append(BriefingItem(
                            section=BriefingSection.INSIGHTS,
                            title=f"Evolution suggestion for {domain}",
                            content=content,
                            urgency=UrgencyLevel.LOW,
                            domain=domain,
                        ))
            except (ValueError, TypeError, AttributeError):
                pass

    async def _add_wellness(self, briefing: DailyBriefing, device_context=None):
        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                parts = []
                health = ctx.health
                if health.heart_rate is not None:
                    parts.append(f"Heart rate: {health.heart_rate} bpm")
                if health.hrv is not None:
                    parts.append(f"HRV: {health.hrv:.0f} ms")
                if health.spo2 is not None:
                    parts.append(f"SpO2: {health.spo2:.0f}%")
                if health.step_count > 0:
                    parts.append(f"Steps: {health.step_count}")
                stress_text = health.stress_level.value.replace("_", " ").title()
                parts.append(f"Stress level: {stress_text}")
                if health.sleep_minutes_last_24h > 0:
                    parts.append(f"Sleep: {health.sleep_minutes_last_24h // 60}h {health.sleep_minutes_last_24h % 60}m")
                content = " | ".join(parts) if parts else "Limited health data available"
                focus_val = ctx.focus.value.replace("_", " ").title()
                activity_val = ctx.activity.value.replace("_", " ").title()
                content += f" | Focus: {focus_val} | Activity: {activity_val}"
                urgency = UrgencyLevel.INFO
                if health.stress_level.value in ("high", "critical"):
                    urgency = UrgencyLevel.HIGH
                elif health.stress_level.value == "moderate":
                    urgency = UrgencyLevel.MEDIUM
                briefing.items.append(BriefingItem(
                    section=BriefingSection.WELLNESS,
                    title="Wellness check",
                    content=content,
                    urgency=urgency,
                    domain="health",
                ))
                return
        briefing.items.append(BriefingItem(
            section=BriefingSection.WELLNESS,
            title="Wellness check",
            content="Baseline wellness: 50%. Will improve with more data.",
            urgency=UrgencyLevel.INFO,
            domain="health",
        ))

    async def _add_reminders(self, briefing: DailyBriefing):
        briefing.items.append(BriefingItem(
            section=BriefingSection.REMINDERS,
            title="Daily reflection",
            content="Consider recording key decisions today for better pattern analysis",
            urgency=UrgencyLevel.LOW,
            action_required=False,
        ))

    def _generate_summary(self, briefing: DailyBriefing) -> str:
        urgent_count = sum(1 for i in briefing.items if i.urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL))
        action_count = sum(1 for i in briefing.items if i.action_required)

        if urgent_count > 0:
            return f"You have {urgent_count} urgent item(s) and {action_count} action(s) required. {briefing.overnight_count} events overnight."
        return f"Relatively calm morning. {briefing.overnight_count} events overnight. {action_count} action(s) needed today."

    def _calculate_wellness(self, briefing: DailyBriefing, device_context=None) -> float:
        base = 0.7
        urgent = sum(1 for i in briefing.items if i.urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL))
        medium = sum(1 for i in briefing.items if i.urgency == UrgencyLevel.MEDIUM)
        score = max(0.1, min(1.0, base - urgent * 0.1 - medium * 0.03))

        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                stress_penalties = {
                    "critical": 0.3, "high": 0.15,
                    "moderate": 0.05, "low": 0.0, "unknown": 0.0,
                }
                score -= stress_penalties.get(ctx.health.stress_level.value, 0.0)
                if ctx.health.sleep_minutes_last_24h > 0 and ctx.health.sleep_minutes_last_24h < 300:
                    score -= 0.1
                if ctx.health.step_count > 5000:
                    score += 0.05
                if ctx.health.exercise_minutes > 30:
                    score += 0.05
                score = max(0.1, min(1.0, score))
        return score

    def get_wellness_metrics(self, device_context=None) -> Dict[str, Any]:
        with self._history_lock:
            if not self._briefing_history:
                return {
                    "overall_score": 0.7,
                    "components": {"sleep": 0.7, "activity": 0.7, "social": 0.7, "stress": 0.7},
                    "trend": "stable",
                    "recommendations": ["Generate your first briefing to get wellness insights"],
                }

            latest = self._briefing_history[-1]
        score = self._calculate_wellness(latest)
        self._wellness_history.append(score)

        urgent = sum(1 for i in latest.items if i.urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL))
        medium = sum(1 for i in latest.items if i.urgency == UrgencyLevel.MEDIUM)
        total = len(latest.items)

        stress = max(0.0, min(1.0, 0.3 + urgent * 0.15 + medium * 0.05))
        sleep = max(0.1, min(1.0, 0.8 - urgent * 0.1))
        activity = max(0.1, min(1.0, 0.5 + (total - urgent) * 0.05)) if total > 0 else 0.5
        social = max(0.1, min(1.0, 0.6 - urgent * 0.1))

        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                stress_map = {"low": 0.2, "moderate": 0.5, "high": 0.8, "critical": 0.95, "unknown": 0.3}
                stress = max(0.0, min(1.0, stress_map.get(ctx.health.stress_level.value, 0.3)))
                if ctx.health.sleep_minutes_last_24h > 0:
                    sleep = min(1.0, ctx.health.sleep_minutes_last_24h / 480)
                if ctx.health.step_count > 0:
                    activity = min(1.0, 0.3 + min(ctx.health.step_count / 10000, 0.7))
                if ctx.health.exercise_minutes > 30:
                    activity = min(1.0, activity + 0.1)

        components = {
            "sleep": round(sleep, 2),
            "activity": round(activity, 2),
            "social": round(social, 2),
            "stress": round(stress, 2),
        }

        trend = "stable"
        if len(self._wellness_history) >= 6:
            recent = list(self._wellness_history)
            avg_recent = sum(recent[-3:]) / 3
            avg_prev = sum(recent[-6:-3]) / 3
            if avg_recent > avg_prev + 0.05:
                trend = "improving"
            elif avg_recent < avg_prev - 0.05:
                trend = "declining"

        recommendations = []
        if stress > 0.6:
            recommendations.append("High stress detected - consider reducing urgent commitments")
        if sleep < 0.5:
            recommendations.append("Rest may be needed - many urgent items impacting rest score")
        if social < 0.4:
            recommendations.append("Low social wellness - consider connecting with others")
        if not recommendations:
            recommendations.append("Wellness looks good - keep maintaining healthy patterns")

        return {
            "overall_score": round(score, 2),
            "components": components,
            "trend": trend,
            "recommendations": recommendations,
        }

    def trigger_briefing(self):
        self._wake_event.set()

    def get_briefing_history(self, limit: int = 7) -> List[Dict]:
        with self._history_lock:
            return [b.to_dict() for b in self._briefing_history[-limit:]]

    def get_latest_briefing_markdown(self) -> str:
        with self._history_lock:
            if not self._briefing_history:
                return "No briefings generated yet."
            return self._briefing_history[-1].to_markdown()

    async def start_scheduled_briefings(self, hour: int = 7):
        self._schedule_hour = hour
        self._schedule_task = asyncio.create_task(self._briefing_schedule_loop())

    async def stop_scheduled_briefings(self):
        if self._schedule_task and not self._schedule_task.done():
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass

    async def _briefing_schedule_loop(self):
        while True:
            try:
                tz = ZoneInfo(self._schedule_timezone)
                now_local = datetime.now(tz)
                target_local = now_local.replace(hour=self._schedule_hour, minute=0, second=0, microsecond=0)
                if target_local <= now_local:
                    target_local = target_local + timedelta(days=1)
                wait_seconds = (target_local - now_local).total_seconds()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=wait_seconds)
                    self._wake_event.clear()
                except asyncio.TimeoutError:
                    pass
                briefing = await self.generate_briefing(evolution_guide=self.evolution_guide)
                logger.info("Scheduled briefing generated: %s", briefing.briefing_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduled briefing error: %s", type(e).__name__)
                await asyncio.sleep(3600)