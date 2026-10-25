"""
Cross-Module Intelligence Report Engine — Unified system intelligence aggregation.

Aggregates metrics, trends, anomalies, and recommendations from ALL cognitive
modules into a single actionable intelligence report. This is the executive
summary of the cognitive architecture's state.

Inputs from:
- PerformanceProfiler: latency profiles, error rates, cache efficiency
- CognitiveVisualizer: module interaction graph, timeline, resource heatmap
- CrossModuleSynthesisEngine: cross-category syntheses, emergent properties
- HealthDashboard: module health scores, alerts
- AnomalyDetector: detected anomalies across metrics
- GoalReasoning: goal assessments, obstacle detections
- EvolutionaryEngine: convergence state, fitness trends

Outputs:
- System health score (0-1 composite)
- Module statuses (healthy/degraded/critical)
- Top anomalies (by severity)
- Cross-module insights (syntheses)
- Optimization recommendations (ordered by impact)
- Trend forecasts (next 24h predictions)
- Action items (prioritized todo list)
"""

import copy
import logging
import re
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 5000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]


_MAX_REPORTS = 200
_MAX_ACTIONS = 50
_MAX_ANOMALIES = 20
_MAX_RECOMMENDATIONS = 20
_MAX_INSIGHTS = 10


class ReportSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ActionPriority(Enum):
    IMMEDIATE = "immediate"
    TODAY = "today"
    THIS_WEEK = "this_week"
    BACKLOG = "backlog"


@dataclass
class ReportAction:
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    priority: str = "medium"
    category: str = ""
    module_source: str = ""
    estimated_effort_minutes: int = 30
    expected_impact: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False

    def __post_init__(self):
        self.title = _sanitize_content(self.title, 200)
        self.description = _sanitize_content(self.description, 2000)
        self.priority = _sanitize(self.priority, 20)
        self.category = _sanitize(self.category, 50)
        self.module_source = _sanitize(self.module_source, 100)
        self.estimated_effort_minutes = max(1, min(480, self.estimated_effort_minutes))
        self.expected_impact = max(0.0, min(1.0, self.expected_impact))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "description": self.description[:200],
            "priority": self.priority,
            "category": self.category,
            "module_source": self.module_source,
            "estimated_effort_minutes": self.estimated_effort_minutes,
            "expected_impact": round(self.expected_impact, 3),
            "resolved": self.resolved,
        }


@dataclass
class IntelligenceReport:
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    report_type: str = "hourly"

    system_health_score: float = 1.0
    module_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    top_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    cross_module_insights: List[Dict[str, Any]] = field(default_factory=list)
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    trend_forecasts: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)

    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.report_type = _sanitize(self.report_type, 50)
        self.system_health_score = max(0.0, min(1.0, self.system_health_score))
        self.summary = _sanitize_content(self.summary, 10000)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "report_type": self.report_type,
            "system_health_score": round(self.system_health_score, 3),
            "module_status_count": len(self.module_statuses),
            "module_statuses": copy.deepcopy(self.module_statuses),
            "anomaly_count": len(self.top_anomalies),
            "top_anomalies": list(self.top_anomalies),
            "insight_count": len(self.cross_module_insights),
            "cross_module_insights": list(self.cross_module_insights),
            "recommendation_count": len(self.optimization_recommendations),
            "optimization_recommendations": list(self.optimization_recommendations),
            "forecast_count": len(self.trend_forecasts),
            "trend_forecasts": list(self.trend_forecasts),
            "action_count": len(self.action_items),
            "action_items": list(self.action_items),
            "summary": self.summary[:500],
            "metadata": copy.deepcopy(self.metadata),
        }


class IntelligenceReportEngine:
    _MAX_REPORTS = _MAX_REPORTS
    _MAX_ACTIONS = _MAX_ACTIONS

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._reports: dict[str, IntelligenceReport] = {}
        self._report_order: deque = deque(maxlen=_MAX_REPORTS)
        self._pending_actions: dict[str, ReportAction] = {}
        self._action_order: deque = deque(maxlen=_MAX_ACTIONS)

        self._stats: Dict[str, int] = {
            "reports_generated": 0,
            "actions_created": 0,
            "actions_resolved": 0,
            "total_anomalies_reported": 0,
            "total_insights_reported": 0,
            "total_recommendations_reported": 0,
        }

    def generate_report(
        self,
        shadow_os: Optional[Any] = None,
        report_type: str = "hourly",
    ) -> IntelligenceReport:
        """Generate a comprehensive intelligence report from all cognitive modules."""
        report_type = _sanitize(report_type, 50)

        report = IntelligenceReport(
            report_type=report_type,
            system_health_score=1.0,
        )

        if shadow_os is None:
            report.summary = "ShadowOS not available — cannot generate full report."
            return self._store_report(report)

        try:
            self._collect_module_statuses(shadow_os, report)
            self._collect_anomalies(shadow_os, report)
            self._collect_insights(shadow_os, report)
            self._collect_recommendations(shadow_os, report)
            self._collect_forecasts(shadow_os, report)
            self._derive_actions(report)
            self._compute_health_score(report)
            self._generate_summary(report)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Report generation partial failure: %s", type(e).__name__)
            report.summary = f"Report partially generated (errors encountered: {type(e).__name__})"

        return self._store_report(report)

    def resolve_action(self, action_id: str) -> bool:
        action_id = _sanitize(action_id, 20)
        with self._lock:
            action = self._pending_actions.get(action_id)
            if action is None:
                return False
            action.resolved = True
        with self._stats_lock:
            self._stats["actions_resolved"] += 1
        return True

    def get_pending_actions(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_ACTIONS, limit))
        with self._lock:
            actions = [a for a in self._pending_actions.values() if not a.resolved]
            actions.sort(key=lambda a: (
                {"immediate": 0, "today": 1, "this_week": 2, "backlog": 3}.get(a.priority, 3),
                -a.expected_impact,
            ))
            return [a.to_dict() for a in actions[:limit]]

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            reports = sorted(
                list(self._reports.values()),
                key=lambda r: r.generated_at,
                reverse=True,
            )
            if reports:
                return reports[0].to_dict()
        return None

    def get_report_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_REPORTS, limit))
        with self._lock:
            reports = sorted(
                list(self._reports.values()),
                key=lambda r: r.generated_at,
                reverse=True,
            )
            return [r.to_dict() for r in reports[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_reports = len(self._reports)
            pending_count = sum(1 for a in self._pending_actions.values() if not a.resolved)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_reports"] = total_reports
        stats["pending_actions"] = pending_count
        return stats

    def _store_report(self, report: IntelligenceReport) -> IntelligenceReport:
        with self._lock:
            self._reports[report.report_id] = report
            self._report_order.append(report.report_id)
            while len(self._reports) > self._MAX_REPORTS:
                old_id = self._report_order.popleft()
                self._reports.pop(old_id, None)

        with self._stats_lock:
            self._stats["reports_generated"] += 1

        return report

    def _collect_module_statuses(self, shadow_os, report: IntelligenceReport) -> None:
        try:
            health = shadow_os.get_health()
            modules = health.get("modules", {})

            for mod_name, status in modules.items():
                report.module_statuses[_sanitize(mod_name, 50)] = {
                    "healthy": True,
                    "detail": str(status)[:100] if status else "unknown",
                }

            report.system_health_score = health.get("stats", {}).get(
                "health_score",
                report.system_health_score,
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Module status collection failed: %s", type(e).__name__)

    def _collect_anomalies(self, shadow_os, report: IntelligenceReport) -> None:
        try:
            anomaly_data = shadow_os.anomaly_detector.get_stats()
            if isinstance(anomaly_data, dict):
                anomaly_count = anomaly_data.get("total_anomalies", 0)
                if anomaly_count > 0:
                    report.top_anomalies.append({
                        "type": "cognitive_anomalies",
                        "count": anomaly_count,
                        "severity": "medium",
                        "detail": f"{anomaly_count} anomalies detected in cognitive metrics",
                    })
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    def _collect_insights(self, shadow_os, report: IntelligenceReport) -> None:
        try:
            syntheses = shadow_os.synthesis_engine.get_synthesis_history(
                limit=_MAX_INSIGHTS,
                active_only=True,
            )
            for syn in syntheses[:5]:
                report.cross_module_insights.append({
                    "title": syn.get("title", "")[:200],
                    "confidence": syn.get("confidence", 0),
                    "action": syn.get("action_recommendation", "")[:200],
                })

            with self._stats_lock:
                self._stats["total_insights_reported"] += len(report.cross_module_insights)

        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    def _collect_recommendations(self, shadow_os, report: IntelligenceReport) -> None:
        try:
            recs = shadow_os.profiler.generate_recommendations()
            report.optimization_recommendations = [
                r for r in recs
                if r.get("priority") in ("high", "critical")
            ][:_MAX_RECOMMENDATIONS]

            with self._stats_lock:
                self._stats["total_recommendations_reported"] += len(
                    report.optimization_recommendations
                )

        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    def _collect_forecasts(self, shadow_os, report: IntelligenceReport) -> None:
        try:
            trends = shadow_os.synthesis_engine.get_trends(
                min_forecast_confidence=0.3,
            )
            report.trend_forecasts = trends[:5]
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    def _derive_actions(self, report: IntelligenceReport) -> None:
        actions = []

        for anomaly in report.top_anomalies[:3]:
            action = ReportAction(
                title=f"Investigate: {anomaly.get('type', 'unknown')}",
                description=anomaly.get("detail", "")[:500],
                priority=anomaly.get("severity", "medium"),
                category="anomaly_investigation",
                module_source="intelligence_report",
                expected_impact=0.5,
            )
            actions.append(action)

        for insight in report.cross_module_insights[:3]:
            action = ReportAction(
                title=f"Explore insight: {insight.get('title', '')[:80]}",
                description=insight.get("action", "")[:500],
                priority="medium",
                category="insight_exploration",
                module_source="synthesis_engine",
                expected_impact=insight.get("confidence", 0.3),
            )
            actions.append(action)

        for rec in report.optimization_recommendations[:3]:
            action = ReportAction(
                title=f"Optimize: {rec.get('module_name', 'unknown')}",
                description=rec.get("description", "")[:500],
                priority=rec.get("priority", "medium"),
                category="optimization",
                module_source=rec.get("module_name", "profiler"),
                expected_impact=rec.get("expected_improvement_pct", 30) / 100,
            )
            actions.append(action)

        report.action_items = [a.to_dict() for a in actions]

        with self._lock:
            for a in actions:
                self._pending_actions[a.action_id] = a
                self._action_order.append(a.action_id)
            while len(self._pending_actions) > self._MAX_ACTIONS:
                old = self._action_order.popleft()
                self._pending_actions.pop(old, None)

        with self._stats_lock:
            self._stats["actions_created"] += len(actions)

    def _compute_health_score(self, report: IntelligenceReport) -> None:
        module_count = len(report.module_statuses)
        if module_count == 0:
            report.system_health_score = 0.0
            return

        healthy = sum(
            1 for s in report.module_statuses.values()
            if s.get("healthy", False)
        )
        base_score = healthy / module_count

        anomaly_penalty = 0.05 * min(len(report.top_anomalies), 5)
        report.system_health_score = round(
            max(0.0, base_score - anomaly_penalty),
            3,
        )

    def _generate_summary(self, report: IntelligenceReport) -> None:
        parts = []

        score_pct = round(report.system_health_score * 100)
        if report.system_health_score >= 0.9:
            parts.append(f"System health: {score_pct}% — Excellent. All modules operating normally.")
        elif report.system_health_score >= 0.7:
            parts.append(f"System health: {score_pct}% — Good. Minor issues detected but non-critical.")
        elif report.system_health_score >= 0.5:
            parts.append(f"System health: {score_pct}% — Fair. Attention required.")
        else:
            parts.append(f"System health: {score_pct}% — Needs immediate attention.")

        if report.top_anomalies:
            parts.append(
                f"{len(report.top_anomalies)} anomaly types detected."
            )

        if report.cross_module_insights:
            parts.append(
                f"{len(report.cross_module_insights)} cross-module insights available."
            )

        if report.optimization_recommendations:
            parts.append(
                f"{len(report.optimization_recommendations)} high-priority optimizations recommended."
            )

        report.summary = " ".join(parts)
