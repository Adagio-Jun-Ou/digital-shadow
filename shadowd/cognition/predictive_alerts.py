#!/usr/bin/env python3
"""
PredictiveAlertEngine - Preemptive intelligence for ShadowOS.

Analyzes patterns across all modules to predict and prevent issues
before they happen. Uses trend analysis, anomaly correlation, and
temporal pattern recognition to generate preemptive alerts.

Prediction categories:
- System health degradation (from health_dashboard trends)
- Security threat escalation (from security_guardian patterns)
- Cognitive overload (from attention_engine + emotional_intelligence)
- Resource exhaustion (from optimization_daemon + system metrics)
- Relationship deterioration (from empathy_mapping trends)
"""

import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("shadowd.cognition.predictive_alerts")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


class PredictionCategory(str, Enum):
    HEALTH = "health"
    SECURITY = "security"
    COGNITIVE = "cognitive"
    RESOURCE = "resource"
    SOCIAL = "social"
    BEHAVIOR = "behavior"
    SYSTEM = "system"


class ConfidenceTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PredictiveAlert:
    alert_id: str = ""
    category: PredictionCategory = PredictionCategory.SYSTEM
    title: str = ""
    description: str = ""
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expected_onset: Optional[datetime] = None
    confidence: float = 0.5
    confidence_tier: ConfidenceTier = ConfidenceTier.MEDIUM
    severity: float = 0.5
    evidence: List[str] = field(default_factory=list)
    recommended_action: str = ""
    source_modules: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = uuid.uuid4().hex[:12]
        self.title = _sanitize_content(self.title, 200)
        self.description = _sanitize_content(self.description, 5000)
        self.recommended_action = _sanitize_content(self.recommended_action, 2000)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.severity = max(0.0, min(1.0, self.severity))
        self.evidence = [_sanitize_content(e, 500) for e in self.evidence[:50]]
        self.source_modules = [_sanitize(m, 100) for m in self.source_modules[:20]]
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "predicted_at": self.predicted_at.isoformat(),
            "expected_onset": self.expected_onset.isoformat() if self.expected_onset else None,
            "confidence": round(self.confidence, 3),
            "confidence_tier": self.confidence_tier.value,
            "severity": round(self.severity, 3),
            "evidence": list(self.evidence),
            "recommended_action": self.recommended_action,
            "source_modules": list(self.source_modules),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "metadata": copy.deepcopy(self.metadata),
        }


class PredictiveAlertEngine:
    """Preemptive intelligence engine for ShadowOS."""

    def __init__(self, shadow_os=None):
        self._shadow_os = shadow_os
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._alerts: Dict[str, PredictiveAlert] = {}
        self._alert_order: deque = deque(maxlen=5000)
        self._history: deque = deque(maxlen=200)

        self._STAT_PATTERNS = {}
        self._last_analysis: Dict[str, float] = {}

        self._stats = {
            "alerts_generated": 0,
            "predictions_made": 0,
            "critical_predictions": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
        }

    def analyze_and_predict(self) -> List[PredictiveAlert]:
        """Run full prediction pipeline across all modules. Returns alerts generated."""
        alerts = []
        os = self._shadow_os
        if os is None:
            return alerts

        alerts.extend(self._predict_health_issues(os))
        alerts.extend(self._predict_security_threats(os))
        alerts.extend(self._predict_cognitive_overload(os))
        alerts.extend(self._predict_resource_exhaustion(os))
        alerts.extend(self._predict_social_deterioration(os))

        for alert in alerts:
            with self._lock:
                self._alerts[alert.alert_id] = alert
                self._alert_order.append(alert.alert_id)
                self._history.append({
                    "alert_id": alert.alert_id,
                    "category": alert.category.value,
                    "confidence": alert.confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        with self._stats_lock:
            self._stats["alerts_generated"] += len(alerts)
            self._stats["predictions_made"] += len(alerts)
            critical = sum(1 for a in alerts if a.confidence_tier == ConfidenceTier.CRITICAL)
            self._stats["critical_predictions"] += critical

        return alerts

    def _predict_health_issues(self, os) -> List[PredictiveAlert]:
        alerts = []
        try:
            if not hasattr(os, 'health_dashboard') or not os.health_dashboard:
                return alerts
            dashboard = os.health_dashboard
            report = dashboard.generate_report()
            modules = report.get("modules", {})
            for mod_name, mod_data in modules.items():
                error_rate = mod_data.get("error_rate", 0)
                if error_rate > 0.3:
                    alerts.append(PredictiveAlert(
                        category=PredictionCategory.HEALTH,
                        title=f"Module {mod_name} degrading",
                        description=f"Error rate {round(error_rate*100,1)}% - risk of failure. "
                                    f"Success rate: {round(mod_data.get('success_rate',1)*100,1)}%",
                        confidence=min(0.95, error_rate + 0.3),
                        confidence_tier=ConfidenceTier.HIGH if error_rate > 0.5 else ConfidenceTier.MEDIUM,
                        severity=error_rate,
                        evidence=[f"error_rate={round(error_rate,3)}"],
                        recommended_action=f"Investigate {mod_name} logs, consider restart or rollback",
                        source_modules=["health_dashboard", mod_name],
                    ))
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        return alerts

    def _predict_security_threats(self, os) -> List[PredictiveAlert]:
        alerts = []
        try:
            if not hasattr(os, 'daemon') or not os.daemon:
                return alerts
            guardian = getattr(os.daemon, 'guardian', None)
            if not guardian:
                return alerts
            recent = guardian.get_recent_events(limit=10) if hasattr(guardian, 'get_recent_events') else []
            if len(recent) > 3:
                high_threat = [e for e in recent if getattr(e, 'threat_level', 0) >= 2]
                if len(high_threat) >= 3:
                    alerts.append(PredictiveAlert(
                        category=PredictionCategory.SECURITY,
                        title="Escalating security threat pattern",
                        description=f"{len(high_threat)} high-threat events in recent history. "
                                    f"Risk of active compromise increasing.",
                        confidence=min(0.95, len(high_threat) * 0.2 + 0.3),
                        confidence_tier=ConfidenceTier.HIGH if len(high_threat) >= 5 else ConfidenceTier.MEDIUM,
                        severity=0.8,
                        evidence=[f"{len(high_threat)} high-threat events detected"],
                        recommended_action="Review security audit log. Consider engaging kill switch for affected modules.",
                        source_modules=["security_guardian"],
                    ))
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        return alerts

    def _predict_cognitive_overload(self, os) -> List[PredictiveAlert]:
        alerts = []
        try:
            if not hasattr(os, 'attention') or not os.attention:
                return alerts
            attn_stats = os.attention.get_stats()
            fatigue = attn_stats.get("fatigue_level", 0)
            cognitive_load = attn_stats.get("cognitive_load", 0)
            if fatigue > 0.7 or cognitive_load > 0.7:
                alerts.append(PredictiveAlert(
                    category=PredictionCategory.COGNITIVE,
                    title="Cognitive overload risk",
                    description=f"Fatigue={round(fatigue,2)}, cognitive_load={round(cognitive_load,2)}. "
                                f"Decision quality may degrade.",
                    confidence=min(0.9, max(fatigue, cognitive_load)),
                    confidence_tier=ConfidenceTier.HIGH if max(fatigue, cognitive_load) > 0.8 else ConfidenceTier.MEDIUM,
                    severity=max(fatigue, cognitive_load),
                    evidence=[f"fatigue={round(fatigue,3)}", f"cognitive_load={round(cognitive_load,3)}"],
                    recommended_action="Pause intensive tasks. Take a break. Reduce notification frequency.",
                    source_modules=["attention_engine"],
                ))
            if hasattr(os, 'emotional_intel') and os.emotional_intel:
                stats = os.emotional_intel.get_stats()
                negative_count = stats.get("negative_emotions", 0)
                if negative_count > 10:
                    alerts.append(PredictiveAlert(
                        category=PredictionCategory.COGNITIVE,
                        title="Emotional state degradation",
                        description=f"{negative_count} negative emotions recently detected. Risk of burnout.",
                        confidence=min(0.85, negative_count * 0.05 + 0.3),
                        confidence_tier=ConfidenceTier.MEDIUM,
                        severity=min(1.0, negative_count * 0.05),
                        evidence=[f"{negative_count} negative emotions"],
                        recommended_action="Engage in positive activities. Consider mindfulness exercises.",
                        source_modules=["emotional_intelligence"],
                    ))
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        return alerts

    def _predict_resource_exhaustion(self, os) -> List[PredictiveAlert]:
        alerts = []
        try:
            if not hasattr(os, 'opt_daemon') or not os.opt_daemon:
                return alerts
            stats = os.opt_daemon.get_stats()
            params_count = stats.get("params_registered", 0)
            if params_count > 100:
                alerts.append(PredictiveAlert(
                    category=PredictionCategory.RESOURCE,
                    title="Parameter registry growth",
                    description=f"{params_count} tunable parameters registered. May indicate configuration drift.",
                    confidence=0.5,
                    confidence_tier=ConfidenceTier.LOW,
                    severity=0.3,
                    evidence=[f"params_registered={params_count}"],
                    recommended_action="Review and prune unused parameters in optimization_daemon.",
                    source_modules=["optimization_daemon"],
                ))
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        return alerts

    def _predict_social_deterioration(self, os) -> List[PredictiveAlert]:
        alerts = []
        try:
            if not hasattr(os, 'social_intel') or not os.social_intel:
                return alerts
            stats = os.social_intel.get_stats()
            if hasattr(os, 'empathy') and os.empathy:
                emp_stats = os.empathy.get_stats()
                unhealthy = emp_stats.get("unhealthy_relationships", 0)
                if unhealthy > 2:
                    alerts.append(PredictiveAlert(
                        category=PredictionCategory.SOCIAL,
                        title="Relationship health declining",
                        description=f"{unhealthy} relationships showing signs of deterioration.",
                        confidence=min(0.8, unhealthy * 0.2 + 0.3),
                        confidence_tier=ConfidenceTier.MEDIUM if unhealthy <= 5 else ConfidenceTier.HIGH,
                        severity=min(1.0, unhealthy * 0.15),
                        evidence=[f"{unhealthy} unhealthy relationships"],
                        recommended_action="Reach out to affected contacts. Use empathy_mapping to understand their needs.",
                        source_modules=["empathy_mapping"],
                    ))
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        return alerts

    def get_alerts(self, category: Optional[str] = None,
                   min_confidence: float = 0.0, limit: int = 50,
                   unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        limit = max(1, min(200, limit))
        with self._lock:
            alerts = list(self._alerts.values())
        if category:
            alerts = [a for a in alerts if a.category.value == category]
        if min_confidence > 0:
            alerts = [a for a in alerts if a.confidence >= min_confidence]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        alerts.sort(key=lambda a: a.predicted_at, reverse=True)
        return [a.to_dict() for a in alerts[:limit]]

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        alert_id = _sanitize(alert_id, 20)
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                return copy.deepcopy(alert.to_dict())
        return None

    def acknowledge(self, alert_id: str) -> bool:
        alert_id = _sanitize(alert_id, 20)
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and not alert.acknowledged:
                alert.acknowledged = True
                with self._stats_lock:
                    self._stats["alerts_acknowledged"] += 1
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        alert_id = _sanitize(alert_id, 20)
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and not alert.resolved:
                alert.resolved = True
                alert.acknowledged = True
                with self._stats_lock:
                    self._stats["alerts_resolved"] += 1
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._alerts)
            unacknowledged = sum(1 for a in self._alerts.values() if not a.acknowledged)
        with self._stats_lock:
            stats = dict(self._stats)
        stats["total_alerts"] = total
        stats["unacknowledged_alerts"] = unacknowledged
        return stats
