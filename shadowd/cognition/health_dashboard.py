"""CognitiveHealthDashboard — Real-time module health scoring and diagnostics.

Monitors the health, performance, and interconnection of all cognitive modules.
Provides a single point of truth for system status, bottleneck detection,
and module-level diagnostics.

Architecture:
  - ModuleHealth: per-module health status with performance metrics
  - HealthAlert: triggered when a module degrades
  - SystemHealthReport: aggregated dashboard of all module health
  - CognitiveHealthDashboard: main engine with monitoring, alerts, and diagnostics
"""

import copy
import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 50) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]

_MAX_MODULES = 50
_MAX_ALERTS = 500
_MAX_HISTORY = 2000
_MAX_CHECKS_PER_MODULE = 100


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModuleHealth:
    module_name: str = ""
    status: str = ""
    latency_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_check: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.module_name:
            self.module_name = _sanitize(self.module_name, 50)
        if self.status:
            self.status = _sanitize(self.status, 30)
        self.latency_ms = max(0.0, self.latency_ms)
        self.success_rate = max(0.0, min(1.0, self.success_rate))
        self.last_check = self.last_check or datetime.now(timezone.utc).isoformat()
        if self.details:
            self.details = {str(k)[:50]: v for k, v in list(self.details.items())[:20]}

    def health_score(self) -> float:
        """Calculate 0-1 health score from metrics."""
        latency_penalty = min(self.latency_ms / 5000.0, 0.3)
        error_penalty = min(self.error_count / 100.0, 0.3)
        score = (self.success_rate * 0.5) + ((1.0 - latency_penalty) * 0.3) + ((1.0 - error_penalty) * 0.2)
        return round(max(0.0, min(1.0, score)), 4)

    def to_dict(self) -> Dict:
        return {
            "module_name": self.module_name,
            "status": self.status,
            "health_score": self.health_score(),
            "latency_ms": round(self.latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "error_count": self.error_count,
            "last_check": self.last_check,
            "details": copy.deepcopy(self.details) if self.details else {},
        }


@dataclass
class HealthAlert:
    alert_id: str = ""
    module_name: str = ""
    severity: str = ""
    message: str = ""
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0
    resolved: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = uuid.uuid4().hex[:12]
        if self.module_name:
            self.module_name = _sanitize(self.module_name, 50)
        if self.severity:
            self.severity = _sanitize(self.severity, 20)
        if self.message:
            self.message = _sanitize_content(self.message, 500)
        if self.metric:
            self.metric = _sanitize(self.metric, 50)
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "module_name": self.module_name,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "resolved": self.resolved,
            "created_at": self.created_at,
        }


@dataclass
class SystemHealthReport:
    overall_status: str = ""
    overall_score: float = 0.0
    module_count: int = 0
    healthy_count: int = 0
    degraded_count: int = 0
    critical_count: int = 0
    offline_count: int = 0
    active_alerts: int = 0
    modules: Dict[str, Dict] = field(default_factory=dict)
    generated_at: str = ""

    def __post_init__(self):
        self.generated_at = self.generated_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status,
            "overall_score": round(self.overall_score, 4),
            "module_count": self.module_count,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "critical_count": self.critical_count,
            "offline_count": self.offline_count,
            "active_alerts": self.active_alerts,
            "modules": copy.deepcopy(self.modules) if self.modules else {},
            "generated_at": self.generated_at,
        }


_ALERT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "latency_ms": {"low": 500.0, "medium": 2000.0, "high": 5000.0, "critical": 10000.0},
    "success_rate": {"low": 0.95, "medium": 0.85, "high": 0.70, "critical": 0.50},
    "error_count": {"low": 5.0, "medium": 20.0, "high": 50.0, "critical": 100.0},
}


class CognitiveHealthDashboard:
    """Real-time module health scoring and diagnostics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._modules: Dict[str, ModuleHealth] = {}
        self._alerts: Dict[str, HealthAlert] = {}
        self._alert_order: deque = deque(maxlen=_MAX_ALERTS)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._check_counts: Dict[str, int] = {}
        self._stats: Dict[str, int] = {
            "total_checks": 0,
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "reports_generated": 0,
        }

    def register_module(self, module_name: str, initial_status: str = "healthy") -> None:
        """Register a module for health monitoring."""
        module_name = _sanitize(module_name, 50)
        initial_status = _sanitize(initial_status, 30)
        with self._lock:
            self._register_module_unlocked(module_name, initial_status)

    def _register_module_unlocked(self, module_name: str, initial_status: str = "healthy") -> None:
        """Register a module without acquiring the lock (caller must hold _lock)."""
        health = ModuleHealth(
            module_name=module_name,
            status=initial_status,
            success_rate=1.0,
            error_count=0,
        )
        if len(self._modules) >= _MAX_MODULES:
            oldest = next(iter(self._modules))
            del self._modules[oldest]
            logger.warning("Evicting module %s (cap=%d)", oldest, _MAX_MODULES)
        self._modules[module_name] = health
        self._check_counts[module_name] = 0

    def update_module(self, module_name: str, latency_ms: float = 0.0,
                      success_rate: float = 1.0, error_count: int = 0,
                      status: Optional[str] = None,
                      details: Optional[Dict] = None) -> ModuleHealth:
        """Update a module's health metrics."""
        module_name = _sanitize(module_name, 50)
        latency_ms = max(0.0, latency_ms)
        success_rate = max(0.0, min(1.0, success_rate))
        error_count = max(0, error_count)

        with self._lock:
            if module_name not in self._modules:
                self._register_module_unlocked(module_name)

            health = self._modules[module_name]
            health.latency_ms = latency_ms
            health.success_rate = success_rate
            health.error_count = error_count
            health.last_check = datetime.now(timezone.utc).isoformat()
            if status:
                health.status = _sanitize(status, 30)
            if details:
                health.details = {str(k)[:50]: _sanitize(str(v), 200) if isinstance(v, str) else v for k, v in list(details.items())[:20]}

            self._check_counts[module_name] = self._check_counts.get(module_name, 0) + 1
            if self._check_counts[module_name] > _MAX_CHECKS_PER_MODULE:
                self._check_counts[module_name] = _MAX_CHECKS_PER_MODULE

        with self._stats_lock:
            self._stats["total_checks"] += 1

        self._check_thresholds(module_name, latency_ms, success_rate, error_count)

        return copy.deepcopy(health)

    def _check_thresholds(self, module_name: str, latency_ms: float,
                         success_rate: float, error_count: int) -> None:
        """Check metrics against thresholds and generate alerts."""
        for metric_name, value, thresholds in [
            ("latency_ms", latency_ms, _ALERT_THRESHOLDS["latency_ms"]),
            ("success_rate", 1.0 - success_rate, _ALERT_THRESHOLDS["success_rate"]),
            ("error_count", float(error_count), _ALERT_THRESHOLDS["error_count"]),
        ]:
            for severity in ["critical", "high", "medium", "low"]:
                threshold = thresholds[severity]
                if value >= threshold:
                    message = "%s %s=%.2f exceeds %s threshold (%.2f)" % (module_name, metric_name, value, severity, threshold)
                    self._create_alert(
                        module_name=module_name,
                        severity=severity,
                        message=message,
                        metric=metric_name,
                        value=value,
                        threshold=threshold,
                    )
                    break

    def _create_alert(self, module_name: str, severity: str, message: str,
                      metric: str = "", value: float = 0.0,
                      threshold: float = 0.0) -> HealthAlert:
        """Create a health alert."""
        alert = HealthAlert(
            module_name=module_name,
            severity=severity,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
        )
        with self._lock:
            self._alerts[alert.alert_id] = alert
            self._alert_order.append(alert.alert_id)
            if len(self._alerts) > _MAX_ALERTS:
                overflow = len(self._alerts) - _MAX_ALERTS
                for aid in list(self._alert_order)[:overflow]:
                    self._alerts.pop(aid, None)
                    logger.warning("Evicting alert %s (cap=%d)", aid, _MAX_ALERTS)

        with self._stats_lock:
            self._stats["alerts_triggered"] += 1

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        alert_id = _sanitize(alert_id, 20)
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False
            alert.resolved = True

        with self._stats_lock:
            self._stats["alerts_resolved"] += 1
        return True

    def get_alerts(self, severity: Optional[str] = None,
                   module_name: Optional[str] = None,
                   resolved: Optional[bool] = None,
                   limit: int = 50) -> List[HealthAlert]:
        """Get alerts, optionally filtered."""
        limit = max(1, min(limit, _MAX_ALERTS))
        with self._lock:
            alerts = list(self._alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if module_name:
            alerts = [a for a in alerts if a.module_name == module_name]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        alerts = sorted(alerts, key=lambda a: a.created_at, reverse=True)
        return copy.deepcopy(alerts[:limit])

    def get_module_health(self, module_name: str) -> Optional[ModuleHealth]:
        """Get health for a specific module."""
        module_name = _sanitize(module_name, 50)
        with self._lock:
            health = self._modules.get(module_name)
        return copy.deepcopy(health) if health else None

    def get_all_modules(self) -> List[ModuleHealth]:
        """Get health for all registered modules."""
        with self._lock:
            modules = list(self._modules.values())
        return [copy.deepcopy(m) for m in modules]

    def generate_report(self) -> SystemHealthReport:
        """Generate a full system health report."""
        with self._lock:
            modules = {name: copy.deepcopy(m.to_dict()) for name, m in self._modules.items()}
            active_alerts = sum(1 for a in self._alerts.values() if not a.resolved)
            scores = [m.health_score() for m in self._modules.values()]

        overall_score = sum(scores) / len(scores) if scores else 1.0

        healthy = sum(1 for m in modules.values() if m.get("status") == "healthy")
        degraded = sum(1 for m in modules.values() if m.get("status") == "degraded")
        critical = sum(1 for m in modules.values() if m.get("status") == "critical")
        offline = sum(1 for m in modules.values() if m.get("status") == "offline")

        if critical > 0:
            overall_status = "critical"
        elif degraded > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        report = SystemHealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            module_count=len(modules),
            healthy_count=healthy,
            degraded_count=degraded,
            critical_count=critical,
            offline_count=offline,
            active_alerts=active_alerts,
            modules=modules,
        )

        with self._lock:
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_score": overall_score,
                "overall_status": overall_status,
            })

        with self._stats_lock:
            self._stats["reports_generated"] += 1

        return report

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get health check history."""
        limit = max(1, min(limit, _MAX_HISTORY))
        with self._lock:
            return [copy.deepcopy(h) for h in list(self._history)[-limit:]]

    def get_stats(self) -> Dict:
        """Get dashboard statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            stats["total_modules"] = len(self._modules)
            stats["total_alerts"] = len(self._alerts)
            stats["active_alerts"] = sum(1 for a in self._alerts.values() if not a.resolved)
        return stats

    def auto_update(self, modules_map: Dict[str, Any]) -> Dict:
        """Pull live stats from all module get_stats() methods and update health.

        Args:
            modules_map: dict of {module_name: module_object} where each object
                         has a get_stats() method returning a dict.

        Returns:
            Summary dict with updated count and alert count.
        """
        updated = 0
        with self._stats_lock: alerts_before = self._stats["alerts_triggered"]

        for name, mod in modules_map.items():
            safe_name = _sanitize(name, 50)
            try:
                stats = mod.get_stats()
                if not isinstance(stats, dict):
                    continue
                latency = stats.get("avg_latency_ms", stats.get("latency_ms", 0.0))
                if isinstance(latency, (int, float)):
                    latency = float(latency)
                else:
                    latency = 0.0

                error_count = stats.get("error_count", stats.get("errors", 0))
                if isinstance(error_count, (int, float)):
                    error_count = int(error_count)
                else:
                    error_count = 0

                success_rate = 1.0
                total_req = stats.get("total_requests", stats.get("total_decisions",
                             stats.get("total_checks", stats.get("total_messages", 0))))
                if isinstance(total_req, (int, float)) and total_req > 0:
                    succ = stats.get("successful_requests", stats.get("successes",
                           stats.get("decisions_made", 0)))
                    if isinstance(succ, (int, float)):
                        success_rate = min(1.0, max(0.0, succ / total_req))

                self.update_module(
                    module_name=safe_name,
                    latency_ms=latency,
                    success_rate=success_rate,
                    error_count=error_count,
                    details=stats,
                )
                updated += 1
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
                logger.debug("auto_update %s failed: %s", safe_name, type(e).__name__)
                with self._lock:
                    if safe_name not in self._modules:
                        self._register_module_unlocked(safe_name, "offline")
                        self._modules[safe_name].error_count = 1
                    else:
                        self._modules[safe_name].status = "offline"
                        self._modules[safe_name].error_count += 1

        with self._stats_lock: new_alerts = self._stats["alerts_triggered"] - alerts_before
        return {
            "modules_updated": updated,
            "modules_total": len(modules_map),
            "new_alerts": new_alerts,
        }