#!/usr/bin/env python3
"""
PhoenixRegenerator — Self-Healing Evolution.

System detects degraded components, diagnoses root causes,
regenerates/replaces them while preserving learning.

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import hashlib
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.phoenix_regenerator")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_COMPONENTS = 200
_MAX_DIAGNOSES = 1000
_MAX_PLANS = 500
_MAX_RESULTS = 500
_MAX_ERRORS_PER_COMPONENT = 100
_MAX_SYMPTOMS = 10
_MAX_STEPS = 10
_MAX_METADATA_KEYS = 20


class HealthStatus(str, Enum):
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    REGENERATING = "regenerating"


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0, decimals: int = 4) -> float:
    return round(max(lo, min(hi, value)), decimals)


@dataclass
class ComponentHealth:
    component_id: str
    component_name: str
    health: HealthStatus
    health_score: float
    error_count: int
    last_error: str
    last_check: str
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.component_id:
            self.component_id = uuid.uuid4().hex[:12]
        self.component_id = _sanitize(self.component_id, 50)
        self.component_name = _sanitize(self.component_name, 200)
        self.health_score = _clamp(self.health_score)
        self.error_count = max(0, self.error_count)
        self.last_error = _sanitize_content(self.last_error, 1000) if self.last_error else ""
        self.last_check = _sanitize_content(self.last_check, 200) if self.last_check else ""
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "health": self.health.value,
            "health_score": round(self.health_score, 4),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_check": self.last_check,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class Diagnosis:
    diagnosis_id: str
    component_id: str
    root_cause: str
    severity: float
    symptoms: List[str]
    recommended_action: str
    confidence: float
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.diagnosis_id:
            self.diagnosis_id = uuid.uuid4().hex[:12]
        self.diagnosis_id = _sanitize(self.diagnosis_id, 50)
        self.component_id = _sanitize(self.component_id, 50)
        self.root_cause = _sanitize_content(self.root_cause, 500)
        self.severity = _clamp(self.severity)
        self.symptoms = [_sanitize_content(s, 200) for s in (self.symptoms or [])[:_MAX_SYMPTOMS]]
        self.recommended_action = _sanitize_content(self.recommended_action, 500)
        self.confidence = _clamp(self.confidence)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "diagnosis_id": self.diagnosis_id,
            "component_id": self.component_id,
            "root_cause": self.root_cause,
            "severity": round(self.severity, 4),
            "symptoms": list(self.symptoms),
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class RegenerationPlan:
    plan_id: str
    component_id: str
    current_state: HealthStatus
    target_state: HealthStatus
    steps: List[str]
    learning_preservation: float
    estimated_time: str
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = uuid.uuid4().hex[:12]
        self.plan_id = _sanitize(self.plan_id, 50)
        self.component_id = _sanitize(self.component_id, 50)
        self.current_state = self.current_state if isinstance(self.current_state, HealthStatus) else self._safe_health_status(self.current_state)
        self.target_state = self.target_state if isinstance(self.target_state, HealthStatus) else self._safe_health_status(self.target_state)
        self.steps = [_sanitize_content(s, 300) for s in (self.steps or [])[:_MAX_STEPS]]
        self.learning_preservation = _clamp(self.learning_preservation)
        self.estimated_time = _sanitize(self.estimated_time, 100)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    @staticmethod
    def _safe_health_status(value):
        try:
            return HealthStatus(value)
        except (ValueError, TypeError):
            return HealthStatus.DEGRADED

    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "component_id": self.component_id,
            "current_state": self.current_state.value,
            "target_state": self.target_state.value,
            "steps": list(self.steps),
            "learning_preservation": round(self.learning_preservation, 4),
            "estimated_time": self.estimated_time,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class RegenerationResult:
    result_id: str
    plan_id: str
    component_id: str
    success: bool
    health_before: float
    health_after: float
    learning_preserved: float
    steps_completed: int
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.result_id:
            self.result_id = uuid.uuid4().hex[:12]
        self.result_id = _sanitize(self.result_id, 50)
        self.plan_id = _sanitize(self.plan_id, 50)
        self.component_id = _sanitize(self.component_id, 50)
        self.health_before = _clamp(self.health_before)
        self.health_after = _clamp(self.health_after)
        self.learning_preserved = _clamp(self.learning_preserved)
        self.steps_completed = max(0, self.steps_completed)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "plan_id": self.plan_id,
            "component_id": self.component_id,
            "success": self.success,
            "health_before": round(self.health_before, 4),
            "health_after": round(self.health_after, 4),
            "learning_preserved": round(self.learning_preserved, 4),
            "steps_completed": self.steps_completed,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


class PhoenixRegenerator:
    """Self-healing regeneration engine.

    Detects degraded components, diagnoses root causes, and regenerates
    them while preserving learned parameters.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._components: Dict[str, ComponentHealth] = {}
        self._component_order: deque = deque(maxlen=_MAX_COMPONENTS)
        self._diagnoses: Dict[str, Diagnosis] = {}
        self._diagnosis_order: deque = deque(maxlen=_MAX_DIAGNOSES)
        self._plans: Dict[str, RegenerationPlan] = {}
        self._plan_order: deque = deque(maxlen=_MAX_PLANS)
        self._results: Dict[str, RegenerationResult] = {}
        self._result_order: deque = deque(maxlen=_MAX_RESULTS)
        self._stats = {
            "components_registered": 0,
            "errors_reported": 0,
            "diagnoses_created": 0,
            "plans_created": 0,
            "regenerations_executed": 0,
            "regenerations_succeeded": 0,
            "regenerations_failed": 0,
            "learning_preserved_avg": 0.0,
        }

    def register_component(self, component_id: str, component_name: str,
                           initial_health: float = 1.0) -> ComponentHealth:
        cid = _sanitize(component_id, 50)
        cname = _sanitize(component_name, 200)
        score = _clamp(initial_health)
        if score >= 0.8:
            status = HealthStatus.OPTIMAL
        elif score >= 0.5:
            status = HealthStatus.DEGRADED
        elif score >= 0.2:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.FAILED
        component = ComponentHealth(
            component_id=cid,
            component_name=cname,
            health=status,
            health_score=score,
            error_count=0,
            last_error="",
            last_check=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )
        with self._lock:
            if cid in self._components:
                return copy.deepcopy(self._components[cid])
            if len(self._components) >= _MAX_COMPONENTS:
                evicted_id = self._component_order.popleft()
                self._components.pop(evicted_id, None)
                logger.warning("Evicted component %s", evicted_id)
            self._components[cid] = component
            self._component_order.append(cid)
            with self._stats_lock:
                self._stats["components_registered"] += 1
        return copy.deepcopy(component)

    def check_health(self, component_id: str) -> Optional[ComponentHealth]:
        cid = _sanitize(component_id, 50)
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return None
            comp.last_check = datetime.now(timezone.utc).isoformat()
            return copy.deepcopy(comp)

    def report_error(self, component_id: str, error_message: str) -> Optional[Diagnosis]:
        cid = _sanitize(component_id, 50)
        emsg = _sanitize_content(error_message, 1000)
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return None
            comp.error_count += 1
            comp.last_error = emsg
            if comp.error_count > _MAX_ERRORS_PER_COMPONENT:
                comp.error_count = _MAX_ERRORS_PER_COMPONENT
            decay = min(0.05 * comp.error_count, 0.5)
            comp.health_score = _clamp(comp.health_score - decay)
            if comp.health_score >= 0.8:
                comp.health = HealthStatus.OPTIMAL
            elif comp.health_score >= 0.5:
                comp.health = HealthStatus.DEGRADED
            elif comp.health_score >= 0.2:
                comp.health = HealthStatus.CRITICAL
            else:
                comp.health = HealthStatus.FAILED
            root_cause = "unknown_error"
            if re.search(r'\btimeout\b|\btimed out\b', emsg, re.IGNORECASE):
                root_cause = "timeout"
            elif re.search(r'\bmemory\b|\boom\b', emsg, re.IGNORECASE):
                root_cause = "memory_pressure"
            elif re.search(r'\bconnection\b|\bnetwork\b', emsg, re.IGNORECASE):
                root_cause = "connectivity_failure"
            elif re.search(r'\bpermission\b|\bauth\b', emsg, re.IGNORECASE):
                root_cause = "permission_denied"
            elif re.search(r'\bcorrupt\b|\binvalid\b', emsg, re.IGNORECASE):
                root_cause = "data_corruption"
            elif re.search(r'\bcrash\b|\bfatal\b', emsg, re.IGNORECASE):
                root_cause = "critical_failure"
            severity = _clamp(comp.error_count * 0.05)
            action = "monitor"
            if severity >= 0.7:
                action = "regenerate"
            elif severity >= 0.4:
                action = "restart"
            elif severity >= 0.2:
                action = "investigate"
            diagnosis = Diagnosis(
                diagnosis_id=uuid.uuid4().hex[:12],
                component_id=cid,
                root_cause=root_cause,
                severity=severity,
                symptoms=[emsg[:200]],
                recommended_action=action,
                confidence=_clamp(1.0 - severity * 0.3),
                metadata={"error_count": comp.error_count},
            )
            if len(self._diagnoses) >= _MAX_DIAGNOSES:
                evicted_id = self._diagnosis_order.popleft()
                self._diagnoses.pop(evicted_id, None)
                logger.warning("Evicted diagnosis %s", evicted_id)
            self._diagnoses[diagnosis.diagnosis_id] = diagnosis
            self._diagnosis_order.append(diagnosis.diagnosis_id)
            with self._stats_lock:
                self._stats["errors_reported"] += 1
                self._stats["diagnoses_created"] += 1
        return copy.deepcopy(diagnosis)

    def diagnose(self, component_id: str, symptoms: Optional[List[str]] = None) -> Optional[Diagnosis]:
        cid = _sanitize(component_id, 50)
        clean_symptoms = [_sanitize_content(s, 200) for s in (symptoms or [])[:_MAX_SYMPTOMS]]
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return None
            root_cause = "unknown"
            confidence = 0.3
            all_symptoms = list(clean_symptoms)
            if comp.last_error:
                all_symptoms.insert(0, comp.last_error[:200])
            all_symptoms = all_symptoms[:_MAX_SYMPTOMS]
            symptom_text = " ".join(all_symptoms).lower()
            if any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["timeout", "timed out", "latency", "slow"]):
                root_cause = "timeout_degradation"
                confidence = 0.75
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["memory", "oom", "heap", "allocation"]):
                root_cause = "memory_pressure"
                confidence = 0.8
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["connection", "network", "unreachable", "dns"]):
                root_cause = "connectivity_failure"
                confidence = 0.7
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["corrupt", "invalid", "malformed", "parse"]):
                root_cause = "data_corruption"
                confidence = 0.65
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["crash", "fatal", "panic", "segfault"]):
                root_cause = "critical_failure"
                confidence = 0.9
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["permission", "auth", "forbidden", "unauthorized"]):
                root_cause = "permission_denied"
                confidence = 0.7
            elif any(re.search(r'\b' + re.escape(w) + r'\b', symptom_text) for w in ["conflict", "race", "deadlock", "concurrent"]):
                root_cause = "concurrency_issue"
                confidence = 0.6
            severity = _clamp(comp.error_count * 0.04 + _clamp(1.0 - comp.health_score) * 0.5)
            recommended = "investigate"
            if root_cause == "critical_failure":
                recommended = "regenerate"
            elif root_cause == "data_corruption":
                recommended = "restore_from_backup"
            elif root_cause == "memory_pressure":
                recommended = "restart_with_resource_limit"
            elif root_cause == "connectivity_failure":
                recommended = "failover_and_retry"
            elif root_cause == "timeout_degradation":
                recommended = "restart_with_timeout_adjustment"
            elif root_cause == "concurrency_issue":
                recommended = "rebuild_with_locking"
            elif root_cause == "permission_denied":
                recommended = "reauthorize"
            diagnosis = Diagnosis(
                diagnosis_id=uuid.uuid4().hex[:12],
                component_id=cid,
                root_cause=root_cause,
                severity=severity,
                symptoms=all_symptoms,
                recommended_action=recommended,
                confidence=confidence,
                metadata={"health_score": comp.health_score, "error_count": comp.error_count},
            )
            if len(self._diagnoses) >= _MAX_DIAGNOSES:
                evicted_id = self._diagnosis_order.popleft()
                self._diagnoses.pop(evicted_id, None)
                logger.warning("Evicted diagnosis %s", evicted_id)
            self._diagnoses[diagnosis.diagnosis_id] = diagnosis
            self._diagnosis_order.append(diagnosis.diagnosis_id)
            with self._stats_lock:
                self._stats["diagnoses_created"] += 1
        return copy.deepcopy(diagnosis)

    def create_regeneration_plan(self, component_id: str,
                                  target_state: Optional[HealthStatus] = None) -> Optional[RegenerationPlan]:
        cid = _sanitize(component_id, 50)
        if target_state is None:
            target_state = HealthStatus.OPTIMAL
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return None
            current = comp.health
            learning = _clamp(0.5 + comp.health_score * 0.4)
            steps = self._generate_steps(current, target_state)
            est_time = "immediate"
            if current == HealthStatus.FAILED:
                est_time = "5-10 minutes"
            elif current == HealthStatus.CRITICAL:
                est_time = "3-5 minutes"
            elif current == HealthStatus.DEGRADED:
                est_time = "1-3 minutes"
            plan = RegenerationPlan(
                plan_id=uuid.uuid4().hex[:12],
                component_id=cid,
                current_state=current,
                target_state=target_state,
                steps=steps,
                learning_preservation=learning,
                estimated_time=est_time,
                metadata={
                    "health_score_at_plan": comp.health_score,
                    "error_count_at_plan": comp.error_count,
                },
            )
            if len(self._plans) >= _MAX_PLANS:
                evicted_id = self._plan_order.popleft()
                self._plans.pop(evicted_id, None)
                logger.warning("Evicted regeneration plan %s", evicted_id)
            self._plans[plan.plan_id] = plan
            self._plan_order.append(plan.plan_id)
            with self._stats_lock:
                self._stats["plans_created"] += 1
        return copy.deepcopy(plan)

    def _generate_steps(self, current: HealthStatus,
                        target: HealthStatus) -> List[str]:
        steps = []
        steps.append("Assess current component state")
        if current in (HealthStatus.FAILED, HealthStatus.CRITICAL):
            steps.append("Extract and preserve learned parameters")
            steps.append("Isolate failing component")
            steps.append("Initialize fresh component instance")
            steps.append("Restore preserved learning data")
        elif current == HealthStatus.DEGRADED:
            steps.append("Capture current operating parameters")
            steps.append("Apply targeted repair patches")
            steps.append("Restore modified parameters")
        elif current == HealthStatus.REGENERATING:
            steps.append("Verify regeneration progress")
            steps.append("Resume preserved state")
        else:
            steps.append("Verify component health")
        steps.append("Run validation checks")
        steps.append("Confirm target state achieved")
        return steps[:_MAX_STEPS]

    def execute_regeneration(self, plan_id: str) -> Optional[RegenerationResult]:
        pid = _sanitize(plan_id, 50)
        with self._lock:
            plan = self._plans.get(pid)
            if plan is None:
                return None
            comp = self._components.get(plan.component_id)
            if comp is None:
                return None
            health_before = comp.health_score
            learning_preserved = plan.learning_preservation
            num_steps = len(plan.steps)
            if comp.health in (HealthStatus.FAILED, HealthStatus.CRITICAL):
                target_score = 0.85
            elif comp.health == HealthStatus.DEGRADED:
                target_score = 0.90
            else:
                target_score = 0.95
            noise = (int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % 100) / 1000.0
            target_score = _clamp(target_score - noise + learning_preserved * 0.05)
            comp.health_score = target_score
            if target_score >= 0.8:
                comp.health = plan.target_state
            elif target_score >= 0.5:
                comp.health = HealthStatus.DEGRADED
            else:
                comp.health = HealthStatus.CRITICAL
            comp.last_check = datetime.now(timezone.utc).isoformat()
            success = target_score >= 0.5
            if not success:
                learning_preserved = learning_preserved * 0.5
            result = RegenerationResult(
                result_id=uuid.uuid4().hex[:12],
                plan_id=pid,
                component_id=plan.component_id,
                success=success,
                health_before=health_before,
                health_after=comp.health_score,
                learning_preserved=_clamp(learning_preserved),
                steps_completed=num_steps if success else num_steps // 2,
                metadata={
                    "target_state": plan.target_state.value,
                    "actual_state": comp.health.value,
                },
            )
            if len(self._results) >= _MAX_RESULTS:
                evicted_id = self._result_order.popleft()
                self._results.pop(evicted_id, None)
                logger.warning("Evicted regeneration result %s", evicted_id)
            self._results[result.result_id] = result
            self._result_order.append(result.result_id)
            with self._stats_lock:
                self._stats["regenerations_executed"] += 1
                if success:
                    self._stats["regenerations_succeeded"] += 1
                else:
                    self._stats["regenerations_failed"] += 1
                total = self._stats["regenerations_succeeded"] + self._stats["regenerations_failed"]
                if total > 0:
                    preserved_sum = (
                        self._stats["learning_preserved_avg"] * (total - 1)
                        + learning_preserved
                    )
                    self._stats["learning_preserved_avg"] = round(preserved_sum / total, 4)
        return copy.deepcopy(result)

    def preserve_learning(self, component_id: str) -> Dict[str, Any]:
        cid = _sanitize(component_id, 50)
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return {}
            learning = {
                "component_id": comp.component_id,
                "component_name": comp.component_name,
                "health_score": comp.health_score,
                "error_count": comp.error_count,
                "metadata": copy.deepcopy(comp.metadata),
            }
            preserved = {}
            for key in ("model_weights", "learned_parameters", "optimization_history",
                         "adaptation_scores", "behavior_patterns"):
                if key in comp.metadata:
                    preserved[key] = copy.deepcopy(comp.metadata[key])
            preserved["health_at_preservation"] = comp.health_score
            preserved["preservation_timestamp"] = datetime.now(timezone.utc).isoformat()
            return preserved

    def restore_learning(self, component_id: str,
                         learning_data: Dict[str, Any]) -> bool:
        cid = _sanitize(component_id, 50)
        if not isinstance(learning_data, dict):
            return False
        with self._lock:
            comp = self._components.get(cid)
            if comp is None:
                return False
            for key, value in list(learning_data.items())[:_MAX_METADATA_KEYS]:
                clean_key = _sanitize(key, 100)
                if clean_key in ("component_id", "component_name",
                                  "health_at_preservation", "preservation_timestamp"):
                    continue
                comp.metadata[clean_key] = copy.deepcopy(value)
            if len(comp.metadata) > _MAX_METADATA_KEYS:
                items = list(comp.metadata.items())[:_MAX_METADATA_KEYS]
                comp.metadata = dict(items)
            return True

    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        cid = _sanitize(component_id, 50)
        with self._lock:
            comp = self._components.get(cid)
            return copy.deepcopy(comp) if comp else None

    def list_components(self, limit: int = 100, offset: int = 0) -> List[ComponentHealth]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            cids = list(self._component_order)[offset:offset + limit]
            return [copy.deepcopy(self._components[cid]) for cid in cids if cid in self._components]

    def list_diagnoses(self, component_id: str = "", limit: int = 100) -> List[Diagnosis]:
        limit = max(1, min(500, limit))
        cid = _sanitize(component_id, 50) if component_id else ""
        with self._lock:
            if cid:
                results = [d for d in self._diagnoses.values() if d.component_id == cid]
                results.sort(key=lambda d: d.created_at, reverse=True)
                return [copy.deepcopy(d) for d in results[:limit]]
            dids = list(self._diagnosis_order)[:limit]
            return [copy.deepcopy(self._diagnoses[did]) for did in dids if did in self._diagnoses]

    def list_regeneration_plans(self, component_id: str = "",
                                limit: int = 100) -> List[RegenerationPlan]:
        limit = max(1, min(500, limit))
        cid = _sanitize(component_id, 50) if component_id else ""
        with self._lock:
            if cid:
                results = [p for p in self._plans.values() if p.component_id == cid]
                results.sort(key=lambda p: p.created_at, reverse=True)
                return [copy.deepcopy(p) for p in results[:limit]]
            pids = list(self._plan_order)[:limit]
            return [copy.deepcopy(self._plans[pid]) for pid in pids if pid in self._plans]

    def list_regeneration_results(self, component_id: str = "",
                                  limit: int = 100) -> List[RegenerationResult]:
        limit = max(1, min(500, limit))
        cid = _sanitize(component_id, 50) if component_id else ""
        with self._lock:
            if cid:
                results = [r for r in self._results.values() if r.component_id == cid]
                results.sort(key=lambda r: r.created_at, reverse=True)
                return [copy.deepcopy(r) for r in results[:limit]]
            rids = list(self._result_order)[:limit]
            return [copy.deepcopy(self._results[rid]) for rid in rids if rid in self._results]

    def get_stats(self) -> Dict:
        with self._lock:
            with self._stats_lock:
                stats = dict(self._stats)
            stats["component_count"] = len(self._components)
            stats["diagnosis_count"] = len(self._diagnoses)
            stats["plan_count"] = len(self._plans)
            stats["result_count"] = len(self._results)
            return stats