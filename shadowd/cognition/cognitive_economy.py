#!/usr/bin/env python3
"""
CognitiveEconomy -- Resource Allocation and Opportunity Cost Engine.

Models cognitive resource allocation as an economy with limited budgets.
Each cognitive activity has a cost in attention, memory, computation, and time.
The engine optimizes allocation to maximize overall cognitive utility, tracks
opportunity costs, and prevents resource overcommitment.

Key concepts:
  - CognitiveResource: a budget pool (attention, memory, computation, time, emotional, creative)
  - AllocationRequest: a request to spend from a resource pool
  - OpportunityCost: the cost of choosing one activity over another
  - EconomyReport: aggregate report on resource utilization and efficiency

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.cognitive_economy")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_RESOURCES = 6
_MAX_ALLOCATIONS = 500
_MAX_OPPORTUNITY_COSTS = 500
_MAX_HISTORY = 1000


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_strict(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ResourceType(str, Enum):
    ATTENTION = "attention"
    MEMORY = "memory"
    COMPUTATION = "computation"
    TIME = "time"
    EMOTIONAL = "emotional"
    CREATIVE = "creative"


class AllocationStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERBUDGET = "overbudget"


class Priority(IntEnum):
    BACKGROUND = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CognitiveResource:
    resource_id: str = ""
    resource_type: str = ""
    total_budget: float = 1.0
    allocated: float = 0.0
    spent: float = 0.0

    def __post_init__(self):
        if not self.resource_id:
            self.resource_id = "cr_%s" % uuid.uuid4().hex[:12]
        self.resource_id = _sanitize_strict(self.resource_id, 64)
        if isinstance(self.resource_type, ResourceType):
            self.resource_type = self.resource_type.value
        self.resource_type = _sanitize_strict(self.resource_type, 50)
        self.total_budget = _clamp(self.total_budget, 0.0, 1.0)
        self.allocated = _clamp(self.allocated, 0.0, self.total_budget)
        self.spent = _clamp(self.spent, 0.0, self.allocated)

    @property
    def available(self) -> float:
        return max(0.0, self.total_budget - self.allocated)

    def to_dict(self) -> Dict:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "total_budget": round(self.total_budget, 4),
            "allocated": round(self.allocated, 4),
            "spent": round(self.spent, 4),
            "available": round(self.available, 4),
        }


@dataclass
class AllocationRequest:
    request_id: str = ""
    allocation_id: str = ""
    activity: str = ""
    resource_type: str = ""
    amount: float = 0.0
    priority: int = 2
    justification: str = ""
    status: str = "proposed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.request_id:
            self.request_id = "ar_%s" % uuid.uuid4().hex[:12]
        self.request_id = _sanitize_strict(self.request_id, 64)
        if not self.allocation_id:
            self.allocation_id = self.request_id
        self.allocation_id = _sanitize_strict(self.allocation_id, 64)
        self.activity = _sanitize_content(self.activity, 500)
        if isinstance(self.resource_type, ResourceType):
            self.resource_type = self.resource_type.value
        self.resource_type = _sanitize_strict(self.resource_type, 50)
        self.amount = _clamp(self.amount, 0.0, 1.0)
        if isinstance(self.priority, Priority):
            self.priority = int(self.priority)
        else:
            self.priority = max(0, min(4, int(self.priority)))
        self.justification = _sanitize_content(self.justification, 2000)
        if isinstance(self.status, AllocationStatus):
            self.status = self.status.value
        try:
            AllocationStatus(self.status)
        except ValueError:
            self.status = AllocationStatus.PROPOSED.value

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "allocation_id": self.allocation_id,
            "activity": self.activity,
            "resource_type": self.resource_type,
            "amount": round(self.amount, 4),
            "priority": self.priority,
            "justification": self.justification,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class OpportunityCost:
    cost_id: str = ""
    chosen_activity: str = ""
    foregone_activity: str = ""
    resource_type: str = ""
    cost_value: float = 0.0
    impact_estimate: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.cost_id:
            self.cost_id = "oc_%s" % uuid.uuid4().hex[:12]
        self.cost_id = _sanitize_strict(self.cost_id, 64)
        self.chosen_activity = _sanitize_content(self.chosen_activity, 500)
        self.foregone_activity = _sanitize_content(self.foregone_activity, 500)
        if isinstance(self.resource_type, ResourceType):
            self.resource_type = self.resource_type.value
        self.resource_type = _sanitize_strict(self.resource_type, 50)
        self.cost_value = _clamp(self.cost_value, 0.0, 1.0)
        self.impact_estimate = _clamp(self.impact_estimate, 0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            "cost_id": self.cost_id,
            "chosen_activity": self.chosen_activity,
            "foregone_activity": self.foregone_activity,
            "resource_type": self.resource_type,
            "cost_value": round(self.cost_value, 4),
            "impact_estimate": round(self.impact_estimate, 4),
            "created_at": self.created_at,
        }


@dataclass
class EconomyReport:
    report_id: str = ""
    total_budget: float = 0.0
    total_spent: float = 0.0
    efficiency: float = 0.0
    opportunity_costs: List[Dict] = field(default_factory=list)
    top_allocations: List[Dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.report_id:
            self.report_id = "er_%s" % uuid.uuid4().hex[:12]
        self.report_id = _sanitize_strict(self.report_id, 64)
        self.total_budget = _clamp(self.total_budget, 0.0, 1.0)
        self.total_spent = _clamp(self.total_spent, 0.0, 1.0)
        self.efficiency = _clamp(self.efficiency, 0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "total_budget": round(self.total_budget, 4),
            "total_spent": round(self.total_spent, 4),
            "efficiency": round(self.efficiency, 4),
            "opportunity_costs": [oc if isinstance(oc, dict) else oc.to_dict() for oc in self.opportunity_costs],
            "top_allocations": [a if isinstance(a, dict) else a.to_dict() for a in self.top_allocations],
            "created_at": self.created_at,
        }


class CognitiveEconomy:
    """Resource Allocation and Opportunity Cost Engine.

    Models cognitive resource allocation as an economy with limited budgets.
    Each cognitive activity has a cost in attention, memory, computation,
    emotional, creative, and time. The engine optimizes allocation to maximize
    overall cognitive utility, tracks opportunity costs, and prevents resource
    overcommitment.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._resources: Dict[str, CognitiveResource] = {}
        self._allocations: Dict[str, AllocationRequest] = {}
        self._allocation_order: deque = deque(maxlen=_MAX_ALLOCATIONS)
        self._opportunity_costs: Dict[str, OpportunityCost] = {}
        self._cost_order: deque = deque(maxlen=_MAX_OPPORTUNITY_COSTS)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._stats = {
            "budgets_set": 0,
            "allocations_requested": 0,
            "allocations_approved": 0,
            "allocations_spent": 0,
            "allocations_released": 0,
            "opportunity_costs_calculated": 0,
            "optimizations_run": 0,
            "reports_generated": 0,
        }

    def set_budget(self, resource_type: str, budget: float) -> CognitiveResource:
        """Set budget for a resource type."""
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        resource_type = _sanitize_strict(resource_type, 50)
        budget = _clamp(budget, 0.0, 100.0)

        with self._lock:
            if resource_type in self._resources:
                existing = self._resources[resource_type]
                existing.total_budget = budget
                existing.allocated = _clamp(existing.allocated, 0.0, budget)
                existing.spent = _clamp(existing.spent, 0.0, existing.allocated)
                result = copy.deepcopy(existing)
            else:
                if len(self._resources) >= _MAX_RESOURCES:
                    logger.warning("Cannot add resource %s (cap=%d)", resource_type, _MAX_RESOURCES)
                    return None
                resource = CognitiveResource(
                    resource_type=resource_type,
                    total_budget=budget,
                )
                self._resources[resource_type] = resource
                result = copy.deepcopy(resource)

            with self._stats_lock:
                self._stats["budgets_set"] += 1

        return result

    def request_allocation(self, activity: str, resource_type: str,
                           amount: float, priority: int = 2,
                           justification: str = "") -> AllocationRequest:
        """Request resource allocation."""
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        resource_type = _sanitize_strict(resource_type, 50)
        activity = _sanitize_content(activity, 500)
        amount = _clamp(amount, 0.0, 1.0)
        justification = _sanitize_content(justification, 2000)
        if isinstance(priority, Priority):
            priority = int(priority)
        else:
            priority = max(0, min(4, int(priority)))

        request = AllocationRequest(
            activity=activity,
            resource_type=resource_type,
            amount=amount,
            priority=priority,
            justification=justification,
            status=AllocationStatus.PROPOSED.value,
        )

        with self._lock:
            if resource_type not in self._resources:
                self._resources[resource_type] = CognitiveResource(
                    resource_type=resource_type,
                    total_budget=1.0,
                )

            resource = self._resources[resource_type]
            if amount > resource.available:
                request.status = AllocationStatus.OVERBUDGET.value

            if len(self._allocations) >= _MAX_ALLOCATIONS:
                evict_id = self._allocation_order.popleft()
                evicted = self._allocations.pop(evict_id, None)
                if evicted:
                    self._release_allocation_unlocked(evicted)
                logger.warning("Evicting allocation %s (cap=%d)", evict_id, _MAX_ALLOCATIONS)

            self._allocations[request.request_id] = request
            self._allocation_order.append(request.request_id)

            with self._stats_lock:
                self._stats["allocations_requested"] += 1

        return copy.deepcopy(request)

    def _find_request(self, request_id: str) -> Optional[AllocationRequest]:
        request_id = _sanitize_strict(request_id, 64)
        req = self._allocations.get(request_id)
        if req is not None:
            return req
        for r in self._allocations.values():
            if r.allocation_id == request_id:
                return r
        return None

    def approve_allocation(self, request_id: str) -> Optional[AllocationRequest]:
        """Approve a pending request."""
        with self._lock:
            request = self._find_request(request_id)
            if not request:
                return None

            if request.status != AllocationStatus.PROPOSED.value:
                return copy.deepcopy(request)

            resource = self._resources.get(request.resource_type)
            if not resource:
                return copy.deepcopy(request)

            if request.amount > resource.available:
                request.status = AllocationStatus.OVERBUDGET.value
                with self._stats_lock:
                    self._stats["allocations_approved"] += 1
                return copy.deepcopy(request)

            resource.allocated = round(resource.allocated + request.amount, 6)
            resource.allocated = _clamp(resource.allocated, 0.0, resource.total_budget)
            request.status = AllocationStatus.APPROVED.value

            with self._stats_lock:
                self._stats["allocations_approved"] += 1

        return copy.deepcopy(request)

    def spend_allocation(self, request_id: str, amount: float) -> Optional[AllocationRequest]:
        """Spend from an allocation."""
        amount = _clamp(amount, 0.0, 1.0)

        with self._lock:
            request = self._find_request(request_id)
            if not request:
                return None

            if request.status not in (AllocationStatus.APPROVED.value, AllocationStatus.ACTIVE.value):
                return copy.deepcopy(request)

            resource = self._resources.get(request.resource_type)
            if not resource:
                return copy.deepcopy(request)

            spend_amount = min(amount, resource.allocated - resource.spent)
            resource.spent = round(resource.spent + spend_amount, 6)
            resource.spent = _clamp(resource.spent, 0.0, resource.allocated)
            request.status = AllocationStatus.ACTIVE.value

            if resource.spent >= resource.allocated:
                request.status = AllocationStatus.COMPLETED.value

            with self._stats_lock:
                self._stats["allocations_spent"] += 1

            self._history.append({
                "type": "spend",
                "request_id": request_id,
                "amount": round(spend_amount, 4),
                "resource_type": request.resource_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return copy.deepcopy(request)

    def release_allocation(self, request_id: str) -> Optional[AllocationRequest]:
        """Release unused allocation back to the pool."""

        with self._lock:
            request = self._find_request(request_id)
            if not request:
                return None

            if request.status in (AllocationStatus.COMPLETED.value,
                                  AllocationStatus.CANCELLED.value):
                return copy.deepcopy(request)

            self._release_allocation_unlocked(request)
            request.status = AllocationStatus.CANCELLED.value

            with self._stats_lock:
                self._stats["allocations_released"] += 1

            self._history.append({
                "type": "release",
                "request_id": request_id,
                "resource_type": request.resource_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return copy.deepcopy(request)

    def _release_allocation_unlocked(self, request: AllocationRequest) -> None:
        """Release allocation back to resource pool (no lock)."""
        resource = self._resources.get(request.resource_type)
        if not resource:
            return

        unspent = max(0.0, request.amount - 0.0)
        if request.status in (AllocationStatus.APPROVED.value,
                              AllocationStatus.ACTIVE.value,
                              AllocationStatus.PROPOSED.value):
            already_spent = getattr(request, '_spent_portion', 0.0)
            released = max(0.0, request.amount - already_spent)
            resource.allocated = round(resource.allocated - released, 6)
            resource.allocated = max(0.0, resource.allocated)
            resource.spent = min(resource.spent, resource.allocated)

    def calculate_opportunity_cost(self, resource_type = None,
                                     amount: float = 0.0,
                                     foregone_activity: str = "",
                                     chosen_activity: str = "",
                                     resource_type_arg: str = "") -> OpportunityCost:
        """Calculate what was lost by choosing one activity over another."""
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        elif resource_type is None:
            resource_type = resource_type_arg
        resource_type = _sanitize_strict(resource_type if resource_type else "", 50)
        chosen_activity = _sanitize_content(chosen_activity, 500)
        foregone_activity = _sanitize_content(foregone_activity, 500)

        cost_value = 0.0
        impact_estimate = 0.0

        with self._lock:
            chosen_requests = [r for r in self._allocations.values()
                               if r.activity == chosen_activity
                               and r.status in (AllocationStatus.APPROVED.value,
                                                 AllocationStatus.ACTIVE.value)]
            foregone_priority = 0
            for r in self._allocations.values():
                if r.activity == foregone_activity and r.status == AllocationStatus.PROPOSED.value:
                    foregone_priority = max(foregone_priority, r.priority)

            total_chosen = sum(r.amount for r in chosen_requests)
            resource = self._resources.get(resource_type)
            if resource:
                cost_value = min(1.0, total_chosen * 0.5 + (1.0 - resource.available) * 0.5)
                impact_estimate = min(1.0, foregone_priority / 4.0 + cost_value * 0.3)

            cost = OpportunityCost(
                chosen_activity=chosen_activity,
                foregone_activity=foregone_activity,
                resource_type=resource_type,
                cost_value=cost_value,
                impact_estimate=impact_estimate,
            )

            if len(self._opportunity_costs) >= _MAX_OPPORTUNITY_COSTS:
                evict_id = self._cost_order.popleft()
                self._opportunity_costs.pop(evict_id, None)
                logger.warning("Evicting opportunity cost %s (cap=%d)", evict_id, _MAX_OPPORTUNITY_COSTS)

            self._opportunity_costs[cost.cost_id] = cost
            self._cost_order.append(cost.cost_id)

            with self._stats_lock:
                self._stats["opportunity_costs_calculated"] += 1

        return copy.deepcopy(cost)

    def optimize_allocation(self) -> Dict:
        """Re-optimize allocation based on priorities and budgets."""
        with self._lock:
            proposed = [r for r in self._allocations.values()
                        if r.status == AllocationStatus.PROPOSED.value]
            proposed.sort(key=lambda r: r.priority, reverse=True)

            resources_snapshot = {rt: res.available for rt, res in self._resources.items()}
            optimized = []
            cancelled = []

            for request in proposed:
                available = resources_snapshot.get(request.resource_type, 0.0)
                if request.amount <= available:
                    resource = self._resources[request.resource_type]
                    resource.allocated = round(resource.allocated + request.amount, 6)
                    resource.allocated = _clamp(resource.allocated, 0.0, resource.total_budget)
                    request.status = AllocationStatus.APPROVED.value
                    resources_snapshot[request.resource_type] = resource.available
                    optimized.append(request.request_id)
                else:
                    request.status = AllocationStatus.OVERBUDGET.value
                    cancelled.append(request.request_id)

            reallocated = 0
            for rt, resource in self._resources.items():
                if resource.spent >= resource.allocated:
                    continue
                idle = resource.allocated - resource.spent
                if idle > 0.01:
                    reallocated += idle

            with self._stats_lock:
                self._stats["optimizations_run"] += 1

            result = {
                "optimized": optimized,
                "cancelled": cancelled,
                "reallocated_potential": round(reallocated, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return copy.deepcopy(result)

    def get_budget(self, resource_type: str) -> Optional[CognitiveResource]:
        """Get current budget status."""
        if isinstance(resource_type, ResourceType):
            resource_type = resource_type.value
        resource_type = _sanitize_strict(resource_type, 50)

        with self._lock:
            resource = self._resources.get(resource_type)
            if not resource:
                return None
            return copy.deepcopy(resource)

    def get_allocations(self, status: Optional[str] = None,
                        resource_type: Optional[str] = None) -> List[Dict]:
        """List allocations, optionally filtered."""
        if status is not None:
            status = _sanitize_strict(status, 50)
        if resource_type is not None:
            if isinstance(resource_type, ResourceType):
                resource_type = resource_type.value
            resource_type = _sanitize_strict(resource_type, 50)

        with self._lock:
            results = []
            for rid in self._allocation_order:
                request = self._allocations.get(rid)
                if not request:
                    continue
                if status and request.status != status:
                    continue
                if resource_type and request.resource_type != resource_type:
                    continue
                results.append(copy.deepcopy(request).to_dict())

        return results

    def get_opportunity_costs(self, resource_type: Optional[str] = None,
                              limit: int = 100) -> List[Dict]:
        """List opportunity costs, optionally filtered."""
        if resource_type is not None:
            if isinstance(resource_type, ResourceType):
                resource_type = resource_type.value
            resource_type = _sanitize_strict(resource_type, 50)
        limit = max(1, min(500, limit))

        with self._lock:
            results = []
            for cid in reversed(list(self._cost_order)):
                cost = self._opportunity_costs.get(cid)
                if not cost:
                    continue
                if resource_type and cost.resource_type != resource_type:
                    continue
                results.append(copy.deepcopy(cost).to_dict())
                if len(results) >= limit:
                    break

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate economy report."""
        with self._lock:
            total_budget = sum(r.total_budget for r in self._resources.values())
            total_spent = sum(r.spent for r in self._resources.values())
            total_allocated = sum(r.allocated for r in self._resources.values())
            efficiency = total_spent / max(total_allocated, 0.01)

            recent_costs = [copy.deepcopy(c).to_dict()
                          for c in list(self._opportunity_costs.values())[-10:]]
            active_allocations = sorted(
                [r for r in self._allocations.values()
                 if r.status in (AllocationStatus.APPROVED.value,
                                  AllocationStatus.ACTIVE.value)],
                key=lambda r: r.priority,
                reverse=True,
            )[:10]
            top_allocations = [copy.deepcopy(a).to_dict() for a in active_allocations]

            report = {
                "report_id": "er_%s" % uuid.uuid4().hex[:12],
                "total_budget": round(total_budget, 4),
                "total_spent": round(total_spent, 4),
                "efficiency": round(efficiency, 4),
                "opportunity_costs": recent_costs,
                "top_allocations": top_allocations,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            with self._stats_lock:
                self._stats["reports_generated"] += 1

        return copy.deepcopy(report)

    def suggest_rebalancing(self) -> List[Dict]:
        """Suggest rebalancing actions."""
        suggestions = []

        with self._lock:
            for rt, resource in self._resources.items():
                utilization = resource.allocated / max(resource.total_budget, 0.01)

                if utilization > 0.9:
                    low_priority = [r for r in self._allocations.values()
                                   if r.resource_type == rt
                                   and r.priority <= Priority.LOW.value
                                   and r.status in (AllocationStatus.APPROVED.value,
                                                     AllocationStatus.ACTIVE.value)]
                    for req in low_priority:
                        suggestions.append({
                            "action": "release",
                            "resource_type": rt,
                            "request_id": req.request_id,
                            "activity": req.activity,
                            "amount": round(req.amount, 4),
                            "priority": req.priority,
                            "reason": "High utilization (%.0f%%), release low-priority allocation" % (utilization * 100),
                        })

                if utilization < 0.3:
                    proposed = [r for r in self._allocations.values()
                              if r.resource_type == rt
                              and r.status == AllocationStatus.PROPOSED.value
                              and r.priority >= Priority.MEDIUM.value]
                    for req in proposed[:3]:
                        suggestions.append({
                            "action": "approve",
                            "resource_type": rt,
                            "request_id": req.request_id,
                            "activity": req.activity,
                            "amount": round(req.amount, 4),
                            "priority": req.priority,
                            "reason": "Low utilization (%.0f%%), approve pending requests" % (utilization * 100),
                        })

                wasted = max(0.0, resource.allocated - resource.spent)
                if wasted > 0.2 and resource.spent > 0:
                    waste_ratio = wasted / max(resource.allocated, 0.01)
                    active = [r for r in self._allocations.values()
                             if r.resource_type == rt
                             and r.status == AllocationStatus.ACTIVE.value
                             and waste_ratio > 0.5]
                    for req in active[:2]:
                        suggestions.append({
                            "action": "reduce",
                            "resource_type": rt,
                            "request_id": req.request_id,
                            "activity": req.activity,
                            "wasted": round(wasted, 4),
                            "waste_ratio": round(waste_ratio, 4),
                            "reason": "High waste (%.0f%% allocated but unspent)" % (waste_ratio * 100),
                        })

        return [copy.deepcopy(s) for s in suggestions]

    def get_stats(self) -> Dict:
        """Get statistics."""
        with self._lock:
            total_budget = sum(r.total_budget for r in self._resources.values())
            total_allocated = sum(r.allocated for r in self._resources.values())
            total_spent = sum(r.spent for r in self._resources.values())
            num_resources = len(self._resources)
            num_allocations = len(self._allocations)
            num_opportunity_costs = len(self._opportunity_costs)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["num_resources"] = num_resources
        stats["num_allocations"] = num_allocations
        stats["num_opportunity_costs"] = num_opportunity_costs
        stats["total_budget"] = round(total_budget, 4)
        stats["total_allocated"] = round(total_allocated, 4)
        stats["total_spent"] = round(total_spent, 4)
        stats["efficiency"] = round(total_spent / max(total_allocated, 0.01), 4)
        return stats