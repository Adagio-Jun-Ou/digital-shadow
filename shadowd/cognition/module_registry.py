"""
ShadowAI Module Capability Registry — Discover and manage module capabilities.

Provides runtime discovery of all 45+ cognitive modules and their capabilities:
- Auto-registration from module introspection
- Capability categorization (observe, reason, create, etc.)
- Dependency tracking (which modules need which others)
- Version compatibility checking
- Dynamic module activation/deactivation

This is the "service mesh" for the cognitive architecture — every module registers
its capabilities, and other modules discover what's available at runtime.
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]

def _sanitize_content(text: str, max_length: int = 1000) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_RE.sub('', text).strip()[:max_length]

_MAX_MODULES = 200
_MAX_CAPABILITIES_PER_MODULE = 20
_MAX_DEPENDENCIES_PER_MODULE = 50


class CapabilityType(Enum):
    OBSERVE = "observe"
    REASON = "reason"
    CREATE = "create"
    EXECUTE = "execute"
    LEARN = "learn"
    COMMUNICATE = "communicate"
    STORE = "store"
    ANALYZE = "analyze"
    PREDICT = "predict"
    VALIDATE = "validate"
    SECURE = "secure"


class ModuleStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class ModuleEntry:
    module_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    module_name: str = ""
    display_name: str = ""
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    status: str = "inactive"
    health_score: float = 1.0
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance_ref: Optional[Any] = None

    def __post_init__(self):
        self.module_name = _sanitize(self.module_name, 100)
        self.display_name = _sanitize_content(self.display_name, 200) or self.module_name
        self.description = _sanitize_content(self.description, 1000)
        self.capabilities = [_sanitize(c, 50) for c in self.capabilities[:_MAX_CAPABILITIES_PER_MODULE]]
        self.dependencies = [_sanitize(d, 100) for d in self.dependencies[:_MAX_DEPENDENCIES_PER_MODULE]]
        self.dependents = [_sanitize(d, 100) for d in self.dependents[:_MAX_DEPENDENCIES_PER_MODULE]]
        self.status = _sanitize(self.status, 20)
        self.health_score = max(0.0, min(1.0, self.health_score))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:30]
        }

    def to_dict(self, include_instance: bool = False) -> Dict[str, Any]:
        result = {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "display_name": self.display_name,
            "description": self.description[:200],
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "status": self.status,
            "health_score": round(self.health_score, 3),
            "registered_at": self.registered_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
        }
        if include_instance and self.instance_ref is not None:
            result["has_instance"] = True
        return result


class ModuleCapabilityRegistry:
    _MAX_MODULES = _MAX_MODULES

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._modules: Dict[str, ModuleEntry] = {}
        self._registry_order: deque = deque()
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        self._stats: Dict[str, int] = {
            "modules_registered": 0,
            "modules_activated": 0,
            "modules_deactivated": 0,
            "capabilities_indexed": 0,
            "dependency_edges": 0,
            "circular_deps_detected": 0,
        }

    def register_module(
        self, module_name: str, capabilities: List[str],
        dependencies: Optional[List[str]] = None,
        display_name: str = "", description: str = "",
        instance: Optional[Any] = None,
    ) -> Optional[str]:
        module_name = _sanitize(module_name, 100)
        deps = [_sanitize(d, 100) for d in (dependencies or [])[:20]]

        has_circular = self._detect_circular_dep(module_name, deps, set())
        if has_circular:
            with self._stats_lock:
                self._stats["circular_deps_detected"] += 1
            logger.warning("Circular dependency detected for %s", module_name)
            return None

        with self._lock:
            if module_name in self._modules:
                entry = self._modules[module_name]
                entry.capabilities = [_sanitize(c, 50) for c in capabilities[:20]]
                entry.dependencies = deps
                entry.instance_ref = instance
                entry.registered_at = datetime.now(timezone.utc)
            else:
                entry = ModuleEntry(
                    module_name=module_name, display_name=display_name or module_name,
                    description=description, capabilities=capabilities,
                    dependencies=deps, instance_ref=instance,
                )
                self._modules[module_name] = entry
                self._registry_order.append(module_name)
                while len(self._modules) > self._MAX_MODULES:
                    old_name = self._registry_order.popleft()
                    self._modules.pop(old_name, None)
                    for cap_set in self._capability_index.values():
                        cap_set.discard(old_name)

            for cap in entry.capabilities:
                self._capability_index[cap].add(module_name)

            for dep in entry.dependencies:
                self._dependency_graph[module_name].add(dep)
                self._reverse_dependency_graph[dep].add(module_name)

            for other_name, other_entry in self._modules.items():
                if module_name in other_entry.dependencies:
                    other_entry.dependents.append(module_name)
                if other_name in entry.dependencies:
                    entry.dependents.append(other_name)

        with self._stats_lock:
            self._stats["modules_registered"] = len(self._modules)
            self._stats["capabilities_indexed"] = len(self._capability_index)
            self._stats["dependency_edges"] = sum(
                len(deps) for deps in self._dependency_graph.values()
            )

        return entry.module_id

    def set_status(self, module_name: str, status: str) -> bool:
        module_name = _sanitize(module_name, 100)
        status = _sanitize(status, 20)
        with self._lock:
            entry = self._modules.get(module_name)
            if entry is None:
                return False
            old_status = entry.status
            entry.status = status
            if status == "active":
                entry.last_active_at = datetime.now(timezone.utc)
                if old_status != "active":
                    with self._stats_lock:
                        self._stats["modules_activated"] += 1
            elif status == "inactive" and old_status == "active":
                with self._stats_lock:
                    self._stats["modules_deactivated"] += 1
        return True

    def set_health(self, module_name: str, health_score: float) -> bool:
        module_name = _sanitize(module_name, 100)
        health_score = max(0.0, min(1.0, health_score))
        with self._lock:
            entry = self._modules.get(module_name)
            if entry is None:
                return False
            entry.health_score = health_score
            if health_score < 0.3:
                entry.status = "degraded"
            elif health_score >= 0.7 and entry.status == "degraded":
                entry.status = "active"
        return True

    def find_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        capability = _sanitize(capability, 50)
        with self._lock:
            module_names = self._capability_index.get(capability, set())
            return [
                self._modules[name].to_dict()
                for name in module_names
                if name in self._modules
            ]

    def find_by_dependency(self, dep_name: str) -> List[Dict[str, Any]]:
        dep_name = _sanitize(dep_name, 100)
        with self._lock:
            dependents = self._reverse_dependency_graph.get(dep_name, set())
            return [
                self._modules[name].to_dict()
                for name in dependents
                if name in self._modules
            ]

    def get_module(self, module_name: str, include_instance: bool = False) -> Optional[Dict[str, Any]]:
        module_name = _sanitize(module_name, 100)
        with self._lock:
            entry = self._modules.get(module_name)
            if entry is None:
                return None
            return entry.to_dict(include_instance=include_instance)

    def list_modules(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        status = _sanitize(status, 20) if status else None
        with self._lock:
            entries = list(self._modules.values())
            if status:
                entries = [e for e in entries if e.status == status]
            return [e.to_dict() for e in entries]

    def get_dependency_graph(self) -> Dict[str, Any]:
        with self._lock:
            nodes = [e.to_dict() for e in self._modules.values()]
            edges = []
            for src, deps in self._dependency_graph.items():
                for tgt in deps:
                    edges.append({"source": src, "target": tgt})
        return {"modules": nodes, "dependency_edges": edges, "edge_count": len(edges)}

    def get_compatible_modules(self, module_name: str) -> List[Dict[str, Any]]:
        module_name = _sanitize(module_name, 100)
        with self._lock:
            entry = self._modules.get(module_name)
            if entry is None:
                return []
            compatible = []
            for name, other in self._modules.items():
                if name == module_name:
                    continue
                shared_caps = set(entry.capabilities) & set(other.capabilities)
                if shared_caps:
                    compatible.append({
                        "module_name": name,
                        "display_name": other.display_name,
                        "shared_capabilities": sorted(shared_caps),
                    })
            return compatible

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_modules = len(self._modules)
            active_count = sum(1 for e in self._modules.values() if e.status == "active")
            degraded_count = sum(1 for e in self._modules.values() if e.status == "degraded")

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_modules"] = total_modules
        stats["active_modules"] = active_count
        stats["degraded_modules"] = degraded_count
        stats["capability_types"] = len(self._capability_index)
        return stats

    def _detect_circular_dep(self, module_name: str, deps: List[str],
                             visited: Set[str]) -> bool:
        if module_name in visited:
            return True
        visited.add(module_name)
        for dep in deps:
            if dep not in self._modules:
                continue
            dep_deps = self._dependency_graph.get(dep, set())
            if self._detect_circular_dep(dep, list(dep_deps), set(visited)):
                return True
        return False
