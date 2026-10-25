"""
Cognitive Architecture Visualizer — Runtime introspection and visualization.

Generates live views of the cognitive architecture:
1. Module dependency graph (who calls whom)
2. Data flow map (what data flows between modules)
3. Cognitive state timeline (mode transitions over time)
4. Resource utilization heatmap
5. Cross-module interaction frequency matrix

Output formats: JSON, GraphViz DOT, Mermaid, HTML snippets

Design: Pure data generation — no rendering, just structured output that
frontends can consume.
"""

import copy
import logging
import re
import threading
import uuid
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _sanitize_content(text: str, max_length: int = 1000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]


_MAX_EVENTS = 5000
_MAX_SNAPSHOTS = 500
_MAX_MODULES = 100
_MAX_INTERACTIONS_PER_PAIR = 1000
_MAX_TIMELINE_POINTS = 200


class VisualizerFormat(Enum):
    JSON = "json"
    DOT = "dot"
    MERMAID = "mermaid"
    HTML = "html"


class EventDirection(Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class InteractionEvent:
    source_module: str
    target_module: str
    event_type: str
    data_size: int = 0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.source_module = _sanitize(self.source_module, 100)
        self.target_module = _sanitize(self.target_module, 100)
        self.event_type = _sanitize(self.event_type, 50)
        self.data_size = max(0, self.data_size)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "event_type": self.event_type,
            "data_size": self.data_size,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ArchitectureSnapshot:
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_modules: List[str] = field(default_factory=list)
    module_states: Dict[str, str] = field(default_factory=dict)
    interaction_counts: Dict[str, int] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "active_modules": list(self.active_modules),
            "module_states": dict(self.module_states),
            "interaction_counts": copy.deepcopy(self.interaction_counts),
            "resource_usage": {k: round(v, 4) for k, v in self.resource_usage.items()},
            "metadata": copy.deepcopy(self.metadata),
        }


class CognitiveVisualizer:
    _MAX_EVENTS = _MAX_EVENTS
    _MAX_SNAPSHOTS = _MAX_SNAPSHOTS
    _MAX_MODULES = _MAX_MODULES

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._events: deque = deque(maxlen=_MAX_EVENTS)
        self._snapshots: deque = deque(maxlen=_MAX_SNAPSHOTS)
        self._module_states: Dict[str, str] = {}
        self._interaction_matrix: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_INTERACTIONS_PER_PAIR)
        )
        self._module_metadata: Dict[str, Dict[str, Any]] = {}

        self._stats: Dict[str, int] = {
            "events_recorded": 0,
            "snapshots_taken": 0,
            "modules_tracked": 0,
            "unique_pairs": 0,
            "graphs_generated": 0,
        }

    def record_interaction(
        self,
        source_module: str,
        target_module: str,
        event_type: str,
        data_size: int = 0,
        confidence: float = 0.0,
    ) -> str:
        event = InteractionEvent(
            source_module=source_module,
            target_module=target_module,
            event_type=event_type,
            data_size=data_size,
            confidence=confidence,
        )

        with self._lock:
            self._events.append(event)
            pair_key = (event.source_module, event.target_module)
            self._interaction_matrix[pair_key].append(event)
            self._module_states[event.source_module] = "active"
            self._module_states[event.target_module] = "active"
            unique_pairs = len(self._interaction_matrix)
            modules_tracked = len(self._module_states)

        with self._stats_lock:
            self._stats["events_recorded"] += 1
            self._stats["unique_pairs"] = unique_pairs
            self._stats["modules_tracked"] = modules_tracked

        return event.event_id

    def take_snapshot(self, active_modules: List[str]) -> str:
        snapshot = ArchitectureSnapshot(
            active_modules=[_sanitize(m, 100) for m in active_modules[:_MAX_MODULES]],
        )

        with self._lock:
            snapshot.module_states = copy.deepcopy(dict(self._module_states))
            for (src, tgt), events in self._interaction_matrix.items():
                key = f"{src}->{tgt}"
                snapshot.interaction_counts[key] = len(events)
            self._snapshots.append(snapshot)

        with self._stats_lock:
            self._stats["snapshots_taken"] += 1

        return snapshot.snapshot_id

    def set_module_state(self, module_name: str, state: str) -> None:
        module_name = _sanitize(module_name, 100)
        state = _sanitize(state, 50)
        with self._lock:
            self._module_states[module_name] = state

    def set_module_metadata(self, module_name: str, meta: Dict[str, Any]) -> None:
        module_name = _sanitize(module_name, 100)
        sanitized = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((meta or {}).items())[:50]
        }
        with self._lock:
            self._module_metadata[module_name] = sanitized

    def generate_dependency_graph(self, fmt: str = "json") -> Any:
        with self._lock:
            nodes = []
            for module in sorted(self._module_states.keys()):
                nodes.append({
                    "id": module,
                    "label": module,
                    "state": self._module_states.get(module, "unknown"),
                    "metadata": copy.deepcopy(self._module_metadata.get(module, {})),
                })

            edges = []
            for (src, tgt), events in self._interaction_matrix.items():
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": len(events),
                    "avg_confidence": round(
                        sum(e.confidence for e in events) / max(1, len(events)), 3
                    ),
                })

        if fmt == "dot":
            return self._to_dot(nodes, edges)
        elif fmt == "mermaid":
            return self._to_mermaid(nodes, edges)
        elif fmt == "html":
            return self._to_html(nodes, edges)
        else:
            return {"nodes": nodes, "edges": edges}

    def generate_interaction_matrix(self) -> Dict[str, Any]:
        with self._lock:
            modules = sorted(self._module_states.keys())
            matrix_data = []

            for src in modules:
                row = {"source": src, "targets": {}}
                for tgt in modules:
                    key = (src, tgt)
                    if key in self._interaction_matrix:
                        row["targets"][tgt] = len(self._interaction_matrix[key])
                matrix_data.append(row)

        return {
            "modules": modules,
            "module_count": len(modules),
            "matrix": matrix_data,
            "total_interactions": sum(
                len(events) for events in self._interaction_matrix.values()
            ),
        }

    def generate_timeline(self, limit: int = _MAX_TIMELINE_POINTS) -> Dict[str, Any]:
        limit = max(1, min(_MAX_EVENTS, limit))

        with self._lock:
            events = list(self._events)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            timeline = [e.to_dict() for e in events[:limit]]

        state_timeline = []
        if events:
            state_timeline = [
                {"module": mod, "state": state}
                for mod, state in sorted(self._module_states.items())[:50]
            ]

        return {
            "event_count": len(timeline),
            "events": timeline,
            "module_states": state_timeline,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_resource_heatmap(self) -> Dict[str, Any]:
        with self._lock:
            usage = {}
            for (src, tgt), events in self._interaction_matrix.items():
                for mod in [src, tgt]:
                    if mod not in usage:
                        usage[mod] = 0.0
                    usage[mod] += len(events) * 0.1

            max_usage = max(usage.values()) if usage else 1.0
            heatmap = {}
            for mod, val in usage.items():
                heatmap[mod] = round(val / max_usage, 3)

        return {
            "modules": list(heatmap.keys()),
            "usage_scores": copy.deepcopy(heatmap),
            "max_module": max(heatmap.items(), key=lambda x: x[1]) if heatmap else ("", 0),
        }

    def get_recent_interactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_EVENTS, limit))
        with self._lock:
            events = list(self._events)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            return [e.to_dict() for e in events[:limit]]

    def get_module_overview(self) -> Dict[str, Any]:
        with self._lock:
            module_list = []
            for mod in sorted(self._module_states.keys()):
                outbound = sum(
                    len(events)
                    for (src, _), events in self._interaction_matrix.items()
                    if src == mod
                )
                inbound = sum(
                    len(events)
                    for (_, tgt), events in self._interaction_matrix.items()
                    if tgt == mod
                )
                module_list.append({
                    "name": mod,
                    "state": self._module_states.get(mod, "unknown"),
                    "outbound_interactions": outbound,
                    "inbound_interactions": inbound,
                    "metadata": copy.deepcopy(self._module_metadata.get(mod, {})),
                })

        return {
            "total_modules": len(module_list),
            "modules": module_list,
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_events = len(self._events)
            total_snapshots = len(self._snapshots)
            active_modules = len(self._module_states)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_events"] = total_events
        stats["total_snapshots"] = total_snapshots
        stats["active_modules"] = active_modules
        return stats

    def _to_dot(self, nodes: List[Dict], edges: List[Dict]) -> str:
        lines = ["digraph ShadowAI {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=filled, fillcolor=lightblue];")

        for node in nodes:
            state = node.get("state", "")
            color = "lightblue"
            if state == "active":
                color = "lightgreen"
            elif state == "error":
                color = "lightcoral"
            lines.append(
                f'  "{node["id"]}" [label="{node["label"]}", fillcolor={color}];'
            )

        for edge in edges:
            weight = edge.get("weight", 1)
            penwidth = max(1, min(10, weight // 5 + 1))
            lines.append(
                f'  "{edge["source"]}" -> "{edge["target"]}" '
                f'[label="{weight}", penwidth={penwidth}];'
            )

        lines.append("}")
        return "\n".join(lines)

    def _to_mermaid(self, nodes: List[Dict], edges: List[Dict]) -> str:
        lines = ["graph TD;"]
        for node in nodes:
            safe_id = node["id"].replace("-", "_").replace(".", "_")
            lines.append(f'  {safe_id}["{node["label"]}"];')

        for edge in edges:
            safe_src = edge["source"].replace("-", "_").replace(".", "_")
            safe_tgt = edge["target"].replace("-", "_").replace(".", "_")
            lines.append(f'  {safe_src} -->|{edge["weight"]}| {safe_tgt};')

        return "\n".join(lines)

    def _to_html(self, nodes: List[Dict], edges: List[Dict]) -> str:
        parts = ['<div class="cognitive-graph">']
        parts.append('<h2>Cognitive Architecture</h2>')
        parts.append('<div class="modules">')
        for node in nodes:
            parts.append(
                f'<div class="module {node.get("state", "")}">'
                f'{node["label"]}</div>'
            )
        parts.append("</div>")
        parts.append('<div class="connections">')
        for edge in edges:
            parts.append(
                f'<div class="edge">{edge["source"]} → '
                f'{edge["target"]} ({edge["weight"]})</div>'
            )
        parts.append("</div>")
        parts.append("</div>")
        return "\n".join(parts)
