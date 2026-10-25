#!/usr/bin/env python3
"""
CognitiveTopology -- Knowledge Network Topology Analysis.

Analyzes the topology of the cognitive knowledge graph -- identifying
clusters, bridges, bottlenecks, and critical paths. Detects whether
knowledge is well-connected or fragmented, identifies isolated domains,
and suggests connections that would improve overall knowledge flow.

Inspired by network science: small-world networks, scale-free networks,
betweenness centrality, clustering coefficients.

Key concepts:
  - TopologyNode: a node in the knowledge graph with centrality metrics
  - TopologyEdge: a typed, weighted connection between nodes
  - TopologyCluster: a group of densely connected nodes
  - TopologyAnalysis: full topology analysis with metrics and suggestions

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import logging
import math
import re
import threading
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("shadowd.cognition.cognitive_topology")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_NODES = 3000
_MAX_EDGES = 10000
_MAX_CLUSTERS = 100
_MAX_ANALYSES = 50
_MAX_NODE_METADATA_KEYS = 50
_MAX_EDGE_METADATA_KEYS = 50
_MAX_METADATA_VALUE_LENGTH = 500
_MAX_SUGGESTIONS = 100


class NodeType(str, Enum):
    CONCEPT = "concept"
    SKILL = "skill"
    EXPERIENCE = "experience"
    BELIEF = "belief"
    GOAL = "goal"
    RESOURCE = "resource"
    RELATIONSHIP = "relationship"


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    ENHANCES = "enhances"
    CONTRADICTS = "contradicts"
    REQUIRES = "requires"
    PRODUCES = "produces"
    INHIBITS = "inhibits"


class ClusterType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    BRIDGED = "bridged"
    ISOLATED = "isolated"
    HUB = "hub"


class TopologyScore(int, Enum):
    FRAGMENTED = 0
    LOOSE = 1
    MODERATE = 2
    CONNECTED = 3
    INTEGRATED = 4


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sanitize_metadata(metadata: dict, max_keys: int = 50, max_val_len: int = 500) -> dict:
    if not isinstance(metadata, dict):
        return {}
    out = {}
    for k, v in list(metadata.items())[:max_keys]:
        sk = _sanitize(str(k), 100)
        if isinstance(v, str):
            sv = _sanitize_content(v, max_val_len)
        elif isinstance(v, (int, float, bool)):
            sv = v
        else:
            sv = _sanitize_content(str(v), max_val_len)
        out[sk] = sv
    return out


@dataclass
class TopologyNode:
    node_id: str = ""
    name: str = ""
    node_type: NodeType = NodeType.CONCEPT
    centrality: float = 0.0
    clustering_coefficient: float = 0.0
    degree: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = "tn_%s" % uuid.uuid4().hex[:12]
        self.node_id = _sanitize(self.node_id, 64)
        self.name = _sanitize(self.name, 500)
        if isinstance(self.node_type, str):
            try:
                self.node_type = NodeType(self.node_type)
            except ValueError:
                self.node_type = NodeType.CONCEPT
        self.centrality = _clamp(self.centrality)
        self.clustering_coefficient = _clamp(self.clustering_coefficient)
        if not isinstance(self.degree, int):
            self.degree = max(0, int(self.degree))
        self.metadata = _sanitize_metadata(
            self.metadata, _MAX_NODE_METADATA_KEYS, _MAX_METADATA_VALUE_LENGTH
        )

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "centrality": round(self.centrality, 4),
            "clustering_coefficient": round(self.clustering_coefficient, 4),
            "degree": self.degree,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class TopologyEdge:
    edge_id: str = ""
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.ENHANCES
    weight: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = "te_%s" % uuid.uuid4().hex[:12]
        self.edge_id = _sanitize(self.edge_id, 64)
        self.source_id = _sanitize(self.source_id, 64)
        self.target_id = _sanitize(self.target_id, 64)
        if isinstance(self.edge_type, str):
            try:
                self.edge_type = EdgeType(self.edge_type)
            except ValueError:
                self.edge_type = EdgeType.ENHANCES
        self.weight = _clamp(self.weight)
        self.metadata = _sanitize_metadata(
            self.metadata, _MAX_EDGE_METADATA_KEYS, _MAX_METADATA_VALUE_LENGTH
        )

    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": round(self.weight, 4),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class TopologyCluster:
    cluster_id: str = ""
    name: str = ""
    cluster_type: ClusterType = ClusterType.DENSE
    node_ids: List[str] = field(default_factory=list)
    internal_density: float = 0.0
    bridging_centrality: float = 0.0

    def __post_init__(self):
        if not self.cluster_id:
            self.cluster_id = "tc_%s" % uuid.uuid4().hex[:12]
        self.cluster_id = _sanitize(self.cluster_id, 64)
        self.name = _sanitize_content(self.name, 200)
        if isinstance(self.cluster_type, str):
            try:
                self.cluster_type = ClusterType(self.cluster_type)
            except ValueError:
                self.cluster_type = ClusterType.DENSE
        self.node_ids = [_sanitize(n, 64) for n in (self.node_ids or [])[:_MAX_NODES]]
        self.internal_density = _clamp(self.internal_density)
        self.bridging_centrality = _clamp(self.bridging_centrality)

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "cluster_type": self.cluster_type.value,
            "node_ids": list(self.node_ids),
            "internal_density": round(self.internal_density, 4),
            "bridging_centrality": round(self.bridging_centrality, 4),
        }


@dataclass
class TopologyAnalysis:
    analysis_id: str = ""
    total_nodes: int = 0
    total_edges: int = 0
    avg_clustering: float = 0.0
    avg_path_length: float = 0.0
    small_world_coefficient: float = 0.0
    topology_score: TopologyScore = TopologyScore.FRAGMENTED
    clusters: List[str] = field(default_factory=list)
    bridges: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.analysis_id:
            self.analysis_id = "ta_%s" % uuid.uuid4().hex[:12]
        self.analysis_id = _sanitize(self.analysis_id, 64)
        if not isinstance(self.total_nodes, int):
            self.total_nodes = max(0, int(self.total_nodes))
        if not isinstance(self.total_edges, int):
            self.total_edges = max(0, int(self.total_edges))
        self.avg_clustering = _clamp(self.avg_clustering)
        self.avg_path_length = max(0.0, self.avg_path_length)
        self.small_world_coefficient = _clamp(self.small_world_coefficient)
        if isinstance(self.topology_score, int) and not isinstance(self.topology_score, TopologyScore):
            try:
                self.topology_score = TopologyScore(self.topology_score)
            except ValueError:
                self.topology_score = TopologyScore.FRAGMENTED
        elif isinstance(self.topology_score, str):
            try:
                self.topology_score = TopologyScore[self.topology_score]
            except (KeyError, ValueError):
                self.topology_score = TopologyScore.FRAGMENTED
        self.clusters = [_sanitize(c, 64) for c in (self.clusters or [])[:_MAX_CLUSTERS]]
        self.bridges = [_sanitize(b, 64) for b in (self.bridges or [])[:_MAX_NODES]]
        self.bottlenecks = [_sanitize(b, 64) for b in (self.bottlenecks or [])[:_MAX_NODES]]

    def to_dict(self) -> Dict:
        return {
            "analysis_id": self.analysis_id,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "avg_clustering": round(self.avg_clustering, 4),
            "avg_path_length": round(self.avg_path_length, 4),
            "small_world_coefficient": round(self.small_world_coefficient, 4),
            "topology_score": self.topology_score.value,
            "clusters": list(self.clusters),
            "bridges": list(self.bridges),
            "bottlenecks": list(self.bottlenecks),
        }


class CognitiveTopology:
    """Knowledge Network Topology Analysis.

    Builds and analyzes a knowledge graph using network science metrics.
    Identifies clusters, bridges, bottlenecks, isolated domains, and
    suggests connections to improve overall knowledge flow.
    """

    _MAX_NODES = _MAX_NODES
    _MAX_EDGES = _MAX_EDGES
    _MAX_CLUSTERS = _MAX_CLUSTERS
    _MAX_ANALYSES = _MAX_ANALYSES

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._nodes: Dict[str, TopologyNode] = {}
        self._node_order: deque = deque(maxlen=_MAX_NODES)
        self._edges: Dict[str, TopologyEdge] = {}
        self._edge_order: deque = deque(maxlen=_MAX_EDGES)
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._edge_adjacency: Dict[str, List[str]] = defaultdict(list)
        self._clusters: Dict[str, TopologyCluster] = {}
        self._cluster_order: deque = deque(maxlen=_MAX_CLUSTERS)
        self._analyses: Dict[str, TopologyAnalysis] = {}
        self._analysis_order: deque = deque(maxlen=_MAX_ANALYSES)
        self._stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "nodes_removed": 0,
            "edges_removed": 0,
            "analyses_run": 0,
            "clusters_found": 0,
            "bridges_found": 0,
            "bottlenecks_found": 0,
            "suggestions_generated": 0,
        }

    def add_node(self, name: str, node_type: NodeType = NodeType.CONCEPT,
                 metadata: Optional[Dict[str, Any]] = None) -> TopologyNode:
        name = _sanitize(name, 500)
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type)
            except ValueError:
                node_type = NodeType.CONCEPT
        if metadata is None:
            metadata = {}

        node = TopologyNode(
            node_id="tn_%s" % uuid.uuid4().hex[:12],
            name=name,
            node_type=node_type,
            metadata=metadata,
        )

        with self._lock:
            if len(self._nodes) >= self._MAX_NODES:
                evict_id = self._node_order.popleft()
                self._remove_node_unlocked(evict_id)
                logger.warning("Evicting node %s (cap=%d)", evict_id, self._MAX_NODES)
            self._nodes[node.node_id] = node
            self._node_order.append(node.node_id)
            self._adjacency[node.node_id] = set()
            self._edge_adjacency[node.node_id] = []
            with self._stats_lock:
                self._stats["nodes_added"] += 1

        return copy.deepcopy(node)

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType = EdgeType.ENHANCES,
                 weight: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[TopologyEdge]:
        source_id = _sanitize(source_id, 64)
        target_id = _sanitize(target_id, 64)
        if isinstance(edge_type, str):
            try:
                edge_type = EdgeType(edge_type)
            except ValueError:
                edge_type = EdgeType.ENHANCES
        weight = _clamp(weight)
        if metadata is None:
            metadata = {}

        edge = TopologyEdge(
            edge_id="te_%s" % uuid.uuid4().hex[:12],
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata,
        )

        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return None

            if len(self._edges) >= self._MAX_EDGES:
                evict_id = self._edge_order.popleft()
                self._remove_edge_unlocked(evict_id)
                logger.warning("Evicting edge %s (cap=%d)", evict_id, self._MAX_EDGES)

            self._edges[edge.edge_id] = edge
            self._edge_order.append(edge.edge_id)
            self._adjacency[source_id].add(target_id)
            self._adjacency[target_id].add(source_id)
            self._edge_adjacency[source_id].append(edge.edge_id)
            self._edge_adjacency[target_id].append(edge.edge_id)

            src = self._nodes[source_id]
            tgt = self._nodes[target_id]
            src.degree = len(self._adjacency[source_id])
            tgt.degree = len(self._adjacency[target_id])

            with self._stats_lock:
                self._stats["edges_added"] += 1

        return copy.deepcopy(edge)

    def remove_node(self, node_id: str) -> bool:
        node_id = _sanitize(node_id, 64)
        with self._lock:
            if node_id not in self._nodes:
                return False
            self._remove_node_unlocked(node_id)
            try:
                self._node_order.remove(node_id)
            except ValueError:
                pass
            with self._stats_lock:
                self._stats["nodes_removed"] += 1
        return True

    def remove_edge(self, edge_id: str) -> bool:
        edge_id = _sanitize(edge_id, 64)
        with self._lock:
            if edge_id not in self._edges:
                return False
            self._remove_edge_unlocked(edge_id)
            try:
                self._edge_order.remove(edge_id)
            except ValueError:
                pass
            with self._stats_lock:
                self._stats["edges_removed"] += 1
        return True

    def analyze_topology(self) -> TopologyAnalysis:
        with self._lock:
            nodes_snap = dict(self._nodes)
            edges_snap = dict(self._edges)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        centrality = self._compute_betweenness_centrality(nodes_snap, adj_snap)
        clustering = self._compute_clustering_coefficients(nodes_snap, adj_snap)
        avg_clustering = self._compute_avg_clustering(nodes_snap, clustering)
        avg_path_length = self._compute_avg_path_length(nodes_snap, adj_snap)
        small_world = self._compute_small_world_coefficient(
            nodes_snap, adj_snap, avg_clustering, avg_path_length
        )
        topology_score = self._evaluate_topology_score(
            nodes_snap, avg_clustering, avg_path_length, adj_snap
        )

        bridges = self._find_bridges_internal(nodes_snap, adj_snap)
        bottlenecks = self._find_bottlenecks_internal(nodes_snap, centrality)

        with self._lock:
            for nid, cent in centrality.items():
                if nid in self._nodes:
                    self._nodes[nid].centrality = cent
            for nid, cc in clustering.items():
                if nid in self._nodes:
                    self._nodes[nid].clustering_coefficient = cc

            cluster_ids = [c.cluster_id for c in self._clusters.values()]
            bridge_ids = [b for b in bridges if b in self._nodes]
            bottleneck_ids = [b for b in bottlenecks if b in self._nodes]

            analysis = TopologyAnalysis(
                analysis_id="ta_%s" % uuid.uuid4().hex[:12],
                total_nodes=len(nodes_snap),
                total_edges=len(edges_snap),
                avg_clustering=avg_clustering,
                avg_path_length=max(0.0, avg_path_length),
                small_world_coefficient=small_world,
                topology_score=topology_score,
                clusters=cluster_ids[:_MAX_CLUSTERS],
                bridges=bridge_ids,
                bottlenecks=bottleneck_ids,
            )

            if len(self._analyses) >= self._MAX_ANALYSES:
                evict_id = self._analysis_order.popleft()
                self._analyses.pop(evict_id, None)
                logger.warning("Evicting analysis %s (cap=%d)", evict_id, self._MAX_ANALYSES)

            self._analyses[analysis.analysis_id] = analysis
            self._analysis_order.append(analysis.analysis_id)

            with self._stats_lock:
                self._stats["analyses_run"] += 1
                self._stats["bridges_found"] = len(bridge_ids)
                self._stats["bottlenecks_found"] = len(bottleneck_ids)

        return copy.deepcopy(analysis)

    def find_clusters(self, min_density: float = 0.3) -> List[TopologyCluster]:
        min_density = _clamp(min_density, 0.0, 1.0)
        with self._lock:
            nodes_snap = dict(self._nodes)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        visited: Set[str] = set()
        component_map = self._find_connected_components(nodes_snap, adj_snap)
        cluster_list = []

        for comp in component_map:
            if len(comp) < 2:
                if len(comp) == 1:
                    nid = list(comp)[0]
                    if nid in nodes_snap and nodes_snap[nid].degree == 0:
                        cluster = TopologyCluster(
                            cluster_id="tc_%s" % uuid.uuid4().hex[:12],
                            name="Isolated: %s" % nodes_snap[nid].name[:80],
                            cluster_type=ClusterType.ISOLATED,
                            node_ids=[nid],
                            internal_density=0.0,
                            bridging_centrality=0.0,
                        )
                        cluster_list.append(cluster)
                continue

            internal_density = self._compute_cluster_density(comp, adj_snap)
            if internal_density < min_density:
                continue

            if internal_density >= 0.7:
                ctype = ClusterType.DENSE
            elif internal_density >= 0.5:
                ctype = ClusterType.HUB
            elif internal_density >= 0.3:
                ctype = ClusterType.BRIDGED
            else:
                ctype = ClusterType.SPARSE

            bridge_cent = 0.0
            for nid in comp:
                external = len(adj_snap.get(nid, set()) - comp)
                if external > 0:
                    bridge_cent = max(bridge_cent, external / max(len(adj_snap.get(nid, set())), 1))

            names = [nodes_snap[n].name for n in comp if n in nodes_snap]
            cluster_name = ", ".join(names[:3])
            if len(names) > 3:
                cluster_name += " (+%d)" % (len(names) - 3)

            cluster = TopologyCluster(
                cluster_id="tc_%s" % uuid.uuid4().hex[:12],
                name=cluster_name[:200],
                cluster_type=ctype,
                node_ids=list(comp),
                internal_density=internal_density,
                bridging_centrality=bridge_cent,
            )
            cluster_list.append(cluster)

        with self._lock:
            for c in cluster_list:
                if len(self._clusters) >= self._MAX_CLUSTERS:
                    evict_id = self._cluster_order.popleft()
                    self._clusters.pop(evict_id, None)
                    logger.warning("Evicting cluster %s (cap=%d)", evict_id, self._MAX_CLUSTERS)
                self._clusters[c.cluster_id] = c
                self._cluster_order.append(c.cluster_id)

            with self._stats_lock:
                self._stats["clusters_found"] = len(self._clusters)

        return [copy.deepcopy(c) for c in cluster_list]

    def find_bridges(self) -> List[str]:
        with self._lock:
            nodes_snap = dict(self._nodes)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        bridges = self._find_bridges_internal(nodes_snap, adj_snap)

        with self._stats_lock:
            self._stats["bridges_found"] = len(bridges)

        return list(bridges)

    def find_bottlenecks(self) -> List[str]:
        with self._lock:
            nodes_snap = dict(self._nodes)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        centrality = self._compute_betweenness_centrality(nodes_snap, adj_snap)
        bottlenecks = self._find_bottlenecks_internal(nodes_snap, centrality)

        with self._stats_lock:
            self._stats["bottlenecks_found"] = len(bottlenecks)

        return list(bottlenecks)

    def find_isolated_nodes(self) -> List[str]:
        with self._lock:
            result = [
                nid for nid, node in self._nodes.items()
                if node.degree == 0 or not self._adjacency.get(nid)
            ]
        return result

    def suggest_connections(self, max_suggestions: int = 20) -> List[Dict]:
        max_suggestions = max(1, min(max_suggestions, _MAX_SUGGESTIONS))
        with self._lock:
            nodes_snap = dict(self._nodes)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        suggestions = []
        existing_edges = set()
        for src, neighbors in adj_snap.items():
            for tgt in neighbors:
                existing_edges.add((min(src, tgt), max(src, tgt)))

        node_list = list(nodes_snap.keys())
        candidates = []

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                if (a, b) in existing_edges:
                    continue
                common = len(adj_snap.get(a, set()) & adj_snap.get(b, set()))
                total = len(adj_snap.get(a, set()) | adj_snap.get(b, set()))
                jaccard = common / max(total, 1)
                dist = self._shortest_path_length(adj_snap, a, b)
                if dist is None or dist > 2:
                    score = jaccard * 0.5
                    if dist is None:
                        score += 0.3
                    elif dist > 3:
                        score += 0.2
                    candidates.append((a, b, score, common, dist))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for a, b, score, common, dist in candidates[:max_suggestions]:
            na = nodes_snap.get(a)
            nb = nodes_snap.get(b)
            if not na or not nb:
                continue
            suggestions.append({
                "source_id": a,
                "source_name": na.name,
                "target_id": b,
                "target_name": nb.name,
                "score": round(score, 4),
                "shared_neighbors": common,
                "current_distance": dist if dist is not None else -1,
            })

        with self._stats_lock:
            self._stats["suggestions_generated"] += len(suggestions)

        return suggestions

    def get_node(self, node_id: str) -> Optional[TopologyNode]:
        node_id = _sanitize(node_id, 64)
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return None
            return copy.deepcopy(node)

    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        source_id = _sanitize(source_id, 64)
        target_id = _sanitize(target_id, 64)
        with self._lock:
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        if source_id not in adj_snap or target_id not in adj_snap:
            return None

        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = deque([(source_id, [source_id])])
        while queue:
            current, path = queue.popleft()
            for neighbor in adj_snap.get(current, set()):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def get_metrics(self) -> Dict:
        with self._lock:
            nodes_snap = dict(self._nodes)
            adj_snap = {k: set(v) for k, v in self._adjacency.items()}

        if not nodes_snap:
            return {
                "total_nodes": 0,
                "total_edges": sum(len(v) for v in adj_snap.values()) // 2,
                "avg_degree": 0.0,
                "avg_clustering": 0.0,
                "avg_path_length": 0.0,
                "small_world_coefficient": 0.0,
                "topology_score": TopologyScore.FRAGMENTED.value,
                "density": 0.0,
            }

        total_edges = sum(len(v) for v in adj_snap.values()) // 2
        n = len(nodes_snap)
        avg_degree = (2 * total_edges) / n if n > 0 else 0.0
        density = (2 * total_edges) / (n * (n - 1)) if n > 1 else 0.0

        clustering = self._compute_clustering_coefficients(nodes_snap, adj_snap)
        avg_clustering = self._compute_avg_clustering(nodes_snap, clustering)
        avg_path_length = self._compute_avg_path_length(nodes_snap, adj_snap)
        small_world = self._compute_small_world_coefficient(
            nodes_snap, adj_snap, avg_clustering, avg_path_length
        )
        topology_score = self._evaluate_topology_score(
            nodes_snap, avg_clustering, avg_path_length, adj_snap
        )

        return {
            "total_nodes": n,
            "total_edges": total_edges,
            "avg_degree": round(avg_degree, 4),
            "avg_clustering": round(avg_clustering, 4),
            "avg_path_length": round(avg_path_length, 4),
            "small_world_coefficient": round(small_world, 4),
            "topology_score": topology_score.value,
            "density": round(_clamp(density), 4),
        }

    def get_stats(self) -> Dict:
        with self._lock:
            nodes_count = len(self._nodes)
            edges_count = len(self._edges)
            clusters_count = len(self._clusters)
            analyses_count = len(self._analyses)
        with self._stats_lock:
            stats = dict(self._stats)
        stats["nodes_count"] = nodes_count
        stats["edges_count"] = edges_count
        stats["clusters_count"] = clusters_count
        stats["analyses_count"] = analyses_count
        return stats

    def _remove_node_unlocked(self, node_id: str) -> None:
        edges_to_remove = list(self._edge_adjacency.get(node_id, []))
        for eid in edges_to_remove:
            self._remove_edge_unlocked(eid)
            try:
                self._edge_order.remove(eid)
            except ValueError:
                pass
        self._nodes.pop(node_id, None)
        self._adjacency.pop(node_id, None)
        self._edge_adjacency.pop(node_id, None)
        for nid, neighbors in self._adjacency.items():
            neighbors.discard(node_id)
            if nid in self._nodes:
                self._nodes[nid].degree = len(neighbors)

    def _remove_edge_unlocked(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge:
            src, tgt = edge.source_id, edge.target_id
            self._adjacency[src].discard(tgt)
            self._adjacency[tgt].discard(src)
            if src in self._edge_adjacency:
                self._edge_adjacency[src] = [e for e in self._edge_adjacency[src] if e != edge_id]
            if tgt in self._edge_adjacency:
                self._edge_adjacency[tgt] = [e for e in self._edge_adjacency[tgt] if e != edge_id]
            if src in self._nodes:
                self._nodes[src].degree = len(self._adjacency[src])
            if tgt in self._nodes:
                self._nodes[tgt].degree = len(self._adjacency[tgt])

    def _compute_betweenness_centrality(self, nodes: Dict, adj: Dict) -> Dict[str, float]:
        if not nodes:
            return {}
        n = len(nodes)
        if n < 2:
            return {nid: 0.0 for nid in nodes}

        centrality = {nid: 0.0 for nid in nodes}
        node_list = list(nodes.keys())

        sample = node_list if n <= 100 else list(node_list)[:100]

        for src in sample:
            visited = {src}
            queue = deque([src])
            predecessors = {nid: [] for nid in nodes}
            distances = {nid: -1 for nid in nodes}
            distances[src] = 0
            sigma = {nid: 0 for nid in nodes}
            sigma[src] = 1

            while queue:
                current = queue.popleft()
                for neighbor in adj.get(current, set()):
                    if distances[neighbor] < 0:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
                    if distances[neighbor] == distances[current] + 1:
                        sigma[neighbor] += sigma[current]
                        predecessors[neighbor].append(current)

            delta = {nid: 0.0 for nid in nodes}
            stack = sorted(
                [nid for nid in nodes if distances[nid] >= 0],
                key=lambda x: distances[x],
                reverse=True,
            )
            for nid in stack:
                for pred in predecessors[nid]:
                    if sigma[nid] > 0:
                        delta[pred] += (sigma[pred] / sigma[nid]) * (1.0 + delta[nid])
                if nid != src:
                    centrality[nid] += delta[nid]

        max_centrality = max(centrality.values()) if centrality else 1.0
        if max_centrality > 0:
            for nid in centrality:
                centrality[nid] = centrality[nid] / max_centrality

        return centrality

    def _compute_clustering_coefficients(self, nodes: Dict, adj: Dict) -> Dict[str, float]:
        result = {}
        for nid in nodes:
            neighbors = adj.get(nid, set())
            k = len(neighbors)
            if k < 2:
                result[nid] = 0.0
                continue
            possible_edges = k * (k - 1) // 2
            actual_edges = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    a, b = neighbor_list[i], neighbor_list[j]
                    if b in adj.get(a, set()):
                        actual_edges += 1
            result[nid] = actual_edges / possible_edges if possible_edges > 0 else 0.0
        return result

    def _compute_avg_clustering(self, nodes: Dict, clustering: Dict[str, float]) -> float:
        if not nodes:
            return 0.0
        return sum(clustering.get(nid, 0.0) for nid in nodes) / len(nodes)

    def _compute_avg_path_length(self, nodes: Dict, adj: Dict) -> float:
        if not nodes or len(nodes) < 2:
            return 0.0

        total_length = 0
        path_count = 0
        node_list = list(nodes.keys())
        sample = node_list if len(node_list) <= 50 else list(node_list)[:50]

        for src in sample:
            visited = {src}
            queue = deque([(src, 0)])
            while queue:
                current, dist = queue.popleft()
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nd = dist + 1
                        total_length += nd
                        path_count += 1
                        queue.append((neighbor, nd))

        return total_length / path_count if path_count > 0 else 0.0

    def _compute_small_world_coefficient(self, nodes: Dict, adj: Dict,
                                          avg_clustering: float,
                                          avg_path_length: float) -> float:
        n = len(nodes)
        if n < 2:
            return 0.0

        total_edges = sum(len(v) for v in adj.values()) // 2
        p = (2 * total_edges) / (n * (n - 1)) if n > 1 else 0.0

        if p <= 0:
            return 0.0

        random_clustering = p
        k = max(2 * total_edges / n, 1.0)
        random_path = 1.0 if n <= 1 else max(1.0, math.log(n) / max(math.log(k), 0.001))

        if random_clustering <= 0 or random_path <= 0:
            return 0.0

        gamma = avg_clustering / random_clustering
        lambda_val = avg_path_length / random_path if random_path > 0 else 1.0

        small_world = (gamma / lambda_val) if lambda_val > 0 else 0.0
        return _clamp(small_world / max(n, 1))

    def _evaluate_topology_score(self, nodes: Dict,
                                   avg_clustering: float,
                                   avg_path_length: float,
                                   adj: Dict = None) -> TopologyScore:
        n = len(nodes)
        if n == 0:
            return TopologyScore.FRAGMENTED

        adj_local = adj if adj is not None else {}
        connected_count = 0
        for nid in nodes:
            if len(adj_local.get(nid, set())) > 0:
                connected_count += 1
        connectivity_ratio = connected_count / n if n > 0 else 0.0

        score_val = (connectivity_ratio * 0.3 +
                     avg_clustering * 0.4 +
                     (1.0 - min(avg_path_length / max(n, 1), 1.0)) * 0.3)
        score_val = _clamp(score_val)

        if score_val >= 0.8:
            return TopologyScore.INTEGRATED
        elif score_val >= 0.6:
            return TopologyScore.CONNECTED
        elif score_val >= 0.4:
            return TopologyScore.MODERATE
        elif score_val >= 0.2:
            return TopologyScore.LOOSE
        else:
            return TopologyScore.FRAGMENTED

    def _find_connected_components(self, nodes: Dict, adj: Dict) -> List[Set[str]]:
        visited: Set[str] = set()
        components = []
        for nid in nodes:
            if nid in visited:
                continue
            component = set()
            queue = deque([nid])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
        return components

    def _compute_cluster_density(self, cluster: Set[str], adj: Dict) -> float:
        if len(cluster) < 2:
            return 0.0
        node_list = list(cluster)
        possible = len(node_list) * (len(node_list) - 1) // 2
        actual = 0
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                if node_list[j] in adj.get(node_list[i], set()):
                    actual += 1
        return actual / possible if possible > 0 else 0.0

    def _find_bridges_internal(self, nodes: Dict, adj: Dict) -> List[str]:
        if not nodes:
            return []

        components = self._find_connected_components(nodes, adj)
        if len(components) <= 1:
            return self._find_structural_bridges(nodes, adj)

        bridges = set()
        for i, comp_a in enumerate(components):
            for nid_a in comp_a:
                for nid_b in nodes:
                    if nid_b in comp_a:
                        continue
                    if nid_a in adj.get(nid_b, set()):
                        bridges.add(nid_a)
                        bridges.add(nid_b)
        return list(bridges)

    def _find_structural_bridges(self, nodes: Dict, adj: Dict) -> List[str]:
        if len(nodes) < 3:
            return list(nodes.keys())

        bridges = []
        disc = {}
        low = {}
        parent = {}
        timer = [0]

        def _dfs(u):
            disc[u] = timer[0]
            low[u] = timer[0]
            timer[0] += 1
            children = 0
            for v in adj.get(u, set()):
                if v not in disc:
                    parent[v] = u
                    children += 1
                    _dfs(v)
                    low[u] = min(low[u], low[v])
                    if u in parent and low[v] >= disc[u]:
                        bridges.append(u)
                    elif u not in parent and children > 1:
                        bridges.append(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for nid in nodes:
            if nid not in disc:
                _dfs(nid)

        return list(set(bridges))

    def _find_bottlenecks_internal(self, nodes: Dict,
                                    centrality: Dict[str, float]) -> List[str]:
        if not centrality:
            return []
        threshold = 0.3
        return [nid for nid, cent in centrality.items() if cent >= threshold and nid in nodes]

    def _shortest_path_length(self, adj: Dict, source: str, target: str) -> Optional[int]:
        if source not in adj and source not in {n for neighbors in adj.values() for n in neighbors}:
            return None
        if target not in adj and target not in {n for neighbors in adj.values() for n in neighbors}:
            return None
        if source == target:
            return 0
        visited = {source}
        queue = deque([(source, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None