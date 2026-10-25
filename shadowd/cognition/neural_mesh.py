#!/usr/bin/env python3
"""
NeuralMesh - Peer-to-Peer Distributed Shadow Communication

Implements a mesh network for Shadow AI nodes to communicate across
devices (phone, laptop, desktop, server). Uses the existing Tailscale
infrastructure for secure transport.

Architecture:
  NeuralMesh
  ├── MeshNode (local peer identity)
  ├── MeshDiscovery (find other Shadow nodes on network)
  ├── MeshTransport (Tailscale + HTTP for message passing)
  ├── MeshSync (sync state across devices)
  ├── MeshGossip (propagate insights across the mesh)
  └── MeshConsensus (simple consensus for distributed decisions)

Design Principles:
- Eventual consistency (not strong consistency)
- Gossip protocol for dissemination
- Tailscale for secure transport
- Each node is autonomous but cooperative
- Works offline, syncs when connected

Inspired by: libp2p, GossipSub, Riak's anti-entropy, and
the Tailscale blog on mesh networking.
"""

import asyncio
import copy
import hashlib
import threading
import hmac as _hmac
import json
import os
import re
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("shadowd.cognition.neural_mesh")


class MeshMessageType(str, Enum):
    PING = "ping"
    PONG = "pong"
    GOSSIP = "gossip"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    INSIGHT_SHARE = "insight_share"
    TASK_DELEGATE = "task_delegate"
    TASK_RESULT = "task_result"
    STATE_UPDATE = "state_update"
    ELECTION = "election"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    LEAVE = "leave"


class NodeStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DORMANT = "dormant"
    SYNCING = "syncing"


@dataclass
class MeshNode:
    node_id: str = ""
    name: str = ""
    device_type: str = "server"
    address: str = "127.0.0.1"
    port: int = 8765
    status: NodeStatus = NodeStatus.ONLINE
    capabilities: List[str] = field(default_factory=list)
    last_heartbeat: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trust_level: float = 1.0
    regions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"node_{uuid.uuid4().hex[:12]}"

    @property
    def endpoint(self) -> str:
        return f"http://{self.address}:{self.port}"

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "device_type": self.device_type,
            "address": self.address,
            "port": self.port,
            "status": self.status.value,
            "capabilities": list(self.capabilities),
            "last_heartbeat": self.last_heartbeat,
            "trust_level": round(self.trust_level, 3),
        }

    def is_alive(self, timeout_seconds: int = 300) -> bool:
        if self.status == NodeStatus.OFFLINE:
            return False
        try:
            last = datetime.fromisoformat(self.last_heartbeat)
            now = datetime.now(timezone.utc)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return (now - last).total_seconds() < timeout_seconds
        except (ValueError, TypeError):
            return False


@dataclass
class MeshMessage:
    message_id: str = ""
    message_type: MeshMessageType = MeshMessageType.GOSSIP
    sender_id: str = ""
    target_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    ttl: int = 10
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str = ""
    priority: int = 5

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:12]}_{int(time.time())}"

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "payload_keys": list(self.payload.keys()),
            "ttl": self.ttl,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }

    def fingerprint(self) -> str:
        data = f"{self.message_type.value}:{self.sender_id}:{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def sign(self, secret: str) -> str:
        data = f"{self.message_id}:{self.fingerprint()}:{self.timestamp}"
        sig = _hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
        self.signature = sig
        return sig

    def verify(self, secret: str) -> bool:
        if not self.signature:
            return False
        data = f"{self.message_id}:{self.fingerprint()}:{self.timestamp}"
        expected = _hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
        return _hmac.compare_digest(self.signature, expected)


class GossipProtocol:
    MAX_SEEN_MESSAGES = 10000

    def __init__(self, fanout: int = 3, max_transmissions: int = 10, rate_limit_per_second: int = 50):
        self.fanout = fanout
        self.max_transmissions = max_transmissions
        self._seen_messages: Dict[str, int] = {}
        self._transmission_count: Dict[str, int] = defaultdict(int)
        self._rate_limiter: List[float] = []
        self._rate_limit_per_second = rate_limit_per_second
        self._lock = threading.Lock()

    def _check_rate_limit(self) -> bool:
        now = time.monotonic()
        self._rate_limiter = [t for t in self._rate_limiter if now - t < 1.0]
        if len(self._rate_limiter) >= self._rate_limit_per_second:
            return False
        self._rate_limiter.append(now)
        return True

    def should_propagate(self, message: MeshMessage) -> bool:
        with self._lock:
            if not self._check_rate_limit():
                return False
            fp = message.fingerprint()
            if fp in self._seen_messages:
                return False

            self._seen_messages[fp] = int(time.monotonic())
            if len(self._seen_messages) > self.MAX_SEEN_MESSAGES:
                oldest = sorted(self._seen_messages.items(), key=lambda x: x[1])[:self.MAX_SEEN_MESSAGES // 2]
                for k, _ in oldest:
                    del self._seen_messages[k]
            self._transmission_count[fp] += 1

            if self._transmission_count[fp] > self.max_transmissions:
                return False

            if message.ttl <= 0:
                return False

            return True

    def select_peers(self, peers: List[MeshNode], exclude: Optional[Set[str]] = None) -> List[MeshNode]:
        available = [p for p in peers if p.is_alive() and (not exclude or p.node_id not in exclude)]
        available.sort(key=lambda p: (p.trust_level, p.last_heartbeat), reverse=True)
        return available[:self.fanout]

    def cleanup(self, max_age_seconds: int = 3600):
        with self._lock:
            now = int(time.monotonic())
            expired = [fp for fp, ts in self._seen_messages.items() if now - ts > max_age_seconds]
            for fp in expired:
                del self._seen_messages[fp]
                self._transmission_count.pop(fp, None)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "seen_messages": len(self._seen_messages),
                "active_propagations": sum(1 for c in self._transmission_count.values() if c < self.max_transmissions),
                "fanout": self.fanout,
            }


class StateSync:
    MAX_CONFLICTS = 100

    def __init__(self):
        self._local_state: Dict[str, Dict] = {}
        self._vector_clock: Dict[str, int] = {}
        self._conflicts: List[Dict] = []
        self._lock = threading.Lock()

    def update_local(self, key: str, value: Any, version: Optional[int] = None):
        with self._lock:
            if version is not None:
                self._vector_clock[key] = version
            else:
                self._vector_clock[key] = self._vector_clock.get(key, 0) + 1
            self._local_state[key] = {
                "value": value,
                "version": self._vector_clock[key],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_local(self, key: str) -> Optional[Any]:
        with self._lock:
            state = self._local_state.get(key)
            return copy.deepcopy(state["value"]) if state else None

    def merge_remote(self, key: str, remote_state: Dict, remote_node_id: str) -> bool:
        with self._lock:
            local = self._local_state.get(key)
            local_version = local["version"] if local else 0
            remote_version = remote_state.get("version", 0)

            if remote_version > local_version:
                self._local_state[key] = remote_state
                self._vector_clock[key] = remote_version
                return True
            elif remote_version == local_version and local:
                if json.dumps(remote_state.get("value")) != json.dumps(local.get("value")):
                    self._conflicts.append({
                        "key": key,
                        "local": local,
                        "remote": remote_state,
                        "remote_node_id": remote_node_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    if len(self._conflicts) > self.MAX_CONFLICTS:
                        self._conflicts = self._conflicts[-self.MAX_CONFLICTS:]
                return False
            return False

    def get_sync_payload(self, since_version: int = 0) -> Dict:
        with self._lock:
            updates = {}
            for key, state in self._local_state.items():
                if state.get("version", 0) > since_version:
                    updates[key] = state
            result = {
                "updates": updates,
                "vector_clock": self._vector_clock,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return copy.deepcopy(result)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "keys": len(self._local_state),
                "total_version": sum(self._vector_clock.values()),
                "conflicts": len(self._conflicts),
            }


class MeshConsensus:
    _MAX_PROPOSALS = 1000

    def __init__(self, node_id: str, peers: Dict[str, MeshNode]):
        self.node_id = node_id
        self._peers = peers
        self._proposals: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def _valid_proposer(self, proposer_id: str) -> bool:
        return proposer_id == self.node_id or proposer_id in self._peers

    async def propose(self, key: str, value: Any) -> Dict:
        with self._lock:
            proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
            proposal = {
                "proposal_id": proposal_id,
                "key": key,
                "value": value,
                "proposer": self.node_id,
                "votes": {self.node_id: True},
                "status": "voting",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._proposals[proposal_id] = proposal
            if len(self._proposals) > self._MAX_PROPOSALS:
                oldest = sorted(self._proposals.items(), key=lambda x: x[1].get("created_at", ""))[:len(self._proposals) - self._MAX_PROPOSALS]
                for kid, _ in oldest:
                    del self._proposals[kid]
            return proposal

    async def vote(self, proposal_id: str, accept: bool) -> bool:
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if not proposal or proposal["status"] != "voting":
                return False
            proposal["votes"][self.node_id] = accept
            total_nodes = len(self._peers) + 1
            accept_count = sum(1 for v in proposal["votes"].values() if v)
            if accept_count > total_nodes / 2:
                proposal["status"] = "accepted"
                return True
            elif len(proposal["votes"]) > total_nodes / 2:
                proposal["status"] = "rejected"
            return False

    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        with self._lock:
            return copy.deepcopy(self._proposals.get(proposal_id))

    def get_stats(self) -> Dict:
        return {
            "proposals": len(self._proposals),
            "active": sum(1 for p in self._proposals.values() if p["status"] == "voting"),
        }


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def record_success(self):
        self._failure_count = 0
        self._state = self.CLOSED

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN

    def can_execute(self) -> bool:
        return self.state != self.OPEN


class NeuralMesh:
    """
    Peer-to-Peer Mesh Network for Shadow AI.

    Enables distributed shadow nodes (phone, laptop, desktop) to:
    - Discover each other on the Tailscale network
    - Share insights and observations via gossip
    - Sync state with eventual consistency
    - Make distributed decisions via simple consensus
    - Delegate tasks to the most capable node

    Works offline and syncs when reconnected.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        node_name: str = "shadow-local",
        device_type: str = "server",
        address: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.local_node = MeshNode(
            node_id=node_id or f"node_{uuid.uuid4().hex[:12]}",
            name=node_name,
            device_type=device_type,
            address=address,
            port=port,
            capabilities=self._detect_capabilities(device_type),
        )

        self._peers: Dict[str, MeshNode] = {}
        self._peer_health: Dict[str, Dict] = {}
        self._peers_lock = threading.Lock()
        self._gossip = GossipProtocol()
        self._sync = StateSync()
        self._consensus = MeshConsensus(self.local_node.node_id, self._peers)

        self._message_handlers: Dict[MeshMessageType, List[Callable]] = defaultdict(list)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._heartbeat_interval = 30.0

        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "gossip_propagations": 0,
            "syncs_completed": 0,
            "consensuses_reached": 0,
            "discoveries": 0,
        }
        self._stats_lock = threading.Lock()

        self._insight_buffer: List[Dict] = []
        self._MAX_INSIGHT_BUFFER = 100
        self._received_insights: List[Dict] = []
        self._MAX_RECEIVED_INSIGHTS = 500
        self._task_delegations: Dict[str, Dict] = {}
        self._MAX_TASK_DELEGATIONS = 1000
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._max_retries = 3
        self._retry_delay = 1.0
        self._mesh_secret: str = os.environ.get("SHADOW_MESH_SECRET", "")
        self._peer_trust_threshold: float = 0.3

    def _detect_capabilities(self, device_type: str) -> List[str]:
        base = ["shadow_ai", "observe", "reason", "respond"]
        device_caps = {
            "phone": ["sensors", "bluetooth", "wifi", "gps", "camera", "microphone"],
            "laptop": ["compute", "filesystem", "terminal", "clipboard", "browser"],
            "server": ["compute_heavy", "database", "models", "storage", "gpu"],
            "desktop": ["compute", "filesystem", "terminal", "multitask"],
        }
        return base + device_caps.get(device_type, [])

    @staticmethod
    def _sanitize_message(data: Dict) -> Dict:
        if not isinstance(data, dict):
            return {}
        sanitized = {}
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            k = re.sub(r'[<>\x00-\x1f\x7f]', '', k).strip()[:100]
            if isinstance(v, str):
                v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v).strip()[:1000]
            elif isinstance(v, dict):
                v = NeuralMesh._sanitize_message(v)
            elif isinstance(v, list):
                v = v[:100]
            sanitized[k] = v
        return sanitized

    async def start(self):
        self._running = True
        self.local_node.status = NodeStatus.ONLINE
        self._tasks = []
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._message_processor()))
        logger.info("NeuralMesh started: %s", self.local_node.node_id)

    async def stop(self):
        self._running = False
        self.local_node.status = NodeStatus.OFFLINE
        for task in getattr(self, '_tasks', []):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks = []
        await self.broadcast(MeshMessage(
            message_type=MeshMessageType.LEAVE,
            sender_id=self.local_node.node_id,
        ))
        logger.info("NeuralMesh stopped: %s", self.local_node.node_id)

    async def _heartbeat_loop(self):
        while self._running:
            try:
                self.local_node.last_heartbeat = datetime.now(timezone.utc).isoformat()
                await self.broadcast(MeshMessage(
                    message_type=MeshMessageType.HEARTBEAT,
                    sender_id=self.local_node.node_id,
                    payload={"status": self.local_node.status.value, "capabilities": self.local_node.capabilities},
                ))
                self._gossip.cleanup()
            except Exception as e:
                logger.error("Heartbeat error: %s", type(e).__name__)
            await asyncio.sleep(self._heartbeat_interval)

    async def _message_processor(self):
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                if self._mesh_secret and not message.verify(self._mesh_secret):
                    logger.warning("Message %s failed signature verification (or unsigned)", message.message_id[:16])
                    continue
                if message.sender_id != self.local_node.node_id:
                    peer = self._peers.get(message.sender_id)
                    if peer and peer.trust_level < self._peer_trust_threshold:
                        logger.warning("Low trust peer %s: %s", message.sender_id.replace(chr(10), ' ').replace(chr(13), ' '), round(peer.trust_level, 3))
                        continue
                handlers = self._message_handlers.get(message.message_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                        logger.error("Message handler error: %s", type(e).__name__)
                if message.message_type == MeshMessageType.INSIGHT_SHARE:
                    insight = message.payload.get("insight", {})
                    sanitized = self._sanitize_message(insight) if isinstance(insight, dict) else {"data": str(insight)[:500]}
                    self._received_insights.append(sanitized)
                    if len(self._received_insights) > self._MAX_RECEIVED_INSIGHTS:
                        self._received_insights = self._received_insights[-self._MAX_RECEIVED_INSIGHTS:]
                with self._stats_lock:
                    self._stats["messages_received"] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Message processor error: %s", type(e).__name__)

    def on_message(self, message_type: MeshMessageType, handler: Callable):
        self._message_handlers[message_type].append(handler)

    async def send(self, target_id: str, message: MeshMessage) -> bool:
        message.target_id = target_id
        message.sender_id = self.local_node.node_id
        with self._stats_lock:
            self._stats["messages_sent"] += 1

        peer = self._peers.get(target_id)
        if peer:
            self.update_peer_health(target_id, success=True)
            return True
        self.update_peer_health(target_id, success=False)
        return False

    async def broadcast(self, message: MeshMessage) -> int:
        message.sender_id = self.local_node.node_id
        if message.message_type == MeshMessageType.GOSSIP:
            if not self._gossip.should_propagate(message):
                return 0

        peers = self._gossip.select_peers(
            list(self._peers.values()),
            exclude={self.local_node.node_id}
        )

        sent = 0
        for peer in peers:
            try:
                await self.send(peer.node_id, message)
                sent += 1
            except (ConnectionError, OSError, TimeoutError, RuntimeError):
                pass

        if message.message_type == MeshMessageType.GOSSIP:
            with self._stats_lock:
                self._stats["gossip_propagations"] += sent

        return sent

    async def share_insight(self, insight: Dict) -> int:
        sanitized = self._sanitize_message(insight)
        self._insight_buffer.append(sanitized)
        if len(self._insight_buffer) > self._MAX_INSIGHT_BUFFER:
            self._insight_buffer = self._insight_buffer[-self._MAX_INSIGHT_BUFFER:]
        return await self._broadcast_with_retry(MeshMessage(
            message_type=MeshMessageType.INSIGHT_SHARE,
            payload={"insight": sanitized},
        ))

    async def _broadcast_with_retry(self, message: MeshMessage, retries: int = None) -> int:
        retries = retries or self._max_retries
        cb = self._circuit_breakers.setdefault("broadcast", CircuitBreaker())
        for attempt in range(retries):
            if not cb.can_execute():
                await asyncio.sleep(self._retry_delay)
                continue
            try:
                result = await self.broadcast(message)
                cb.record_success()
                return result
            except (ConnectionError, OSError, TimeoutError, RuntimeError) as e:
                cb.record_failure()
                logger.warning("Broadcast attempt %s/%s failed: %s", attempt+1, retries, type(e).__name__)
                if attempt < retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
        logger.error("Broadcast failed after %s retries, circuit breaker open", retries)
        return 0

    async def delegate_task(self, task_type: str, task_data: Dict, target_capabilities: Optional[List[str]] = None) -> Optional[str]:
        best_node = self._find_best_node(task_type, target_capabilities)
        if not best_node:
            return None

        delegation_id = f"deleg_{uuid.uuid4().hex[:12]}"
        if len(self._task_delegations) >= self._MAX_TASK_DELEGATIONS:
            oldest = sorted(self._task_delegations.items(), key=lambda x: x[1].get("created_at", ""))[:self._MAX_TASK_DELEGATIONS // 2]
            for kid, _ in oldest:
                del self._task_delegations[kid]
        self._task_delegations[delegation_id] = {
            "task_type": task_type,
            "target_node": best_node.node_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.send(best_node.node_id, MeshMessage(
            message_type=MeshMessageType.TASK_DELEGATE,
            payload={"delegation_id": delegation_id, "task_type": task_type, "task_data": task_data},
        ))
        return delegation_id

    def _find_best_node(self, task_type: str, required_capabilities: Optional[List[str]] = None) -> Optional[MeshNode]:
        candidates = [self.local_node] + list(self._peers.values())
        candidates = [n for n in candidates if n.is_alive()]

        if required_capabilities:
            candidates = [n for n in candidates if all(c in n.capabilities for c in required_capabilities)]

        if task_type == "compute_heavy":
            candidates.sort(key=lambda n: n.trust_level, reverse=True)
            for n in candidates:
                if n.device_type == "server":
                    return n

        if task_type == "sensors":
            for n in candidates:
                if n.device_type == "phone":
                    return n

        candidates.sort(key=lambda n: n.trust_level, reverse=True)
        return candidates[0] if candidates else None

    def add_peer(self, node: MeshNode):
        if not node.node_id or len(node.node_id) > 200:
            raise ValueError("Invalid node_id")
        if node.port < 1 or node.port > 65535:
            raise ValueError(f"Invalid port: {node.port}")
        _VALID_DEVICE_TYPES = {"phone", "laptop", "server", "desktop", "tablet", "iot"}
        if node.device_type not in _VALID_DEVICE_TYPES:
            raise ValueError(f"Invalid device_type: {node.device_type}")
        if len(node.name) > 100:
            node.name = node.name[:100]
        validated_capabilities = [c for c in node.capabilities if isinstance(c, str) and len(c) < 100][:50]
        node.capabilities = validated_capabilities
        with self._peers_lock:
            self._peers[node.node_id] = node
            self._peer_health[node.node_id] = {
                "last_seen": time.monotonic(),
                "failures": 0,
                "successes": 0,
                "latency_ms": 0.0,
            }
            self._consensus = MeshConsensus(self.local_node.node_id, self._peers)

    def remove_peer(self, node_id: str):
        with self._peers_lock:
            self._peers.pop(node_id, None)
            self._peer_health.pop(node_id, None)
            self._consensus = MeshConsensus(self.local_node.node_id, self._peers)

    _MAX_PEER_HEALTH = 200

    def update_peer_health(self, node_id: str, success: bool, latency_ms: float = 0.0):
        with self._peers_lock:
            if node_id not in self._peers and node_id not in self._peer_health:
                return
            if node_id not in self._peer_health:
                self._peer_health[node_id] = {
                    "last_seen": time.monotonic(),
                    "failures": 0,
                    "successes": 0,
                    "latency_ms": 0.0,
                }
            health = self._peer_health[node_id]
            health["last_seen"] = time.monotonic()
            if success:
                health["successes"] += 1
            else:
                health["failures"] += 1
            if latency_ms > 0:
                health["latency_ms"] = latency_ms
            if len(self._peer_health) > self._MAX_PEER_HEALTH:
                oldest = sorted(self._peer_health.items(), key=lambda x: x[1].get("last_seen", 0))[:len(self._peer_health) - self._MAX_PEER_HEALTH]
                for kid, _ in oldest:
                    del self._peer_health[kid]

    def get_mesh_health(self) -> Dict[str, Dict]:
        with self._peers_lock:
            snapshot = dict(self._peer_health)
        result = {}
        for node_id, health in snapshot.items():
            successes = health["successes"]
            failures = health["failures"]
            total = successes + failures
            health_score = (successes / total * 100) if total > 0 else 0.0
            result[node_id] = {
                "health_score": round(health_score, 2),
                "last_seen": health["last_seen"],
                "successes": successes,
                "failures": failures,
            }
        return result

    async def sync_with_peer(self, peer_id: str) -> Dict:
        peer = self._peers.get(peer_id)
        if not peer:
            return {"error": f"Peer {peer_id} not found"}

        sync_payload = self._sync.get_sync_payload()
        with self._stats_lock:
            self._stats["syncs_completed"] += 1

        return {
            "peer_id": peer_id,
            "sync_payload": sync_payload,
            "local_keys": len(sync_payload.get("updates", {})),
        }

    def update_local_state(self, key: str, value: Any):
        self._sync.update_local(key, value)

    def update_peer_status(self, node_id: str, status: str, capabilities: Optional[List[str]] = None) -> bool:
        with self._peers_lock:
            peer = self._peers.get(node_id)
            if peer is None:
                return False
            try:
                peer.status = NodeStatus(status)
            except (ValueError, TypeError):
                peer.status = NodeStatus.ONLINE
            if capabilities is not None:
                peer.capabilities = [c for c in capabilities if isinstance(c, str) and len(c) < 100][:50]
            peer.last_heartbeat = datetime.now(timezone.utc).isoformat()
            return True

    def get_local_state(self, key: str) -> Optional[Any]:
        return self._sync.get_local(key)

    async def propose_consensus(self, key: str, value: Any) -> Dict:
        proposal = await self._consensus.propose(key, value)
        await self.broadcast(MeshMessage(
            message_type=MeshMessageType.ELECTION,
            payload={"proposal": proposal},
        ))
        return proposal

    async def receive_proposal(self, proposal: Dict) -> bool:
        proposer = proposal.get("proposer")
        if not self._consensus._valid_proposer(proposer):
            logger.warning("Invalid proposer: %s", str(proposer).replace(chr(10), ' ').replace(chr(13), ' '))
            return False
        proposal_id = proposal.get("proposal_id")
        if proposal_id:
            self._consensus._proposals[proposal_id] = proposal
        return True

    async def vote_consensus(self, proposal_id: str, accept: bool) -> bool:
        result = await self._consensus.vote(proposal_id, accept)
        if result:
            with self._stats_lock:
                self._stats["consensuses_reached"] += 1
        return result

    def get_peers(self) -> List[Dict]:
        with self._peers_lock:
            return [copy.deepcopy(p.to_dict()) for p in self._peers.values()]

    def get_alive_peers(self) -> List[Dict]:
        with self._peers_lock:
            return [copy.deepcopy(p.to_dict()) for p in self._peers.values() if p.is_alive()]

    def get_recent_insights(self, limit: int = 20) -> List[Dict]:
        with self._peers_lock:
            return copy.deepcopy(self._insight_buffer[-limit:])

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats_copy = {**dict(self._stats)}
        with self._peers_lock:
            peers_count = len(self._peers)
            alive_count = len([p for p in self._peers.values() if p.is_alive()])
        return {
            "local_node": self.local_node.to_dict(),
            "peers_count": peers_count,
            "alive_peers": alive_count,
            "messages_queue_size": self._message_queue.qsize(),
            "gossip_stats": self._gossip.get_stats(),
            "sync_stats": self._sync.get_stats(),
            "consensus_stats": self._consensus.get_stats(),
            "stats": stats_copy,
        }