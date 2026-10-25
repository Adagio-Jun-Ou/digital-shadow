#!/usr/bin/env python3
"""
Digital Shadow Daemon - The unified Shadow AI that orchestrates all 10 abilities.
"""

import asyncio
import copy
import json
import os
import sys
import threading
import time as _time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from .websocket_manager import (
    WebSocketServer,
    ConnectionManager,
    WSMessageType,
    WSMessage,
)

from .security import SecurityManager, SecurityConfig

# MCP Server import
try:
    from .mcp_server_enhanced import EnhancedMCPServer
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from .config import get_config
import re as _re

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(os.environ.get("SHADOW_DATA_DIR", PROJECT_ROOT / "data"))
LOG_DIR = DATA_DIR / "logs"
MEMORY_DIR = DATA_DIR / "memory"

SHADOW_CORS_ORIGINS = os.environ.get(
    "SHADOW_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8765,http://127.0.0.1:8765"
).split(",")

SHADOW_AUTH_ENABLED = os.environ.get("SHADOW_AUTH_ENABLED", "false").lower() == "true"

if not SHADOW_AUTH_ENABLED:
    import warnings
    warnings.warn("SHADOW_AUTH_ENABLED is false - all API endpoints are unauthenticated. Set SHADOW_AUTH_ENABLED=true for production.", RuntimeWarning)

for d in [LOG_DIR, MEMORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _build_log_handlers():
    """Choose log format from SHADOW_LOG_JSON env var.
    JSON format is one log record per line — friendly for log shippers."""
    if os.environ.get("SHADOW_LOG_JSON", "").lower() in ("1", "true", "yes"):
        import json as _json
        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return _json.dumps(payload, ensure_ascii=False)
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    file_h = logging.FileHandler(LOG_DIR / "shadowd.log")
    stream_h = logging.StreamHandler(sys.stdout)
    file_h.setFormatter(formatter)
    stream_h.setFormatter(formatter)
    return [file_h, stream_h]


_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=_log_level, handlers=_build_log_handlers(), force=True)
logger = logging.getLogger("shadowd")


class ShadowContext:
    _MAX_OBSERVATIONS = 1000
    _MAX_DEVICES = 100
    _MAX_DICT_KEYS = 50
    _MAX_CONTENT_SIZE = 65536

    def __init__(self):
        self._lock = threading.Lock()
        self.observations: list = []
        self.current_activity: Optional[dict] = None
        self.connected_devices: dict = {}
        self.shadow_ai = None
        self.shikigami = None  # set during startup by lifespan
        self.ws_server: Optional[WebSocketServer] = None
        self.ws_manager: Optional[ConnectionManager] = None
        self.mcp_server: Optional[EnhancedMCPServer] = None
        self.orchestrator = None

    async def learn(self, observation: dict):
        obs = {
            **observation,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if isinstance(obs.get("content"), dict):
            if len(obs["content"]) > self._MAX_DICT_KEYS:
                obs["content"] = dict(list(obs["content"].items())[:self._MAX_DICT_KEYS])
            for k in list(obs["content"].keys()):
                sk = _sanitize(k, 100)
                if sk != k:
                    obs["content"][sk] = obs["content"].pop(k)
                if isinstance(obs["content"].get(sk), str):
                    obs["content"][sk] = _sanitize_content(obs["content"][sk], self._MAX_CONTENT_SIZE)
        if isinstance(obs.get("type"), str):
            obs["type"] = _sanitize(obs["type"], 50)
        with self._lock:
            self.observations.append(obs)
            if len(self.observations) > self._MAX_OBSERVATIONS:
                self.observations = self.observations[-self._MAX_OBSERVATIONS:]

        if self.shadow_ai:
            await self.shadow_ai.observe(observation)

    async def broadcast_observation(self, observation: dict):
        if self.ws_server:
            msg = WSMessage(
                type=WSMessageType.OBSERVE,
                data={"observation": observation}
            )
            await self.ws_server.manager.broadcast(msg)

    async def broadcast_insight(self, insight: dict):
        if self.ws_server:
            msg = WSMessage(
                type=WSMessageType.INSIGHT,
                data={"insight": insight}
            )
            await self.ws_server.manager.broadcast_to_topic("insights", msg)

    def get_observations(self, limit: int = 20):
        with self._lock:
            return copy.deepcopy(self.observations[-limit:])

    def get_current_activity(self):
        with self._lock:
            return copy.deepcopy(self.current_activity) if self.current_activity else None

    def set_current_activity(self, activity: Optional[dict]):
        with self._lock:
            if activity is not None and isinstance(activity, dict):
                activity = {_sanitize_content(str(k), 100): _sanitize_content(str(v), 1000) for k, v in activity.items()}
            self.current_activity = activity

    def connect_device(self, device_id: str, name: str):
        device_id_sanitized = _sanitize(device_id, 64)
        name_sanitized = _sanitize_content(name, 100)
        with self._lock:
            if len(self.connected_devices) >= self._MAX_DEVICES:
                oldest_key = next(iter(self.connected_devices))
                del self.connected_devices[oldest_key]
            self.connected_devices[device_id_sanitized] = {
                "name": name_sanitized,
                "connected_at": datetime.now(timezone.utc).isoformat()
            }
        return len(self.connected_devices)

    def get_connected_devices(self):
        with self._lock:
            return dict(self.connected_devices)

    @property
    def observation_count(self):
        with self._lock:
            return len(self.observations)


shadow = ShadowContext()
security = None
_background_tasks: list = []
_start_time = 0.0
_login_rate_limits: dict = {}  # ip -> [timestamps]
_login_rate_lock = threading.Lock()


async def auth_dependency(request: Request):
    """Optional authentication dependency."""
    if not SHADOW_AUTH_ENABLED or not security:
        return None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = security.verify_token(token)
        if payload:
            return payload
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        key_obj = security.verify_api_key(api_key)
        if key_obj:
            return {"user_id": key_obj.user_id, "scopes": key_obj.scopes}
    raise HTTPException(status_code=401, detail="Authentication required")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = _time.time()
    logger.info("=" * 50)
    logger.info("Shadow AI Daemon Starting...")
    logger.info("Data Directory: %s", DATA_DIR)
    logger.info("Auth enabled: %s", SHADOW_AUTH_ENABLED)
    logger.info("=" * 50)

    config = get_config()
    logger.info("Config loaded: model=%s, port=%s", config.llm.model, config.server.port)

    security_config = SecurityConfig()
    if SHADOW_AUTH_ENABLED:
        global security
        security = SecurityManager(security_config)
        admin_pw = os.environ.get("SHADOW_ADMIN_PASSWORD")
        if not admin_pw:
            admin_pw = os.urandom(24).hex()
            logger.warning("=" * 60)
            logger.warning("AUTO-GENERATED ADMIN PASSWORD (save this!):")
            logger.warning("  %s", admin_pw)
            logger.warning("=" * 60)
        security.create_user(
            email="admin@shadow.local",
            username="admin",
            password=admin_pw,
            scopes=["read", "write", "admin"],
            is_superuser=True,
        )
        logger.info("Security initialized (auth required)")
    
    from .observers import start_all_observers
    from .mind import MindEngine
    from shikigami import get_registry
    from .shadow_ai import ShadowAI
    from .coordinator import AbilityCoordinator
    
    # Initialize WebSocket server
    shadow.ws_manager = ConnectionManager()
    shadow.ws_server = WebSocketServer(
        manager=shadow.ws_manager,
        enable_heartbeat=True,
        ping_interval=30.0,
        pong_timeout=10.0
    )
    
    # Set up WebSocket message handlers
    @shadow.ws_server.manager.on_message(WSMessageType.ASK)
    async def handle_ws_ask(client, message: WSMessage) -> WSMessage:
        if not shadow.shadow_ai:
            return WSMessage(
                type=WSMessageType.ERROR,
                data={"error": "Shadow AI not initialized", "message_id": message.message_id}
            )
        
        question = message.data.get("question", "")
        context = message.data.get("context", {})
        
        result = await shadow.shadow_ai.ask(question, context)
        
        return WSMessage(
            type=WSMessageType.ASK_RESPONSE,
            data={
                "message_id": message.message_id,
                "response": result.get("unified_response", ""),
                "ability_used": result.get("ability_used"),
                "insight": result.get("insight"),
            }
        )
    
    @shadow.ws_server.manager.on_message(WSMessageType.ASK_STREAM)
    async def handle_ws_ask_stream(client, message: WSMessage):
        if not shadow.shadow_ai:
            await shadow.ws_server.manager.send_to_client(
                client.client_id,
                WSMessage(
                    type=WSMessageType.ERROR,
                    data={"error": "Shadow AI not initialized", "message_id": message.message_id}
                )
            )
            return
        
        question = message.data.get("question", "")
        context = message.data.get("context", {})
        
        async def response_generator():
            result = await shadow.shadow_ai.ask(question, context)
            if isinstance(result.get("unified_response"), str):
                for word in result["unified_response"].split():
                    yield word + " "
            else:
                yield str(result.get("unified_response", ""))
        
        await shadow.ws_server.streaming.stream_response(
            client.client_id,
            message.message_id,
            response_generator()
        )
    
    @shadow.ws_server.manager.on_message(WSMessageType.AGENT_STATUS)
    async def handle_agent_status(client, message: WSMessage):
        if hasattr(shadow, 'coordinator') and shadow.coordinator:
            status = shadow.coordinator.get_ability_status()
            await shadow.ws_server.manager.send_to_client(
                client.client_id,
                WSMessage(
                    type=WSMessageType.AGENT_STATUS,
                    data={"status": status, "message_id": message.message_id}
                )
            )
        else:
            await shadow.ws_server.manager.send_to_client(
                client.client_id,
                WSMessage(
                    type=WSMessageType.ERROR,
                    data={"error": "Coordinator not initialized", "message_id": message.message_id}
                )
            )
    
    @shadow.ws_server.manager.on_message(WSMessageType.MEMORY_QUERY)
    async def handle_memory_query(client, message: WSMessage):
        limit = message.data.get("limit", 20)
        observations = shadow.get_observations(limit)
        await shadow.ws_server.manager.send_to_client(
            client.client_id,
            WSMessage(
                type=WSMessageType.MEMORY_RESPONSE,
                data={
                    "observations": observations,
                    "count": shadow.observation_count,
                    "message_id": message.message_id
                }
            )
        )
    
    # Start heartbeat
    await shadow.ws_server.start_heartbeat()
    
    # Initialize MCP Server
    if HAS_MCP:
        shadow.mcp_server = EnhancedMCPServer(
            name="Shadow AI",
            version="2.0.0",
            data_dir=DATA_DIR,
            websocket_manager=shadow.ws_manager,
            shadow_ai=None  # Will be set after Shadow AI initializes
        )
        logger.info("MCP Server initialized")
    else:
        logger.warning("MCP package not installed. MCP Server disabled.")
    
    observer_task = asyncio.create_task(start_all_observers(shadow))
    _background_tasks.append(observer_task)

    # Proactive insight engine: periodic scan for manipulation patterns,
    # sentiment shifts, anomalies. Broadcasts on /ws insights topic.
    if os.environ.get("SHADOW_PROACTIVE_DISABLED", "").lower() not in ("1", "true", "yes"):
        from .proactive_engine import run_proactive_loop
        proactive_task = asyncio.create_task(run_proactive_loop(shadow))
        _background_tasks.append(proactive_task)
        logger.info("Proactive insight engine enabled")

    shadow.mind = MindEngine()
    await shadow.mind.initialize()
    
    shadow.shikigami = get_registry()
    shadow.shadow_ai = ShadowAI(shadow.shikigami, shadow.mind, data_dir=str(DATA_DIR))
    shadow.coordinator = AbilityCoordinator(shadow.shadow_ai)
    
    # Wire event bus to broadcast insights
    from .event_bus import get_event_bus, EventType
    
    async def _broadcast_insights(event):
        """Broadcast critical insights via WebSocket."""
        if event.priority <= 2 and shadow.ws_server:
            from .websocket_manager import WSMessage, WSMessageType
            msg = WSMessage(type=WSMessageType.INSIGHT, data={
                "event_type": event.event_type.value,
                "source": event.source,
                "data": event.data,
                "priority": event.priority,
            })
            await shadow.ws_server.manager.broadcast(msg)
    
    event_bus = get_event_bus()
    event_bus.subscribe(EventType.INSIGHT, _broadcast_insights)
    event_bus.subscribe(EventType.STUCK, _broadcast_insights)
    event_bus.subscribe(EventType.ALERT, _broadcast_insights)
    
    # Initialize Super Orchestrator
    try:
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        shadow.orchestrator = SuperShadowOrchestrator(data_dir=DATA_DIR)
        logger.info("Super Orchestrator initialized")
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.warning("Super Orchestrator not available: %s", type(e).__name__)
        shadow.orchestrator = None
    
    # Connect Shadow AI to MCP server
    if shadow.mcp_server:
        shadow.mcp_server.shadow_ai = shadow.shadow_ai
    
    # Start proactive monitor
    proactive_task = asyncio.create_task(shadow.shadow_ai.run_proactive_monitor())
    _background_tasks.append(proactive_task)

    # Auto-compaction: periodically summarize old observations via LLM
    _compact_interval = float(os.environ.get("SHADOW_COMPACT_INTERVAL", "120"))
    shadow._compact_stop = asyncio.Event()

    async def _compact_loop():
        await asyncio.sleep(30)
        while not shadow._compact_stop.is_set():
            try:
                await asyncio.wait_for(shadow._compact_stop.wait(), timeout=_compact_interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                if shadow.shadow_ai and len(shadow.shadow_ai.observations) > 50:
                    with shadow.shadow_ai._lock:
                        old = shadow.shadow_ai.observations[:-50]
                        shadow.shadow_ai.observations = shadow.shadow_ai.observations[-50:]
                    if len(old) > 20:
                        summary = await shadow.mind.compact_context(
                            [{"content": json.dumps(o, default=str)[:500]} for o in old[-100:]]
                        )
                        compact_obs = {"type": "compacted_summary", "content": summary,
                                       "compacted_count": len(old),
                                       "timestamp": datetime.now(timezone.utc).isoformat()}
                        with shadow.shadow_ai._lock:
                            shadow.shadow_ai.observations.insert(0, compact_obs)
                        if shadow.shikigami and hasattr(shadow.shikigami, 'observations'):
                            try:
                                shadow.shikigami.observations.append(compact_obs)
                                with shadow.shikigami._lock:
                                    if len(shadow.shikigami.observations) > 5000:
                                        shadow.shikigami.observations = shadow.shikigami.observations[-5000:]
                            except (AttributeError, RuntimeError, TypeError):
                                pass
                        logger.info("Compacted %d observations via LLM summary", len(old))
            except (ValueError, RuntimeError, AttributeError, TypeError, OSError) as e:
                logger.warning("Compaction loop error: %s", type(e).__name__)

    compact_task = asyncio.create_task(_compact_loop())
    _background_tasks.append(compact_task)
    logger.info("Observation auto-compaction enabled (interval=%ss)", _compact_interval)
    
    logger.info("Shadow AI is now watching...")
    logger.info("10 unified abilities ready")
    logger.info("WebSocket server ready")
    if shadow.mcp_server:
        logger.info("MCP Server ready (run 'shadow-mcp' to start)")
    yield
    
    # Cleanup
    if shadow.shadow_ai and hasattr(shadow.shadow_ai, 'stop'):
        await shadow.shadow_ai.stop()
    if hasattr(shadow, '_compact_stop'):
        shadow._compact_stop.set()
    
    for task in _background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    _background_tasks.clear()
    
    if shadow.ws_server:
        await shadow.ws_server.stop_heartbeat()
        await shadow.ws_server.cleanup()
    logger.info("Shadow AI Daemon Shutting Down...")


app = FastAPI(
    title="Shadow AI",
    description="Unified AI that orchestrates 10 superhuman abilities for life optimization",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SHADOW_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

try:
    from .cognition.api_router import router as cognition_router
    app.include_router(cognition_router)
except ImportError as e:
    logger.warning("Cognitive module router not loaded: %s", type(e).__name__)


# ============ HEALTH CHECK ============

@app.get("/health")
async def health(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Health check endpoint for monitoring."""
    from .event_bus import get_event_bus
    
    checks = {"status": "healthy", "version": "1.0.0"}
    
    if _auth or not SHADOW_AUTH_ENABLED:
        checks["mind"] = "ok" if shadow.mind else "unavailable"
        checks["shikigami"] = "ok" if shadow.shikigami else "unavailable"
        checks["shadow_ai"] = "ok" if shadow.shadow_ai else "unavailable"
        checks["orchestrator"] = "ok" if shadow.orchestrator else "unavailable"
        checks["event_bus"] = "ok"
        checks["stuck_detector"] = "ok" if shadow.shadow_ai and hasattr(shadow.shadow_ai, 'stuck_detector') else "unavailable"
        checks["context_budget"] = "ok" if shadow.mind and hasattr(shadow.mind, 'context_budget') else "unavailable"
    else:
        checks["authenticated"] = False
    
    if shadow.mind and (_auth or not SHADOW_AUTH_ENABLED):
        try:
            # Attempt a lightweight API call to check connectivity
            client = await shadow.mind._get_client()
            checks["llm_client"] = "initialized" if not client.is_closed else "reconnecting"
        except (OSError, RuntimeError, AttributeError, TypeError):
            checks["llm_client"] = "error"
    
    # Event bus stats
    if _auth or not SHADOW_AUTH_ENABLED:
        event_bus = get_event_bus()
        eb_stats = event_bus.get_stats()
        checks["events_published"] = eb_stats.get("events_published", 0)
        checks["events_processed"] = eb_stats.get("events_processed", 0)
    
    return checks


@app.get("/metrics")
async def metrics(_auth=Depends(auth_dependency)):
    """Prometheus-compatible metrics endpoint."""

    lines = []
    lines.append("# HELP shadow_observations_total Total observations collected")
    lines.append("# TYPE shadow_observations_total counter")
    lines.append(f"shadow_observations_total {shadow.observation_count}")
    
    lines.append("# HELP shadow_connected_devices Total connected devices")
    lines.append("# TYPE shadow_connected_devices gauge")
    lines.append(f"shadow_connected_devices {len(shadow.connected_devices)}")
    
    lines.append("# HELP shadow_insights_total Total insights generated")
    lines.append("# TYPE shadow_insights_total counter")
    insights_count = len(shadow.shadow_ai.insights) if shadow.shadow_ai else 0
    lines.append(f"shadow_insights_total {insights_count}")
    
    lines.append("# HELP shadow_background_tasks Active background tasks")
    lines.append("# TYPE shadow_background_tasks gauge")
    lines.append(f"shadow_background_tasks {len(_background_tasks)}")
    
    lines.append("# HELP shadow_uptime_seconds Daemon uptime in seconds")
    lines.append("# TYPE shadow_uptime_seconds gauge")
    uptime = _time.time() - _start_time if _start_time > 0 else 0
    lines.append(f"shadow_uptime_seconds {round(uptime, 3)}")
    
    if shadow.orchestrator:
        stats = shadow.orchestrator.get_status()
        lines.append("# HELP shadow_tasks_processed Total tasks processed")
        lines.append("# TYPE shadow_tasks_processed counter")
        lines.append(f"shadow_tasks_processed {stats.get('tasks_processed', 0)}")
        
        lines.append("# HELP shadow_functions_generated Functions generated")
        lines.append("# TYPE shadow_functions_generated counter")
        lines.append(f"shadow_functions_generated {stats.get('functions_generated', 0)}")
    
    # Event bus metrics
    from .event_bus import get_event_bus
    event_bus = get_event_bus()
    eb_stats = event_bus.get_stats()
    lines.append("# HELP shadow_events_published Total events published")
    lines.append("# TYPE shadow_events_published counter")
    lines.append(f"shadow_events_published {eb_stats.get('events_published', 0)}")
    
    lines.append("# HELP shadow_events_processed Total events processed")
    lines.append("# TYPE shadow_events_processed counter")
    lines.append(f"shadow_events_processed {eb_stats.get('events_processed', 0)}")
    
    lines.append("# HELP shadow_events_errors Total event errors")
    lines.append("# TYPE shadow_events_errors counter")
    lines.append(f"shadow_events_errors {eb_stats.get('errors', 0)}")

    # Per-ability invocation metrics.
    from . import ability_metrics
    lines.extend(ability_metrics.prometheus_lines())

    return "\n".join(lines)


@app.get("/abilities/health")
async def abilities_health(_auth=Depends(auth_dependency)):
    """Per-ability health snapshot.

    Returns: registered abilities, count of invocations seen since startup,
    success rate, p50/p95 latency, and last invocation timestamp.
    Combines the static list_shikigami() with live ability_metrics.
    """
    try:
        from . import ability_metrics
        if not shadow.shikigami:
            raise HTTPException(status_code=503, detail="Shikigami not initialized")
        registered = shadow.shikigami.list_shikigami()
        live = ability_metrics.snapshot()
        out = []
        for entry in registered:
            name = entry["type"]
            stats = live.get(name)
            row = {
                "name": entry.get("name"),
                "type": name,
                "emoji": entry.get("emoji"),
                "power": entry.get("power"),
                "count": stats["count"] if stats else 0,
                "successes": stats["successes"] if stats else 0,
                "failures": stats["failures"] if stats else 0,
                "success_rate": (
                    round(stats["successes"] / stats["count"], 4)
                    if stats and stats["count"] > 0 else None
                ),
                "p50_latency_s": stats["p50_latency_s"] if stats else None,
                "p95_latency_s": stats["p95_latency_s"] if stats else None,
                "avg_latency_s": stats["avg_latency_s"] if stats else None,
                "last_invoked_ts": stats["last_invoked_ts"] if stats else None,
            }
            out.append(row)
        return {"abilities": out, "registered": len(registered)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("abilities/health failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="ability health lookup failed")


@app.get("/audit/recent")
async def audit_recent(limit: int = 100, _auth=Depends(auth_dependency)):
    """Return recent ability invocations from the audit log.

    Only returns content when SHADOW_ABILITY_AUDIT=1 is set. Each entry has
    a *hashed* query (sha256 prefix) — never raw user text — plus latency,
    success, confidence, and observation count.
    """
    try:
        from .ability_audit import get_audit
        audit = get_audit()
        # Clamp limit to a sane range.
        limit = max(1, min(int(limit), 1000))
        entries = audit.read_recent(lines=limit)
        return {
            "enabled": audit.enabled,
            "count": len(entries),
            "entries": entries,
        }
    except Exception as e:
        logger.exception("audit/recent failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="audit lookup failed")


# ============ WEBSOCKET ROUTES ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    if SHADOW_AUTH_ENABLED and security:
        token = websocket.query_params.get("token") or websocket.headers.get("Authorization", "").removeprefix("Bearer ")
        if token:
            payload = security.verify_token(token)
            if not payload:
                await websocket.close(code=4001, reason="Invalid token")
                return
        else:
            await websocket.close(code=4001, reason="Authentication required")
            return
    if not shadow.ws_server:
        await websocket.close(code=1011, reason="WebSocket server not initialized")
        return
    await shadow.ws_server.handle_connection(websocket)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint_with_id(websocket: WebSocket, client_id: str):
    """WebSocket endpoint with custom client ID."""
    if SHADOW_AUTH_ENABLED and security:
        token = websocket.query_params.get("token") or websocket.headers.get("Authorization", "").removeprefix("Bearer ")
        if token:
            payload = security.verify_token(token)
            if not payload:
                await websocket.close(code=4001, reason="Invalid token")
                return
        else:
            await websocket.close(code=4001, reason="Authentication required")
            return
    if len(client_id) > 100 or not client_id.replace("-", "").replace("_", "").isalnum():
        await websocket.close(code=1008, reason="Invalid client ID")
        return
    if not shadow.ws_server:
        await websocket.close(code=1011, reason="WebSocket server not initialized")
        return
    await shadow.ws_server.handle_connection(websocket, client_id)


# ============ MODELS ============

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    context: Optional[dict] = None
    shikigami: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Explicit ability name (e.g. 'oracle', 'x_ray'). Auto-selected if omitted.",
    )

class TrackPersonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    devices: str = Field(..., min_length=1, max_length=500)
    relation: Optional[str] = Field(default="", max_length=200)

    @field_validator("name", "devices", "relation")
    @classmethod
    def sanitize_strings(cls, v):
        if isinstance(v, str):
            return _sanitize_content(v)
        return v

class TrackDecisionRequest(BaseModel):
    situation: str = Field(..., min_length=1, max_length=2000)
    choice: str = Field(..., min_length=1, max_length=2000)
    outcome: str = Field(..., min_length=1, max_length=2000)
    satisfaction: int = Field(default=5, ge=1, le=10)

    @field_validator("situation", "choice", "outcome")
    @classmethod
    def sanitize_strings(cls, v):
        return _sanitize_content(v)

class LearnPatternRequest(BaseModel):
    cause: str = Field(..., min_length=1, max_length=2000)
    effect: str = Field(..., min_length=1, max_length=2000)

    @field_validator("cause", "effect")
    @classmethod
    def sanitize_strings(cls, v):
        return _sanitize_content(v)

class ActRequest(BaseModel):
    action: str = Field(..., min_length=1, max_length=10000)

    @field_validator("action")
    @classmethod
    def sanitize_strings(cls, v):
        return _sanitize_content(v)


# ============ ENDPOINTS ============

_WEB_DIR = (Path(__file__).parent.parent / "web").resolve()


def _safe_static(name: str, media_type: Optional[str] = None):
    """Serve a file from web/ if path-safe, else 404."""
    from fastapi.responses import FileResponse
    target = (_WEB_DIR / name).resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if media_type:
        return FileResponse(str(target), media_type=media_type)
    return FileResponse(str(target))


@app.get("/")
async def root():
    # If a PWA file is present, serve it. Otherwise return JSON status —
    # tests and curl users hit `/` expecting JSON, so keep that path open via
    # the Accept header (TestClient/curl default to */*; browsers send
    # text/html). When the file is missing we always return JSON.
    from fastapi import Request
    target = (_WEB_DIR / "shadow_pwa.html").resolve()
    if target.exists() and str(target).startswith(str(_WEB_DIR)):
        # Use the explicit /dashboard route for HTML; keep / as JSON so
        # existing tests that do `client.get("/")` still pass.
        pass

    if shadow.shadow_ai:
        status = shadow.shadow_ai.get_status()
    else:
        status = {"status": "initializing"}

    return {
        "name": "Shadow AI",
        "version": "1.0.0",
        "mode": "unified proactive optimization",
        "status": _sanitize_content(str(status.get("status", "")), 100) if isinstance(status, dict) else "unknown",
        "dashboard": "/dashboard" if target.exists() else None,
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the minimal, v1-focused dashboard (tracks the working endpoints)."""
    return _safe_static("shadow.html")


@app.get("/dashboard/full")
async def dashboard_full():
    """The original full PWA (targets the cognition v2 /os/* routes)."""
    return _safe_static("shadow_pwa.html")


@app.get("/shadow_os.html")
async def shadow_os_html():
    return _safe_static("shadow_os.html")


@app.get("/watch")
async def watch_app():
    return _safe_static("watch_app.html")


@app.get("/manifest.json")
async def shadow_manifest():
    return _safe_static("manifest.json", media_type="application/json")


@app.get("/sw.js")
async def shadow_service_worker():
    return _safe_static("sw.js", media_type="application/javascript")


@app.get("/status")
async def status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Detailed status."""
    coordinator_status = {}
    if hasattr(shadow, 'coordinator') and shadow.coordinator:
        coordinator_status = shadow.coordinator.get_ability_status()

    shadow_status = copy.deepcopy(shadow.shadow_ai.get_status()) if shadow.shadow_ai else {"status": "not initialized"}

    return {
        "daemon": "running",
        "observations": shadow.observation_count,
        "shadow_ai": shadow_status,
        "coordinator": coordinator_status
    }


@app.get("/coordinator/status")
async def coordinator_status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get coordinator status and ability profiles."""
    if not hasattr(shadow, 'coordinator'):
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    return {
        "abilities": shadow.coordinator.get_ability_status(),
        "context_state": copy.deepcopy(dict(shadow.coordinator.context.state)),
        "domain": shadow.coordinator.context.domain
    }


@app.get("/observers/status")
async def observers_status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Per-observer task health (alive / errored / cancelled)."""
    from .observers import observer_status
    return {"observers": observer_status()}


@app.get("/insights/recent")
async def insights_recent(limit: int = 20, _auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Recent proactive insights from the background scanner."""
    from .proactive_engine import recent_insights
    return {"insights": recent_insights(max(1, min(100, limit)))}


@app.post("/compact")
async def compact_observations(_auth=Depends(auth_dependency)):
    """Manually trigger observation compaction via LLM summarization.
    Compresses observations older than the most recent 50 into a dense summary."""
    if not shadow.shadow_ai or not shadow.mind:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    with shadow.shadow_ai._lock:
        obs = list(shadow.shadow_ai.observations)
        if len(obs) <= 50:
            return {"compacted": 0, "remaining": len(obs), "message": "Too few observations to compact."}
        old = obs[:-50]
        shadow.shadow_ai.observations = obs[-50:]
    summary = await shadow.mind.compact_context(
        [{"content": json.dumps(o, default=str)[:500]} for o in old[-200:]]
    )
    compact_obs = {"type": "compacted_summary", "content": summary,
                   "compacted_count": len(old),
                   "timestamp": datetime.now(timezone.utc).isoformat()}
    with shadow.shadow_ai._lock:
        shadow.shadow_ai.observations.insert(0, compact_obs)
    if shadow.shikigami and hasattr(shadow.shikigami, 'observations'):
        try:
            shadow.shikigami.observations.append(compact_obs)
            with shadow.shikigami._lock:
                if len(shadow.shikigami.observations) > 5000:
                    shadow.shikigami.observations = shadow.shikigami.observations[-5000:]
        except (AttributeError, RuntimeError, TypeError):
            pass
    shadow.shadow_ai._save_data()
    return {"compacted": len(old), "remaining": len(shadow.shadow_ai.observations),
            "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary}


@app.get("/compact/status")
async def compact_status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get compaction status."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    with shadow.shadow_ai._lock:
        return {
            "total_observations": len(shadow.shadow_ai.observations),
            "compacted_summaries": sum(1 for o in shadow.shadow_ai.observations if isinstance(o, dict) and o.get("type") == "compacted_summary"),
            "recent_raw": sum(1 for o in shadow.shadow_ai.observations if not (isinstance(o, dict) and o.get("type") == "compacted_summary")),
        }


@app.get("/config")
async def config_view(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Sanitized config snapshot — env vars, feature flags, paths.
    Never returns credentials."""
    safe_env = {
        "SHADOW_AUTH_ENABLED": SHADOW_AUTH_ENABLED,
        "SHADOW_PROACTIVE_INTERVAL": os.environ.get("SHADOW_PROACTIVE_INTERVAL", "60"),
        "SHADOW_MANIPULATION_THRESHOLD": os.environ.get("SHADOW_MANIPULATION_THRESHOLD", "0.35"),
        "SHADOW_PROACTIVE_DISABLED": os.environ.get("SHADOW_PROACTIVE_DISABLED", "false"),
        "SHADOW_FORGE_EXECUTE_WRITES": os.environ.get("SHADOW_FORGE_EXECUTE_WRITES", "false"),
        "SHADOW_FORGE_DRY_RUN": os.environ.get("SHADOW_FORGE_DRY_RUN", "false"),
        "LLM_MODEL": os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL"),
        "LLM_API_BASE": os.environ.get("LLM_API_BASE") or os.environ.get("OLLAMA_API_BASE"),
        "OLLAMA_EMBED_BASE": os.environ.get("OLLAMA_EMBED_BASE", "http://127.0.0.1:11434/api/embed"),
        "OLLAMA_EMBED_MODEL": os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    }
    feature_flags = {
        "auth_enabled": SHADOW_AUTH_ENABLED,
        "llm_configured": bool((os.environ.get("LLM_API_KEY") or "").strip()
                               or (os.environ.get("OLLAMA_API_KEY") or "").strip()),
        "proactive_engine": os.environ.get("SHADOW_PROACTIVE_DISABLED", "").lower() not in ("1", "true", "yes"),
        "mcp_installed": False,
    }
    try:
        import mcp  # noqa: F401
        feature_flags["mcp_installed"] = True
    except ImportError:
        pass
    return {
        "env": safe_env,
        "flags": feature_flags,
        "paths": {
            "data_dir": str(DATA_DIR),
            "workspace": str((Path.home() / "digital-shadow" / "workspace").resolve()),
            "token_file_hint": "~/.config/shadow/token (CLI managed)",
        },
    }


@app.get("/snapshot")
async def unified_snapshot(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get unified environmental snapshot."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    return await shadow.shadow_ai.get_unified_snapshot()


@app.post("/ask")
async def ask_unified(request: AskRequest, _auth=Depends(auth_dependency)):
    """
    Ask Shadow AI anything.
    Uses multi-agent coordinator to route to relevant abilities.
    """
    if not shadow.shadow_ai or not hasattr(shadow, 'coordinator'):
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    # Use coordinator for multi-agent coordination
    domain = request.context.get("domain") if request.context else None
    result = await shadow.coordinator.coordinate(request.question, domain)
    return result


@app.get("/ask")
async def ask_simple(q: str = Query(..., max_length=10000), _auth=Depends(auth_dependency)):
    """Simple query interface."""
    if not q or len(q) > 10000:
        raise HTTPException(status_code=400, detail="Query must be 1-10000 characters")
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")

    result = await shadow.shadow_ai.ask(q, {"observations": shadow.get_observations(20)})
    return {"response": result["unified_response"]}


@app.post("/ask/stream")
async def ask_stream(request: AskRequest, _auth=Depends(auth_dependency)):
    """Stream LLM response via Server-Sent Events."""
    if not shadow.shadow_ai or not shadow.mind:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    from shadowd.streaming import generate_sse_stream, stream_llm_response
    from shadowd.mind import OLLAMA_API_BASE, OLLAMA_API_KEY, DEFAULT_MODEL
    
    messages = shadow.mind._build_messages(
        request.question,
        request.context,
        shadow.get_observations(20)
    )
    
    async def token_generator():
        client = await shadow.mind._get_client()
        async for token in stream_llm_response(
            client=client,
            messages=messages,
            model=DEFAULT_MODEL,
            api_base=OLLAMA_API_BASE,
            api_key=OLLAMA_API_KEY
        ):
            yield token
    
    return StreamingResponse(
        generate_sse_stream(token_generator(), model=DEFAULT_MODEL),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/act")
async def take_action(request: ActRequest, _auth=Depends(auth_dependency)):
    """Execute an action."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    result = await shadow.shadow_ai.act(request.action)
    return result


# ============ MEMORY TOOLS (Letta-style) ============

class MemoryReplaceRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    old_string: str = Field(..., min_length=1, max_length=5000)
    new_string: str = Field(..., max_length=5000)

    @field_validator("label")
    @classmethod
    def sanitize_label(cls, v):
        sanitized = _sanitize(v)
        if not sanitized:
            raise ValueError("label must not be empty after sanitization")
        return sanitized

class MemoryInsertRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    new_string: str = Field(..., min_length=1, max_length=5000)
    insert_line: int = -1

    @field_validator("label")
    @classmethod
    def sanitize_label(cls, v):
        sanitized = _sanitize(v)
        if not sanitized:
            raise ValueError("label must not be empty after sanitization")
        return sanitized

class MemoryRethinkRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    new_memory: str = Field(..., min_length=1, max_length=20000)

    @field_validator("label")
    @classmethod
    def sanitize_label(cls, v):
        sanitized = _sanitize(v)
        if not sanitized:
            raise ValueError("label must not be empty after sanitization")
        return sanitized


@app.get("/memory")
async def get_memory(limit: int = 20, _auth=Depends(auth_dependency)):
    """Get memory blocks rendered as XML or recent observations."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    # If limit requested, return observations
    if limit != 20:
        return {
            "observations": shadow.get_observations(limit),
            "count": shadow.observation_count
        }
    
    # Otherwise return memory blocks
    from shadowd.memory_tools import render_memory_xml
    from shadowd.block_memory import BlockBasedMemory
    
    memory_dir = DATA_DIR / "memory_blocks"
    memory = BlockBasedMemory(data_dir=memory_dir)
    
    # Populate with current data if blocks are empty
    if not memory.core_get("human") and shadow.shadow_ai.people:
        memory.core_set("human", json.dumps(shadow.shadow_ai.people, indent=2)[:2000])
    if not memory.core_get("recent") and shadow.observations:
        recent_obs = shadow.get_observations(20)
        memory.core_set("recent", json.dumps(recent_obs, indent=2, default=str)[:1000])
    if not memory.core_get("goals") and shadow.shadow_ai.patterns:
        memory.core_set("goals", json.dumps(shadow.shadow_ai.patterns, indent=2)[:1000])
    
    blocks = memory.core_memory.blocks
    
    return {
        "xml": render_memory_xml(blocks),
        "blocks": {k: {"value": v.value, "chars": len(v.value), "limit": v.limit} for k, v in blocks.items()}
    }


@app.post("/memory/replace")
async def memory_replace(request: MemoryReplaceRequest, _auth=Depends(auth_dependency)):
    """Replace an exact string in a memory block."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    from shadowd.memory_tools import MemoryTools
    from shadowd.block_memory import BlockBasedMemory
    
    memory_dir = DATA_DIR / "memory_blocks"
    memory = BlockBasedMemory(data_dir=memory_dir)
    tools = MemoryTools(memory.core_memory)
    
    # Sync data into memory blocks
    if shadow.shadow_ai.people:
        memory.core_set("human", json.dumps(shadow.shadow_ai.people, indent=2)[:2000])
    
    result = tools.memory_replace(request.label, request.old_string, request.new_string)
    
    # Sync changes back
    if result.success and request.label == "human":
        try:
            human_val = memory.core_get("human")
            if human_val:
                shadow.shadow_ai.people = json.loads(human_val)
                shadow.shadow_ai._save_data()
        except (json.JSONDecodeError, TypeError):
            pass
    
    return result.model_dump()


@app.post("/memory/insert")
async def memory_insert(request: MemoryInsertRequest, _auth=Depends(auth_dependency)):
    """Insert content into a memory block at a specific line."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    from shadowd.memory_tools import MemoryTools
    from shadowd.block_memory import BlockBasedMemory
    
    memory_dir = DATA_DIR / "memory_blocks"
    memory = BlockBasedMemory(data_dir=memory_dir)
    tools = MemoryTools(memory.core_memory)
    
    result = tools.memory_insert(request.label, request.new_string, request.insert_line)
    return result.model_dump()


@app.post("/memory/rethink")
async def memory_rethink(request: MemoryRethinkRequest, _auth=Depends(auth_dependency)):
    """Completely rewrite a memory block."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    from shadowd.memory_tools import MemoryTools
    from shadowd.block_memory import BlockBasedMemory
    
    memory_dir = DATA_DIR / "memory_blocks"
    memory = BlockBasedMemory(data_dir=memory_dir)
    tools = MemoryTools(memory.core_memory)
    
    result = tools.memory_rethink(request.label, request.new_memory)
    return result.model_dump()


@app.get("/memory/budget")
async def memory_budget(_auth=Depends(auth_dependency)):
    """Get context window budget overview."""
    if not shadow.mind:
        raise HTTPException(status_code=503, detail="Mind engine not initialized")
    
    budget = shadow.mind.context_budget
    system_prompt = shadow.mind.context_prompt
    messages = shadow.mind._build_messages("budget check", {}, shadow.get_observations(20))
    
    return budget.get_budget_overview(
        system_prompt=system_prompt,
        messages=messages,
        observations=shadow.get_observations(50)
    )


@app.post("/track/person")
async def track_person(request: TrackPersonRequest, _auth=Depends(auth_dependency)):
    """Learn to identify a person."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    result = shadow.shadow_ai.track_person(request.name, request.devices, request.relation)
    return {"success": True, "message": result}


@app.post("/track/decision")
async def track_decision(request: TrackDecisionRequest, _auth=Depends(auth_dependency)):
    """Learn from a decision."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    result = shadow.shadow_ai.track_decision(
        request.situation, request.choice, request.outcome, request.satisfaction
    )
    return {"success": True, "message": result}


@app.post("/track/pattern")
async def learn_pattern(request: LearnPatternRequest, _auth=Depends(auth_dependency)):
    """Learn a causation pattern."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    
    result = shadow.shadow_ai.learn_pattern(request.cause, request.effect)
    return {"success": True, "message": result}


@app.get("/insights")
async def get_insights(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get proactive insights."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")

    status = shadow.shadow_ai.get_status() if hasattr(shadow.shadow_ai, 'get_status') else {}
    insights_list = getattr(shadow.shadow_ai, 'insights', [])
    recent = insights_list[-20:] if insights_list else []
    critical = [i for i in recent if getattr(i, 'priority', '') in ("critical", "high")]
    return {
        "critical": [{"ability": getattr(i, 'ability', ''), "message": _sanitize_content(str(getattr(i, 'message', '')), 500), "action": _sanitize_content(str(getattr(i, 'action', '')), 500)} for i in critical],
        "total_insights": len(insights_list)
    }


@app.get("/observations")
async def get_observations(limit: int = 20, _auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get recent observations."""
    return {
        "observations": shadow.get_observations(limit),
        "count": shadow.observation_count
    }


@app.get("/context")
async def get_context(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    current_activity = shadow.get_current_activity()
    return {
        "current_activity": current_activity,
        "recent_observations": shadow.get_observations(10),
        "devices": list(shadow.get_connected_devices().keys())
    }


# ============ WEBSOCKET STATUS ENDPOINTS ============

@app.get("/ws/clients")
async def get_ws_clients(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get connected WebSocket clients."""
    if not shadow.ws_server:
        raise HTTPException(status_code=503, detail="WebSocket server not initialized")

    return {
        "connected_clients": shadow.ws_server.manager.get_client_count(),
        "clients": copy.deepcopy(shadow.ws_server.manager.get_connected_clients())
    }


@app.get("/ws/topics/{topic}/subscribers")
async def get_topic_subscribers(topic: str, _auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get subscribers for a topic."""
    if not topic or len(topic) > 100 or not topic.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid topic name")
    if not shadow.ws_server:
        raise HTTPException(status_code=503, detail="WebSocket server not initialized")
    
    return {
        "topic": topic,
        "subscribers": shadow.ws_server.manager.get_topic_subscribers(topic)
    }


@app.post("/ws/broadcast")
async def broadcast_message(message: dict, _auth=Depends(auth_dependency)):
    """Broadcast message to all connected clients. Requires auth."""
    if not shadow.ws_server:
        raise HTTPException(status_code=503, detail="WebSocket server not initialized")

    try:
        msg_json = json.dumps(message, default=str)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Message contains non-serializable data")
    if len(msg_json) > 65536:
        raise HTTPException(status_code=413, detail="Message too large (max 64KB)")

    msg = WSMessage(
        type=WSMessageType.EVENT,
        data=message
    )
    await shadow.ws_server.manager.broadcast(msg)

    return {
        "status": "broadcasted",
        "client_count": shadow.ws_server.manager.get_client_count()
    }


@app.post("/ws/topics/{topic}/broadcast")
async def broadcast_to_topic(topic: str, message: dict, _auth=Depends(auth_dependency)):
    """Broadcast message to topic subscribers. Requires auth."""
    if not topic or len(topic) > 100 or not topic.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid topic name")
    if not shadow.ws_server:
        raise HTTPException(status_code=503, detail="WebSocket server not initialized")

    try:
        msg_json = json.dumps(message, default=str)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Message contains non-serializable data")
    if len(msg_json) > 65536:
        raise HTTPException(status_code=413, detail="Message too large (max 64KB)")

    msg = WSMessage(
        type=WSMessageType.EVENT,
        data=message
    )
    await shadow.ws_server.manager.broadcast_to_topic(topic, msg)

    return {
        "status": "broadcasted",
        "topic": _sanitize(topic, 100),
        "subscriber_count": len(shadow.ws_server.manager.get_topic_subscribers(topic))
    }


# ============ MCP SERVER ENDPOINTS ============

@app.get("/mcp/status")
async def get_mcp_status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """Get MCP server status."""
    if not shadow.mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    return {
        "available": True,
        "name": shadow.mcp_server.name,
        "version": shadow.mcp_server.version,
        "metrics": shadow.mcp_server.metrics.to_dict()
    }


@app.get("/mcp/tools")
async def get_mcp_tools(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """List available MCP tools."""
    if not shadow.mcp_server or not shadow.mcp_server.mcp:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")

    tools = shadow.mcp_server.metrics.tools_called
    return {
        "tools": list(tools.keys()) if tools else [],
        "call_counts": dict(tools) if tools else {}
    }


@app.get("/mcp/resources")
async def get_mcp_resources(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """List accessed MCP resources."""
    if not shadow.mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")

    resources = shadow.mcp_server.metrics.resources_accessed
    return {
        "resources": list(resources.keys()) if resources else [],
        "access_counts": dict(resources) if resources else {}
    }


# ============ AUTH ENDPOINTS ============

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=200)
    password: str = Field(..., max_length=200)

class TokenRequest(BaseModel):
    refresh_token: str = Field(..., max_length=500)


def _check_login_rateLimit(client_key: str, max_attempts: int = 5, window_seconds: int = 300) -> bool:
    max_clients = 10000
    now = _time.time()
    with _login_rate_lock:
        if client_key not in _login_rate_limits:
            _login_rate_limits[client_key] = []
        attempts = _login_rate_limits[client_key]
        attempts[:] = [t for t in attempts if now - t < window_seconds]
        if len(attempts) >= max_attempts:
            return False
        attempts.append(now)
        if len(_login_rate_limits) > max_clients:
            keys_to_evict = list(_login_rate_limits.keys())[:max_clients // 2]
            for k in keys_to_evict:
                del _login_rate_limits[k]
        return True

@app.post("/auth/login")
async def login(request: LoginRequest, req: Request):
    """Authenticate and get access token."""
    client_key = req.client.host if req.client else "unknown"
    if not _check_login_rateLimit(client_key):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    if not security:
        raise HTTPException(status_code=501, detail="Authentication not configured")
    result = security.login(request.email, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return result

@app.post("/auth/refresh")
async def refresh_token(request: TokenRequest, req: Request):
    """Refresh access token."""
    client_key = f"refresh:{req.client.host}" if req.client else "refresh:unknown"
    if not _check_login_rateLimit(client_key, max_attempts=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many refresh attempts")
    if not security:
        raise HTTPException(status_code=501, detail="Authentication not configured")
    new_token = security.refresh_access_token(request.refresh_token)
    if not new_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return {"access_token": new_token, "token_type": "bearer"}

@app.get("/auth/status")
async def auth_status():
    """Check authentication status."""
    return {"auth_enabled": SHADOW_AUTH_ENABLED, "auth_configured": security is not None}


# ============ LEGACY SHIKIGAMI ENDPOINTS ============

@app.get("/shikigami")
async def list_shikigami(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    """List all 10 abilities."""
    if not shadow.shadow_ai:
        raise HTTPException(status_code=503, detail="Shadow AI not initialized")
    return {"shikigami": shadow.shadow_ai.registry.list_shikigami()}


# Default council when the question carries no clear ability signal: a broad
# strategic blend (future + decisions + patterns + people + routine).
_DEFAULT_COUNCIL = ("oracle", "shadow_council", "pattern_sight", "social_radar", "habit_tracker")
# Analytical core used to guarantee a minimum breadth of voices when a question
# only maps to one or two abilities — a council of one isn't a council.
_COUNCIL_CORE = ("oracle", "pattern_sight", "shadow_council")
_COUNCIL_MIN = 3   # floor on the number of voices in a relevance-picked council
_COUNCIL_MAX = 8   # ceiling, to keep fan-out latency bounded


def _select_council_abilities(question: str):
    """Pick the council's member abilities from a natural-language question.

    Returns ``(abilities, mode)`` where ``abilities`` is a deduped list of
    ShikigamiType and ``mode`` is:
      - ``"relevance"`` — the question mapped to concrete abilities (the common
        case). The omni_sense catch-all is dropped, and the analytical core is
        topped up to guarantee at least ``_COUNCIL_MIN`` voices, capped at
        ``_COUNCIL_MAX``.
      - ``"default"`` — the question carried no ability signal; falls back to
        the broad ``_DEFAULT_COUNCIL`` blend.

    This makes the documented "picks the most relevant subset" contract real:
    e.g. "how have I felt lately?" now consults Tide/Mirror instead of the old
    fixed five.
    """
    from shikigami import ShikigamiType
    from .shadow_ai import detect_relevant_abilities

    def _to_types(names):
        out, seen = [], set()
        for n in names:
            try:
                st = ShikigamiType(n)
            except ValueError:
                continue
            if st not in seen:
                seen.add(st)
                out.append(st)
        return out

    try:
        detected = detect_relevant_abilities(question)
    except Exception:  # pragma: no cover - selection must never fail the request
        logger.exception("council ability selection failed; using default council")
        return _to_types(_DEFAULT_COUNCIL), "default"

    # Drop the omni_sense catch-all: it only surfaces when nothing else matched,
    # and adds little to a council that already has concrete voices.
    meaningful, seen = [], set()
    for st in detected:
        if st == ShikigamiType.OMNI_SENSE or st in seen:
            continue
        seen.add(st)
        meaningful.append(st)

    if not meaningful:
        return _to_types(_DEFAULT_COUNCIL), "default"

    chosen = list(meaningful)
    for st in _to_types(_COUNCIL_CORE):
        if len(chosen) >= _COUNCIL_MIN:
            break
        if st not in seen:
            seen.add(st)
            chosen.append(st)
    return chosen[:_COUNCIL_MAX], "relevance"


class CouncilRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    abilities: Optional[List[str]] = Field(
        default=None,
        description="Specific abilities to consult. If omitted, the daemon "
                    "picks the most relevant subset from the question.",
    )
    context: Optional[dict] = None
    dry_run: bool = Field(
        default=False,
        description="If true, return the planned ability list without "
                    "actually invoking them. Useful for previewing.",
    )


@app.post("/shikigami/council")
async def summon_council(request: CouncilRequest, _auth=Depends(auth_dependency)):
    """Fan a question out to multiple abilities in parallel, then synthesize
    a single verdict from their outputs. The 'Council of Abilities'."""
    from shikigami import ShikigamiType

    if not shadow.shikigami:
        raise HTTPException(status_code=503, detail="Shikigami not initialized")

    # Choose which abilities to consult.
    if request.abilities:
        try:
            chosen = [ShikigamiType(a.lower().strip()) for a in request.abilities[:10]]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unknown ability: {e}")
        selection = "explicit"
    else:
        # Relevance-driven: map the question to the abilities that actually
        # bear on it (falls back to a broad default blend on no signal).
        chosen, selection = _select_council_abilities(request.question)

    # Dry-run: return the plan without invoking any ability.
    if request.dry_run:
        return {
            "dry_run": True,
            "question": request.question,
            "selection": selection,
            "planned_abilities": [s.value for s in chosen],
            "ability_count": len(chosen),
            "observations_in_context": len(
                (request.context or {}).get("observations",
                                            shadow.get_observations(50)) or []
            ),
            "verdict": (
                "🏛️ **Dry run** — would consult "
                f"{', '.join(s.value for s in chosen)} but did not invoke them."
            ),
        }

    context = request.context or {"observations": shadow.get_observations(50)}

    # Pre-parse + index observations once so each council member doesn't re-
    # iterate / re-parse them. The wrapper is list-compatible so abilities
    # that aren't ParsedObservations-aware still see normal dicts.
    try:
        from shikigami.parsed_observations import ParsedObservations
        if isinstance(context.get("observations"), list):
            parsed = ParsedObservations(context["observations"])
            parsed.precompute_timestamps()
            context = dict(context)
            context["observations"] = parsed
    except Exception:  # pragma: no cover - never break the council on cache failure
        pass

    async def _run(stype):
        try:
            r = await shadow.shikigami.execute(stype, request.question, context)
            return stype.value, {
                "success": r.success, "output": r.output, "data": r.data,
                "confidence": getattr(r, "confidence", 0.0),
            }
        except Exception as e:
            logger.exception("council member %s raised: %s", stype.value, type(e).__name__)
            return stype.value, {"success": False, "output": f"⚠️ {type(e).__name__}",
                                  "data": {"error_type": type(e).__name__}, "confidence": 0.0}

    results = dict(await asyncio.gather(*[_run(s) for s in chosen]))

    # Synthesis: surface highest-confidence verdict, then a brief mention from
    # each consulted member.
    successes = {k: v for k, v in results.items() if v["success"]}
    leader = max(successes.items(), key=lambda kv: kv[1].get("confidence", 0.0)) \
             if successes else None

    # Cross-ability narrative: pull signals + recommendations from the
    # consulted members' structured data.
    try:
        from shikigami.narrative import weave
        woven = weave({name: r["data"] for name, r in successes.items()})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("narrative weave failed: %s", type(e).__name__)
        woven = None

    synthesis = "🏛️ **Council of Abilities**\n\n"
    _sel_label = {"explicit": "you chose", "relevance": "relevance-picked",
                  "default": "default blend"}.get(selection, selection)
    synthesis += f"**Consulted**: {len(results)} abilities ({', '.join(results.keys())}) — {_sel_label}\n"
    synthesis += f"**Succeeded**: {len(successes)}\n"
    if woven is not None:
        synthesis += f"**Health gauge**: {woven.health_score:.2f}/1.0\n"
        if woven.themes:
            synthesis += f"**Threads**: {', '.join(woven.themes[:5])}\n"
    synthesis += "\n"
    if leader:
        synthesis += f"**Lead voice**: {leader[0]} (conf={leader[1]['confidence']:.2f})\n"
        leader_out = leader[1]["output"]
        synthesis += f"  {leader_out[:300]}{'…' if len(leader_out) > 300 else ''}\n\n"

    if woven is not None and woven.signals:
        synthesis += "**SIGNALS**:\n"
        for s in woven.signals[:12]:
            synthesis += f"  • {s}\n"
        synthesis += "\n"
    if woven is not None and woven.contradictions:
        synthesis += "**CONTRADICTIONS** (cross-signal):\n"
        for c in woven.contradictions:
            synthesis += f"  ⚠ {c.description}\n      ({', '.join(c.abilities)})\n"
        synthesis += "\n"
    if woven is not None and woven.recommendations:
        synthesis += "**RECOMMENDATIONS** (cross-signal):\n"
        for rec in woven.recommendations[:5]:
            synthesis += f"  [{rec.priority}] {rec.title}\n      {rec.rationale[:200]}\n"
        synthesis += "\n"

    synthesis += "**MEMBER VOICES**:\n"
    for name, r in results.items():
        if name == (leader[0] if leader else None):
            continue
        if r["success"]:
            preview = r["output"].split("\n")[0:3]
            synthesis += f"  • [{name}] " + " ".join(p.strip() for p in preview)[:200] + "\n"
        else:
            synthesis += f"  ⛔ [{name}] {r['output'][:120]}\n"

    response = {
        "verdict": synthesis,
        "selection": selection,
        "leader": leader[0] if leader else None,
        "members": results,
    }
    if woven is not None:
        response["narrative"] = {
            "summary": woven.summary,
            "signals": woven.signals,
            "themes": woven.themes,
            "health_score": woven.health_score,
            "contradictions": [
                {"description": c.description, "abilities": c.abilities}
                for c in woven.contradictions
            ],
            "recommendations": [
                {
                    "title": r.title, "rationale": r.rationale,
                    "priority": r.priority,
                    "triggering_abilities": r.triggering_abilities,
                }
                for r in woven.recommendations
            ],
        }
    return response


@app.post("/shikigami/summon")
async def summon_ability(request: AskRequest, _auth=Depends(auth_dependency)):
    """Invoke a specific shikigami ability, or let it auto-select."""
    from shikigami import ShikigamiType

    if not shadow.shikigami:
        raise HTTPException(status_code=503, detail="Shikigami not initialized")

    context = request.context or {"observations": shadow.get_observations(20)}

    if request.shikigami:
        try:
            shikigami_type = ShikigamiType(request.shikigami.lower().strip())
        except ValueError:
            valid = [s.value for s in ShikigamiType]
            raise HTTPException(
                status_code=400,
                detail=f"Unknown shikigami '{request.shikigami}'. Valid: {valid}",
            )
    else:
        shikigami_type = shadow.shikigami.auto_select(request.question)

    # Defensive: a buggy ability shouldn't crash the daemon or leak internals
    # to the client. Always return a structured response. Also record per-
    # ability metrics so /metrics can report invocation counts + p95 latency,
    # and write to the audit log if enabled.
    from . import ability_metrics
    from .ability_audit import get_audit
    start = _time.time()
    obs_count = len((context or {}).get("observations", []) or [])
    audit = get_audit()
    try:
        result = await shadow.shikigami.execute(shikigami_type, request.question, context)
        elapsed = _time.time() - start
        ability_metrics.record(shikigami_type.value, result.success, elapsed)
        audit.record(
            shikigami_type.value, request.question, result.success, elapsed,
            confidence=getattr(result, "confidence", 0.0),
            observation_count=obs_count,
        )
        return {
            "ability": shikigami_type.value,
            "success": result.success,
            "output": result.output,
            "data": result.data,
            "auto_selected": request.shikigami is None,
        }
    except Exception as e:
        elapsed = _time.time() - start
        ability_metrics.record(shikigami_type.value, False, elapsed)
        audit.record(
            shikigami_type.value, request.question, False, elapsed,
            confidence=0.0, observation_count=obs_count,
            error_type=type(e).__name__,
        )
        logger.exception("ability %s raised: %s", shikigami_type.value, type(e).__name__)
        return {
            "ability": shikigami_type.value,
            "success": False,
            "output": f"⚠️ {shikigami_type.value} hit an internal error ({type(e).__name__}). "
                      f"Daemon log has details.",
            "data": {"error_type": type(e).__name__},
            "auto_selected": request.shikigami is None,
        }


class ObserveRequest(BaseModel):
    activity_type: str = Field(..., min_length=1, max_length=50)
    content: dict = Field(default_factory=dict)
    device: Optional[str] = Field(default=None, max_length=100)

    @field_validator("activity_type")
    @classmethod
    def validate_activity_type(cls, v):
        v = _SANITIZE_STRICT_RE.sub('', v)
        if not v:
            raise ValueError("activity_type must not be empty after sanitization")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v):
        if v and len(json.dumps(v, default=str)) > 65536:
            raise ValueError("Content too large (max 64KB)")
        if v and len(v) > 50:
            raise ValueError("Content too many keys (max 50)")
        return v

@app.post("/observe")
async def observe(request: ObserveRequest, _auth=Depends(auth_dependency)):
    """Add observation."""
    from .security_utils import get_rate_limiter
    limiter = get_rate_limiter("observe", max_requests=30, window_seconds=60)
    client_key = f"observe:{request.device or 'unknown'}"
    if not limiter.check(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    observation = {
        "type": request.activity_type,
        "content": request.content,
        "device": request.device or "localhost"
    }
    await shadow.learn(observation)
    return {"status": "observed", "count": shadow.observation_count}


@app.post("/device/connect")
async def device_connect(device_id: str, name: str, _auth=Depends(auth_dependency)):
    if not _re.match(r'^[a-zA-Z0-9_-]{1,64}$', device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id format")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Invalid device name")
    count = shadow.connect_device(device_id, name)
    return {"status": "connected", "devices": count}


# ============ ORCHESTRATOR ENDPOINTS ============

@app.get("/orchestrator/status")
async def orchestrator_status(_auth=Depends(auth_dependency) if SHADOW_AUTH_ENABLED else None):
    if not shadow.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return shadow.orchestrator.get_status()

@app.post("/orchestrator/process")
async def orchestrator_process(request: AskRequest, _auth=Depends(auth_dependency)):
    if not shadow.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    result = await shadow.orchestrator.process({
        "input": request.question,
        "context": request.context or {}
    })
    return result

@app.post("/orchestrator/learn")
async def orchestrator_learn(request: AskRequest, _auth=Depends(auth_dependency)):
    if not shadow.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    result = shadow.orchestrator.learn(request.question)
    return result


# ============ WEB UI ============

@app.get("/ui")
async def serve_ui():
    """Serve web UI."""
    return FileResponse(PROJECT_ROOT / "web" / "index.html")


def run():
    config = get_config()
    host = config.server.host
    port = config.server.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()