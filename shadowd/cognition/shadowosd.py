#!/usr/bin/env python3
"""
ShadowOS Daemon - Standalone server for the Shadow Operating System.

Runs ShadowOS as an independent FastAPI server on a dedicated port (8766),
separate from the legacy shadowd daemon. Provides the /os/* API surface
and the new cognitive dashboard.

Usage:
  python -m shadowd.cognition.shadowosd [--port PORT] [--host HOST]
"""

import argparse
import logging
import os
import re as _re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader

from .shadow_os import ShadowOS
from .api_router import router
from .daily_briefing import DailyBriefingEngine
from .escalation import EscalationManager

logger = logging.getLogger("shadowd.cognition.shadowosd")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        import json as _json
        log_entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exc"] = self.formatException(record.exc_info)
        return _json.dumps(log_entry, default=str)


def _setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

_setup_logging()

DATA_DIR = Path(os.environ.get("SHADOW_DATA_DIR", Path.home() / "digital-shadow" / "data"))
CORS_ORIGINS = os.environ.get(
    "SHADOW_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8765,http://127.0.0.1:8765,http://localhost:8766,http://127.0.0.1:8766"
).split(",")

CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
CORS_HEADERS = ["Content-Type", "Authorization", "X-API-Key"]

# Module-level singleton pattern for _shadow_os_instance. Thread-safe initialization
# is managed via the lifespan context (async) and the FastAPI event loop. If manual
# startup outside FastAPI is needed, wrap assignment in _ESCALATION_MANAGER_LOCK or
# equivalent to prevent races. See Sprint 26 / Sprint 28 audit notes for details.
_shadow_os_instance: ShadowOS = None
_briefing_engine: DailyBriefingEngine = None
_rate_limit_store: Dict[str, List[float]] = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_PER_MINUTE = 120
_RATE_LIMIT_MAX_ENTRIES = 10000

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_RE.sub('', text).strip()[:max_length]


_ESCALATION_MANAGER_LOCK = threading.Lock()
_BRIEFING_LOCK = threading.Lock()

_OS_API_KEY = os.environ.get("SHADOW_OS_API_KEY", "")
_os_auth_enabled = bool(_OS_API_KEY)
_os_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_os_api_key(api_key: str = Depends(_os_api_key_header)):
    if not _os_auth_enabled:
        return True
    if not api_key or api_key != _OS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shadow_os_instance
    start_time = time.monotonic()
    logger.info("=" * 50)
    logger.info("ShadowOS Daemon Starting...")
    logger.info("Data Directory: %s", DATA_DIR)
    logger.info("=" * 50)

    os_inst = ShadowOS(data_dir=str(DATA_DIR / "shadow_os"))
    _shadow_os_instance = os_inst
    await os_inst.start()

    import shadowd.cognition.api_router as api_module
    api_module._shadow_os = os_inst

    elapsed = time.monotonic() - start_time
    logger.info("ShadowOS ready in %.2fs - all modules initialized", elapsed)

    yield

    logger.info("ShadowOS shutting down...")
    await os_inst.stop()
    logger.info("ShadowOS stopped")

    _shadow_os_instance = None


app = FastAPI(
    title="ShadowOS",
    description="Shadow Operating System - Cognitive AI Agent Platform",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_limit_lock:
        requests = _rate_limit_store.get(client_id, [])
        requests = [t for t in requests if now - t < 60]
        if len(requests) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        requests.append(now)
        _rate_limit_store[client_id] = requests[-RATE_LIMIT_PER_MINUTE * 2:]
        if len(_rate_limit_store) > _RATE_LIMIT_MAX_ENTRIES:
            overflow = len(_rate_limit_store) - _RATE_LIMIT_MAX_ENTRIES // 2
            keys_to_remove = list(_rate_limit_store.keys())[:overflow]
            for k in keys_to_remove:
                del _rate_limit_store[k]
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

app.include_router(router)


@app.get("/health")
async def health_check():
    if _shadow_os_instance is None:
        return {"status": "initializing"}
    return _shadow_os_instance.get_health()


@app.get("/")
async def root():
    target = (_WEB_DIR / "shadow_pwa.html").resolve()
    if str(target).startswith(str(_WEB_DIR)) and target.exists():
        return FileResponse(str(target))
    if _shadow_os_instance is None:
        return {"status": "initializing"}
    return _shadow_os_instance.get_status()


_WEB_DIR = (Path(__file__).parent.parent.parent / "web").resolve()


@app.get("/dashboard")
async def dashboard():
    target = (_WEB_DIR / "shadow_os.html").resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))


@app.get("/shadow")
async def shadow_pwa():
    target = (_WEB_DIR / "shadow_pwa.html").resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))


@app.get("/watch")
async def watch_app():
    target = (_WEB_DIR / "watch_app.html").resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))


@app.get("/manifest.json")
async def shadow_manifest():
    target = (_WEB_DIR / "manifest.json").resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target), media_type="application/json")


@app.get("/sw.js")
async def shadow_service_worker():
    target = (_WEB_DIR / "sw.js").resolve()
    if not str(target).startswith(str(_WEB_DIR)) or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target), media_type="application/javascript")


@app.get("/briefing/generate", dependencies=[Depends(_verify_os_api_key)])
async def generate_briefing():
    global _briefing_engine
    if _shadow_os_instance is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    with _BRIEFING_LOCK:
        if _briefing_engine is None:
            _briefing_engine = DailyBriefingEngine(
                memory=_shadow_os_instance.memory,
                personality=_shadow_os_instance.personality,
                mesh=_shadow_os_instance.mesh,
                evolution_guide=_shadow_os_instance.evolution,
            )
    briefing = await _briefing_engine.generate_briefing(
        device_context=_shadow_os_instance.device_context,
        evolution_guide=_shadow_os_instance.evolution,
    )
    return briefing.to_dict()


@app.post("/briefing/generate", dependencies=[Depends(_verify_os_api_key)])
async def generate_briefing_post():
    return await generate_briefing()


@app.post("/escalation/evaluate", dependencies=[Depends(_verify_os_api_key)])
async def evaluate_escalation(action_type: str, parameters: dict = None, domain: str = None):
    action_type = _sanitize(action_type, 200)
    if domain:
        domain = _sanitize(domain, 200)
    if parameters:
        parameters = {_sanitize(str(k), 100): _sanitize_content(str(v), 500) if isinstance(v, str) else v for k, v in list(parameters.items())[:50]}
    else:
        parameters = {}
    if _shadow_os_instance is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    with _ESCALATION_MANAGER_LOCK:
        if not hasattr(_shadow_os_instance, '_escalation_manager') or _shadow_os_instance._escalation_manager is None:
            _shadow_os_instance._escalation_manager = EscalationManager()
    can, req = await _shadow_os_instance._escalation_manager.can_execute(action_type, parameters, domain)
    return {"can_execute": can, "action": req.to_dict()}


@app.get("/escalation/rules", dependencies=[Depends(_verify_os_api_key)])
async def get_escalation_rules():
    if _shadow_os_instance is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    with _ESCALATION_MANAGER_LOCK:
        if not hasattr(_shadow_os_instance, '_escalation_manager') or _shadow_os_instance._escalation_manager is None:
            _shadow_os_instance._escalation_manager = EscalationManager()
    return {"action_rules": _shadow_os_instance._escalation_manager.get_action_rules(), "domain_rules": _shadow_os_instance._escalation_manager.get_domain_rules()}



@app.websocket("/ws/live")
async def websocket_live_dashboard(websocket):
    """Real-time dashboard WebSocket - streams module stats every 5s."""
    import json as _json
    await websocket.accept()
    try:
        while True:
            try:
                if _shadow_os_instance and hasattr(_shadow_os_instance, 'dashboard_agg'):
                    snap = _shadow_os_instance.dashboard_agg.get_snapshot(force=True)
                    await websocket.send_text(_json.dumps(snap.to_dict(), default=str))
                elif _shadow_os_instance:
                    health = _shadow_os_instance.get_health()
                    await websocket.send_text(_json.dumps(health, default=str))
                await asyncio.sleep(5)
            except (RuntimeError, ValueError, TypeError, asyncio.TimeoutError):
                break
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/deploy/{shadow_id}")
async def websocket_deploy_feed(websocket, shadow_id: str, auth_token: str = ""):
    """Real-time feed for a deployed shadow (reports + alerts)."""
    import json as _json
    if not _shadow_os_instance or not hasattr(_shadow_os_instance, 'shadow_deploy'):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    shadow = _shadow_os_instance.shadow_deploy.get_shadow(shadow_id)
    if not shadow or (auth_token and shadow.get("auth_token","") != auth_token):
        await websocket.send_text(_json.dumps({"error": "unauthorized"}))
        await websocket.close(code=4003)
        return
    last_report_id = None
    try:
        while True:
            try:
                reports = _shadow_os_instance.shadow_deploy.get_reports(
                    shadow_id=shadow_id, limit=5, alerts_only=False
                )
                if reports and reports[0].get("report_id") != last_report_id:
                    last_report_id = reports[0].get("report_id")
                    await websocket.send_text(_json.dumps({
                        "shadow_id": shadow_id,
                        "recent_reports": reports[:5]
                    }, default=str))
                await asyncio.sleep(10)
            except (RuntimeError, ValueError, TypeError, asyncio.TimeoutError):
                break
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="ShadowOS Daemon")
    parser.add_argument("--host", default=os.environ.get("SHADOW_OS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHADOW_OS_PORT", "8766")))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    uvicorn.run(
        "shadowd.cognition.shadowosd:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()