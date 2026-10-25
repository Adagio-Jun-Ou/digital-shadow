#!/usr/bin/env python3
"""
ShadowOS API Router - FastAPI endpoints for the Shadow Operating System

Adds new cognitive module endpoints to the existing Shadow AI daemon.
All new endpoints are prefixed with /os/ to avoid conflicts with existing ones.

New Endpoints:
  /os/status          - Full ShadowOS status
  /os/health          - Health check for all modules
  /os/graph           - ShadowGraph status & visualization
  /os/graph/process   - Process an event through the cognitive graph
  /os/graph/trace     - Get execution traces
  /os/memory/search   - Search DarkMemory
  /os/memory/recent    - Get recent memories
  /os/memory/blocks   - List/set memory blocks
  /os/kg/query        - Query the temporal knowledge graph
  /os/crew/agents     - List ShadowCrew agents
  /os/crew/mission    - Create & execute a mission
  /os/mesh/status     - NeuralMesh status
  /os/mesh/insight    - Share an insight
  /os/stealth/audit   - Get privacy audit log
  /os/stealth/check   - Check data access
  /os/personality     - Get personality profile
  /os/personality/distill - Run personality distillation
  /os/personality/predict - Predict a decision
  /os/voice/start     - Start voice session
  /os/voice/process   - Process voice input
  /os/voice/end       - End voice session
"""

import threading
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile
from fastapi.responses import Response as FastAPIResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
import os

from .shadow_os import ShadowOS
from .anticipatory_engine import AnticipationContext
from .shadow_arise import ShadowArise, CapabilityType, AgentStatus
from .device_context import DeviceType, LocationType

router = APIRouter(prefix="/os", tags=["ShadowOS"])

_MAX_ELEMENT_LENGTH = 50

_shadow_os: Optional[ShadowOS] = None
_shadow_os_lock = threading.Lock()

_API_KEY = os.environ.get("SHADOW_OS_API_KEY", "")
_auth_enabled = bool(_API_KEY)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not _auth_enabled:
        return True
    if not api_key or api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


_MAX_DICT_KEYS = 50


def _validate_dict_keys(v):
    if v is not None and len(v) > _MAX_DICT_KEYS:
        raise ValueError(f'dict exceeds max {_MAX_DICT_KEYS} keys')
    return v


class EmpathyPersonRequest(BaseModel):
    name: str = Field(..., max_length=100)
    relationship: str = Field(default="acquaintance", max_length=50)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class EmpathyInteractionRequest(BaseModel):
    person_id: str = Field(..., max_length=100)
    interaction_type: str = Field(..., max_length=50)
    sentiment: float = Field(default=0.5, ge=-1.0, le=1.0)
    notes: str = Field(default="", max_length=1000)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class EmpathyPredictRequest(BaseModel):
    person_id: str = Field(..., max_length=100)
    topic: str = Field(..., max_length=500)
    approach: str = Field(default="direct", max_length=100)


class KGEntityRequest(BaseModel):
    entity_id: str = Field(..., max_length=100)
    entity_type: str = Field(..., max_length=50)
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGRelationRequest(BaseModel):
    source_id: str = Field(..., max_length=100)
    target_id: str = Field(..., max_length=100)
    relation_type: str = Field(..., max_length=50)
    properties: Dict[str, Any] = Field(default_factory=dict)


def get_shadow_os() -> ShadowOS:
    global _shadow_os
    if _shadow_os is None:
        with _shadow_os_lock:
            if _shadow_os is None:
                _shadow_os = ShadowOS()
    return _shadow_os


class EventRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=2000)
    domain: Optional[str] = Field(default=None, max_length=100)
    urgency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    event_data: Dict[str, Any] = Field(default_factory=dict, max_length=50)


class ObservationRequest(BaseModel):
    type: str = Field(default="observation", max_length=50)
    content: Optional[str] = Field(default=None, max_length=65536)
    domain: Optional[str] = Field(default=None, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class ActionRequest(BaseModel):
    action: str = Field(..., max_length=500)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
        return _validate_dict_keys(v)


class MissionRequest(BaseModel):
    objective: str = Field(..., max_length=2000)
    mode: str = Field(default="hierarchical", max_length=50)


class MemoryBlockRequest(BaseModel):
    label: str = Field(..., max_length=100)
    value: str = Field(..., max_length=10000)


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    domain: Optional[str] = Field(default=None, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)


class KGQueryRequest(BaseModel):
    entity_id: Optional[str] = Field(default=None, max_length=100)
    relation_type: Optional[str] = Field(default=None, max_length=100)
    depth: int = Field(default=1, ge=1, le=3)


class MeshPeerRequest(BaseModel):
    node_id: str = Field(..., max_length=200)
    name: str = Field(..., max_length=100)
    device_type: str = Field(default="server", max_length=50)
    address: str = Field(default="127.0.0.1", max_length=100)
    # NOTE: Privileged ports (<1024) are intentionally allowed because
    # Tailscale Serve commonly maps services to low ports (e.g. 443, 80).
    # The SSRF is_safe_address() check blocks dangerous metadata/RFC1918 addrs.
    port: int = Field(default=8765, ge=1, le=65535)

    def is_safe_address(self) -> bool:
        import re
        addr = self.address.strip()
        if not re.match(r'^[a-zA-Z0-9.\-]+$', addr):
            return False
        blocked = ['169.254.169.254', 'metadata.google', 'metadata.azure']
        if addr in blocked:
            return False
        if addr.startswith('10.') or addr.startswith('192.168.'):
            return False
        if re.match(r'^172\.(1[6-9]|2[0-9]|3[0-1])\.', addr):
            return False
        if addr == '0.0.0.0':
            return False
        return True


class VoiceProcessRequest(BaseModel):
    transcript: str = Field(..., max_length=5000)


class PersonalityPredictRequest(BaseModel):
    options: List[Dict[str, Any]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class StealthCheckRequest(BaseModel):
    subject: str = Field(..., max_length=200)
    action: str = Field(..., max_length=200)
    data_id: str = Field(..., max_length=200)
    category: str = Field(..., max_length=50)


class TrinityAnalyzeRequest(BaseModel):
    situation: str = Field(..., max_length=5000)


class DreamMemoryRequest(BaseModel):
    content: str = Field(..., max_length=5000)
    domain: str = Field(default="general", max_length=50)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None

    @field_validator('tags', mode='before')
    @classmethod
    def _validate_tags(cls, v):
        if v is not None:
            return [t[:_MAX_ELEMENT_LENGTH] for t in v]
        return v


class PresenceDeviceRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    owner: str = Field(..., max_length=100)
    device_type: str = Field(default="unknown", max_length=20)


class PresenceUpdateRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    status: str = Field(default="active", max_length=20)
    metadata: Optional[Dict[str, Any]] = None


@router.get("/status", dependencies=[Depends(verify_api_key)])
async def os_status():
    os = get_shadow_os()
    return os.get_status()


@router.get("/health")
async def os_health():
    os = get_shadow_os()
    return os.get_health()


@router.get("/health/live")
async def os_liveness():
    return {"status": "alive"}


@router.get("/health/ready")
async def os_readiness():
    os = get_shadow_os()
    try:
        health = os.get_health()
        module_count = len(health.get("modules", {}))
        return {"status": "ready", "modules": module_count}
    except Exception:
        return {"status": "not_ready", "modules": 0}


@router.get("/graph", dependencies=[Depends(verify_api_key)])
async def graph_status():
    os = get_shadow_os()
    return os.graph.get_status()


@router.get("/graph/dot", dependencies=[Depends(verify_api_key)])
async def graph_visualization():
    os = get_shadow_os()
    return {"dot": os.get_graph_visualization()}


@router.post("/graph/process", dependencies=[Depends(verify_api_key)])
async def graph_process(request: EventRequest):
    os = get_shadow_os()
    if request.query:
        result = await os.process_query(request.query, domain=request.domain)
        return result
    else:
        return await os.graph.process_event(
            event=request.event_data,
        )


@router.get("/graph/trace", dependencies=[Depends(verify_api_key)])
async def graph_trace(limit: int = Query(default=10, ge=1, le=100)):
    os = get_shadow_os()
    return {"traces": os.graph.get_trace_summary(limit)}


@router.get("/graph/checkpoints", dependencies=[Depends(verify_api_key)])
async def graph_checkpoints(limit: int = Query(default=10, ge=1, le=100)):
    os = get_shadow_os()
    return {"checkpoints": os.graph.get_checkpoint_summary(limit)}


@router.post("/memory/search", dependencies=[Depends(verify_api_key)])
async def memory_search(request: SearchRequest):
    os = get_shadow_os()
    results = await os.search_memory(request.query, limit=request.limit, domain=request.domain)
    return {"results": results, "count": len(results)}


@router.get("/memory/recent", dependencies=[Depends(verify_api_key)])
async def memory_recent(limit: int = Query(default=20, ge=1, le=100), atom_type: Optional[str] = None):
    os = get_shadow_os()
    from .dark_memory import AtomType
    atom_type_val = AtomType(atom_type) if atom_type else None
    results = await os.memory.get_recent(limit=limit, atom_type=atom_type_val)
    return {"results": [r.to_dict() for r in results], "count": len(results)}


@router.get("/memory/blocks", dependencies=[Depends(verify_api_key)])
async def memory_blocks():
    os = get_shadow_os()
    return {"blocks": os.get_memory_blocks()}


@router.get("/memory/blocks/{label}", dependencies=[Depends(verify_api_key)])
async def memory_block_get(label: str):
    os = get_shadow_os()
    block = os.get_memory_block(label)
    if not block:
        raise HTTPException(status_code=404, detail=f"Block '{label}' not found")
    return block


@router.post("/memory/blocks", dependencies=[Depends(verify_api_key)])
async def memory_block_set(request: MemoryBlockRequest):
    os = get_shadow_os()
    if len(request.label) > 100 or len(request.value) > 10000:
        raise HTTPException(status_code=400, detail="Label or value too long")
    success = os.set_memory_block(request.label, request.value)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set block (may be protected)")
    return {"status": "ok", "label": request.label}


@router.get("/memory/stats", dependencies=[Depends(verify_api_key)])
async def memory_stats():
    os = get_shadow_os()
    return os.memory.get_stats()


@router.post("/kg/query", dependencies=[Depends(verify_api_key)])
async def kg_query(request: KGQueryRequest):
    os = get_shadow_os()
    result = await os.memory.query_kg(
        entity_id=request.entity_id,
        relation_type=request.relation_type,
        depth=request.depth,
    )
    return result


@router.post("/kg/entity", dependencies=[Depends(verify_api_key)])
async def kg_add_entity(request: KGEntityRequest):
    os = get_shadow_os()
    result = await os.memory.store_entity(request.entity_id, request.entity_type, request.properties)
    return {"entity_id": result}


@router.post("/kg/relation", dependencies=[Depends(verify_api_key)])
async def kg_add_relation(request: KGRelationRequest):
    os = get_shadow_os()
    result = await os.memory.store_relation(request.source_id, request.target_id, request.relation_type, request.properties)
    return {"relation_id": result}


@router.get("/crew/agents", dependencies=[Depends(verify_api_key)])
async def crew_agents():
    os = get_shadow_os()
    return {"agents": os.get_crew_agents()}


@router.get("/crew/missions", dependencies=[Depends(verify_api_key)])
async def crew_missions(limit: int = Query(default=20, ge=1, le=200)):
    os = get_shadow_os()
    return {"missions": os.crew.list_missions(limit)}


@router.post("/crew/mission", dependencies=[Depends(verify_api_key)])
async def crew_create_mission(request: MissionRequest):
    os = get_shadow_os()
    from .shadow_crew import DelegationMode
    try:
        mode = DelegationMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
    mission = await os.create_mission(request.objective, mode)
    return mission


@router.post("/crew/mission/{mission_id}/execute", dependencies=[Depends(verify_api_key)])
async def crew_execute_mission(mission_id: str):
    os = get_shadow_os()
    result = await os.execute_mission(mission_id)
    return result


@router.get("/mesh/status", dependencies=[Depends(verify_api_key)])
async def mesh_status():
    os = get_shadow_os()
    return os.mesh.get_stats()


@router.get("/mesh/peers", dependencies=[Depends(verify_api_key)])
async def mesh_peers():
    os = get_shadow_os()
    return {"peers": os.get_mesh_peers()}


class MeshInsightRequest(BaseModel):
    insight: Dict[str, Any] = Field(default_factory=dict)


@router.post("/mesh/insight", dependencies=[Depends(verify_api_key)])
async def mesh_share_insight(request: MeshInsightRequest):
    os = get_shadow_os()
    count = await os.mesh.share_insight(request.insight)
    return {"propagated_to": count}


@router.post("/mesh/peer", dependencies=[Depends(verify_api_key)])
async def mesh_add_peer(request: MeshPeerRequest):
    if not request.is_safe_address():
        raise HTTPException(status_code=400, detail="Invalid peer address")
    os = get_shadow_os()
    from .neural_mesh import MeshNode
    node = MeshNode(
        node_id=request.node_id, name=request.name, device_type=request.device_type,
        address=request.address, port=request.port,
    )
    try:
        os.mesh.add_peer(node)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {type(e).__name__}")
    return {"status": "added", "node_id": request.node_id}


@router.get("/stealth/audit", dependencies=[Depends(verify_api_key)])
async def stealth_audit(subject: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)):
    os = get_shadow_os()
    return {"audit_log": os.get_audit_log(subject, limit)}


@router.post("/stealth/check", dependencies=[Depends(verify_api_key)])
async def stealth_check(request: StealthCheckRequest):
    os = get_shadow_os()
    from .stealth_layer import DataCategory, AccessAction
    try:
        category = DataCategory(request.category)
        action = AccessAction(request.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid category or action: {type(e).__name__}")
    granted, reason = os.stealth.check_access(request.subject, action, request.data_id, category)
    return {"granted": granted, "reason": reason}


@router.get("/stealth/stats", dependencies=[Depends(verify_api_key)])
async def stealth_stats():
    os = get_shadow_os()
    return os.stealth.get_stats()


@router.get("/personality", dependencies=[Depends(verify_api_key)])
async def personality_profile():
    os = get_shadow_os()
    return os.get_personality()


@router.post("/personality/distill", dependencies=[Depends(verify_api_key)])
async def personality_distill(auth: bool = Depends(verify_api_key)):
    os = get_shadow_os()
    result = await os.personality.distill()
    return result


@router.post("/personality/predict", dependencies=[Depends(verify_api_key)])
async def personality_predict(request: PersonalityPredictRequest):
    os = get_shadow_os()
    result = os.personality.predict_decision(request.options, request.context)
    return result


@router.get("/personality/patterns", dependencies=[Depends(verify_api_key)])
async def personality_patterns(min_confidence: float = Query(default=0.0)):
    os = get_shadow_os()
    return {"patterns": os.personality.get_patterns(min_confidence)}


@router.get("/voice/status", dependencies=[Depends(verify_api_key)])
async def voice_status():
    os = get_shadow_os()
    return os.voice.get_state()


class VoiceStartRequest(BaseModel):
    voice: Optional[str] = Field(default=None, max_length=100)


@router.post("/voice/start", dependencies=[Depends(verify_api_key)])
async def voice_start(request: VoiceStartRequest = None):
    os = get_shadow_os()
    session = await os.start_voice_session(request.voice if request else None)
    return session


@router.post("/voice/process", dependencies=[Depends(verify_api_key)])
async def voice_process(request: VoiceProcessRequest):
    os = get_shadow_os()
    result = await os.process_voice(request.transcript)
    return result


@router.post("/voice/end", dependencies=[Depends(verify_api_key)])
async def voice_end(auth: bool = Depends(verify_api_key)):
    os = get_shadow_os()
    result = await os.end_voice_session()
    if not result:
        raise HTTPException(status_code=404, detail="No active voice session")
    return result


@router.get("/voice/sessions", dependencies=[Depends(verify_api_key)])
async def voice_sessions(limit: int = Query(default=20)):
    os = get_shadow_os()
    return {"sessions": os.voice.list_sessions(limit)}


class VoiceSTTRequest(BaseModel):
    """Schema for the /voice/stt route's query params. The route uses
    `language: str = Query(...)` directly so this model is only used by
    code that wants a Pydantic surface (tests, CLI tooling)."""
    language: str = Field(default="en", max_length=10)


class VoiceTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: Optional[str] = Field(default=None, max_length=100)


@router.post("/voice/tts", dependencies=[Depends(verify_api_key)])
async def voice_tts(request: VoiceTTSRequest):
    os = get_shadow_os()
    audio_bytes = await os.voice.speak(request.text, voice=request.voice)
    if not audio_bytes:
        raise HTTPException(status_code=503, detail="TTS engine unavailable")
    return FastAPIResponse(content=audio_bytes, media_type="audio/mpeg")


@router.get("/voice/voices", dependencies=[Depends(verify_api_key)])
async def voice_list_voices():
    os = get_shadow_os()
    voices = await os.voice.list_voices()
    return {"voices": voices}


@router.post("/voice/stt", dependencies=[Depends(verify_api_key)])
async def voice_stt(language: str = Query(default="en"), file: UploadFile = None):
    os = get_shadow_os()
    if not file:
        raise HTTPException(status_code=400, detail="Audio file required")
    audio_data = await file.read()
    transcript = await os.voice.listen(audio_data, language=language)
    if not transcript:
        raise HTTPException(status_code=503, detail="STT engine unavailable or no speech detected")
    return {"transcript": transcript}


@router.post("/voice/stt-upload", dependencies=[Depends(verify_api_key)])
async def voice_stt_upload(language: str = Query(default="en"), file: UploadFile = None):
    os = get_shadow_os()
    if not file:
        raise HTTPException(status_code=400, detail="Audio file required")
    audio_data = await file.read()
    transcript = await os.voice.listen(audio_data, language=language)
    if not transcript:
        raise HTTPException(status_code=503, detail="STT engine unavailable or no speech detected")
    return {"transcript": transcript}


@router.post("/observe", dependencies=[Depends(verify_api_key)])
async def os_observe(request: ObservationRequest):
    os = get_shadow_os()
    observation = {
        "type": request.type,
        "content": request.content if request.content is not None else "",
        "domain": request.domain,
        "metadata": request.metadata,
    }
    result = await os.process_observation(observation)
    return result


@router.post("/act", dependencies=[Depends(verify_api_key)])
async def os_act(request: ActionRequest):
    os = get_shadow_os()
    result = await os.execute_action(request.action, request.parameters)
    return result


@router.post("/query", dependencies=[Depends(verify_api_key)])
async def os_query(request: EventRequest):
    os = get_shadow_os()
    result = await os.process_query(request.query or "", domain=request.domain)
    return result


@router.get("/improvement/stats", dependencies=[Depends(verify_api_key)])
async def improvement_stats():
    from .self_improvement import SelfImprovementEngine
    if hasattr(get_shadow_os(), '_improvement_engine'):
        return get_shadow_os()._improvement_engine.get_stats()
    return {"status": "not_initialized"}


@router.post("/improvement/cycle", dependencies=[Depends(verify_api_key)])
async def improvement_cycle(auth: bool = Depends(verify_api_key)):
    from .self_improvement import SelfImprovementEngine
    if hasattr(get_shadow_os(), '_improvement_engine'):
        report = await get_shadow_os()._improvement_engine.run_improvement_cycle()
        return report
    return {"status": "not_initialized"}


@router.get("/improvement/decisions", dependencies=[Depends(verify_api_key)])
async def improvement_decisions(limit: int = Query(default=20)):
    from .self_improvement import SelfImprovementEngine
    if hasattr(get_shadow_os(), '_improvement_engine'):
        return {"decisions": get_shadow_os()._improvement_engine.get_recent_decisions(limit)}
    return {"decisions": []}


@router.get("/improvement/prediction-accuracy", dependencies=[Depends(verify_api_key)])
async def prediction_accuracy():
    from .self_improvement import SelfImprovementEngine
    if hasattr(get_shadow_os(), '_improvement_engine'):
        return {"calibrator": get_shadow_os()._improvement_engine.calibrator.get_stats()}
    return {"status": "not_initialized"}


# ═════════════════════════════════════════════════════════════════
# ANOMALY DETECTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════

class AnomalyObserveRequest(BaseModel):
    metric: str = Field(..., max_length=100)
    value: float
    context: Optional[Dict[str, Any]] = Field(default=None, max_length=20)

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        return _validate_dict_keys(v)


class AnomalyTemporalRequest(BaseModel):
    event_type: str = Field(..., max_length=100)


class AnomalyAcknowledgeRequest(BaseModel):
    anomaly_id: str = Field(..., max_length=64)


@router.get("/anomaly/stats", dependencies=[Depends(verify_api_key)])
async def anomaly_stats():
    os = get_shadow_os()
    if os and hasattr(os, 'anomaly_detector'):
        return {
            "anomaly_detector": os.anomaly_detector.get_stats(),
            "streams": os.anomaly_detector.get_stream_stats(),
        }
    return {"status": "not_initialized"}


@router.get("/anomaly/list", dependencies=[Depends(verify_api_key)])
async def anomaly_list(
    severity: Optional[str] = Query(default=None),
    anomaly_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    unacknowledged_only: bool = Query(default=False),
):
    os = get_shadow_os()
    if not os or not hasattr(os, 'anomaly_detector'):
        return {"anomalies": []}
    from .anomaly_detector import AnomalySeverity, AnomalyType
    try:
        sev = AnomalySeverity(severity) if severity else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    try:
        atype = AnomalyType(anomaly_type) if anomaly_type else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid anomaly_type: {anomaly_type}")
    return {"anomalies": os.anomaly_detector.get_anomalies(
        severity=sev, anomaly_type=atype, limit=limit,
        unacknowledged_only=unacknowledged_only,
    )}


@router.post("/anomaly/observe", dependencies=[Depends(verify_api_key)])
async def anomaly_observe(request: AnomalyObserveRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'anomaly_detector'):
        return {"status": "not_initialized"}
    anomaly = os.anomaly_detector.observe(request.metric, request.value, request.context)
    if anomaly:
        return {"anomaly": anomaly.to_dict()}
    return {"anomaly": None, "status": "normal"}


@router.post("/anomaly/temporal", dependencies=[Depends(verify_api_key)])
async def anomaly_temporal(request: AnomalyTemporalRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'anomaly_detector'):
        return {"status": "not_initialized"}
    anomaly = os.anomaly_detector.observe_temporal(request.event_type)
    if anomaly:
        return {"anomaly": anomaly.to_dict()}
    return {"anomaly": None, "status": "normal"}


@router.post("/anomaly/acknowledge", dependencies=[Depends(verify_api_key)])
async def anomaly_acknowledge(request: AnomalyAcknowledgeRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'anomaly_detector'):
        return {"status": "not_initialized"}
    found = os.anomaly_detector.acknowledge(request.anomaly_id)
    return {"acknowledged": found}


# ═════════════════════════════════════════════════════════════════
# GOAL PLANNER ENDPOINTS
# ═════════════════════════════════════════════════════════════════

class GoalCreateRequest(BaseModel):
    title: str = Field(max_length=200)
    description: str = Field(default="", max_length=500)
    priority: str = Field(default="medium", max_length=20)
    domain: Optional[str] = Field(default=None, max_length=100)
    deadline: Optional[str] = Field(default=None, max_length=30)
    tags: Optional[List[str]] = None

    @field_validator('tags', mode='before')
    @classmethod
    def _validate_tags(cls, v):
        if v is not None:
            return [t[:_MAX_ELEMENT_LENGTH] for t in v]
        return v


class GoalDecomposeRequest(BaseModel):
    goal_id: str = Field(..., max_length=64)
    steps: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        if v is not None:
            for step in v:
                if len(step) > _MAX_DICT_KEYS:
                    raise ValueError(f'step exceeds max {_MAX_DICT_KEYS} keys')
        return v


class StepActionRequest(BaseModel):
    goal_id: str = Field(..., max_length=64)
    step_id: str = Field(..., max_length=64)


@router.post("/goals/create", dependencies=[Depends(verify_api_key)])
async def goal_create(request: GoalCreateRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    from .goal_planner import GoalPriority
    try:
        priority = GoalPriority(request.priority)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
    goal = os.goal_planner.create_goal(
        title=request.title,
        description=request.description,
        priority=priority,
        domain=request.domain,
        deadline=request.deadline,
        tags=request.tags,
    )
    return goal.to_dict()


@router.get("/goals/list", dependencies=[Depends(verify_api_key)])
async def goals_list(
    status: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
    limit: int = Query(default=50),
):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        return {"goals": []}
    from .goal_planner import GoalStatus
    try:
        gs = GoalStatus(status) if status else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    return {"goals": os.goal_planner.list_goals(status=gs, domain=domain, limit=limit)}


@router.get("/goals/{goal_id}", dependencies=[Depends(verify_api_key)])
async def goal_get(goal_id: str):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    result = os.goal_planner.get_goal(goal_id)
    if not result:
        raise HTTPException(status_code=404, detail="Goal not found")
    return result


@router.post("/goals/decompose", dependencies=[Depends(verify_api_key)])
async def goal_decompose(request: GoalDecomposeRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    goal = os.goal_planner.decompose_goal(request.goal_id, request.steps)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal.to_dict()


@router.post("/goals/start-step", dependencies=[Depends(verify_api_key)])
async def goal_start_step(request: StepActionRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    success = os.goal_planner.start_step(request.goal_id, request.step_id)
    return {"success": success}


@router.post("/goals/complete-step", dependencies=[Depends(verify_api_key)])
async def goal_complete_step(request: StepActionRequest):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    success = os.goal_planner.complete_step(request.goal_id, request.step_id)
    return {"success": success}


@router.post("/goals/cancel", dependencies=[Depends(verify_api_key)])
async def goal_cancel(goal_id: str = Query(...)):
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        raise HTTPException(status_code=503, detail="Goal planner not initialized")
    success = os.goal_planner.cancel_goal(goal_id)
    return {"success": success}


@router.get("/goals/stats", dependencies=[Depends(verify_api_key)])
async def goal_stats():
    os = get_shadow_os()
    if not os or not hasattr(os, 'goal_planner'):
        return {"status": "not_initialized"}
    return os.goal_planner.get_stats()


# ═══════════════════════════════════════════════════════════════════
# THOUGHT SENSE ENDPOINTS
# ═════════════════════════════════════════════════════════════════

class ThoughtAnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=10000)


@router.post("/thought/analyze", dependencies=[Depends(verify_api_key)])
async def thought_analyze(request: ThoughtAnalyzeRequest):
    os = get_shadow_os()
    result = os.thought_sense.analyze(request.text)
    return result.to_dict()


@router.get("/thought/stats", dependencies=[Depends(verify_api_key)])
async def thought_stats():
    os = get_shadow_os()
    return os.thought_sense.get_stats()


# ═══════════════════════════════════════════════════════════════════
# RECALL SYNTHESIS ENDPOINTS
# ═════════════════════════════════════════════════════════════════

class RecallObserveRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    domain: Optional[str] = Field(default=None, max_length=50)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class RecallSynthesizeRequest(BaseModel):
    query: str = Field(..., max_length=500)
    domain: Optional[str] = Field(default=None, max_length=50)


@router.post("/recall/observe", dependencies=[Depends(verify_api_key)])
async def recall_observe(request: RecallObserveRequest):
    os = get_shadow_os()
    obs_id = os.recall_synthesis.add_observation(
        text=request.text, domain=request.domain, importance=request.importance
    )
    return {"observation_id": obs_id}


@router.post("/recall/synthesize", dependencies=[Depends(verify_api_key)])
async def recall_synthesize(request: RecallSynthesizeRequest):
    os = get_shadow_os()
    result = os.recall_synthesis.synthesize(query=request.query, domain=request.domain)
    return result.to_dict()


@router.get("/recall/patterns", dependencies=[Depends(verify_api_key)])
async def recall_patterns():
    os = get_shadow_os()
    patterns = os.recall_synthesis.find_cross_domain_patterns()
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/recall/insights", dependencies=[Depends(verify_api_key)])
async def recall_insights(domain: Optional[str] = None, min_confidence: float = 0.3):
    os = get_shadow_os()
    insights = os.recall_synthesis.get_insights(domain=domain, min_confidence=min_confidence)
    return {"insights": [i.to_dict() for i in insights]}


# ═════════════════════════════════════════════════════════════════
# TEMPORAL REASONING ENDPOINTS
# ═════════════════════════════════════════════════════════════════

class TemporalRecordRequest(BaseModel):
    event_type: str = Field(..., max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


@router.post("/temporal/record", dependencies=[Depends(verify_api_key)])
async def temporal_record(request: TemporalRecordRequest):
    os = get_shadow_os()
    event_id = os.temporal_reasoning.record_event(
        event_type=request.event_type, metadata=request.metadata
    )
    return {"event_id": event_id}


@router.get("/temporal/patterns", dependencies=[Depends(verify_api_key)])
async def temporal_patterns(event_type: Optional[str] = None):
    os = get_shadow_os()
    patterns = os.temporal_reasoning.detect_patterns(event_type=event_type)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/temporal/predict/{event_type}", dependencies=[Depends(verify_api_key)])
async def temporal_predict(event_type: str, within_hours: float = 48):
    os = get_shadow_os()
    predictions = os.temporal_reasoning.predict_next(event_type, within_hours=within_hours)
    return {"predictions": [p.to_dict() for p in predictions]}


@router.get("/temporal/trend/{event_type}", dependencies=[Depends(verify_api_key)])
async def temporal_trend(event_type: str, window_days: int = 30):
    os = get_shadow_os()
    trend = os.temporal_reasoning.analyze_trend(event_type, window_days=window_days)
    return trend.to_dict()


@router.get("/temporal/stats", dependencies=[Depends(verify_api_key)])
async def temporal_stats():
    os = get_shadow_os()
    return os.temporal_reasoning.get_stats()


# ═══════════════════════════════════════════════════════════════════
# EMPATHY MAPPING ENDPOINTS
# ═════════════════════════════════════════════════════════════════

@router.post("/empathy/person", dependencies=[Depends(verify_api_key)])
async def empathy_add_person(request: EmpathyPersonRequest):
    os = get_shadow_os()
    person_id = os.add_person(request.name, request.relationship, request.metadata)
    return {"person_id": person_id}


@router.post("/empathy/interaction", dependencies=[Depends(verify_api_key)])
async def empathy_record_interaction(request: EmpathyInteractionRequest):
    os = get_shadow_os()
    interaction_id = os.empathy.record_interaction(
        request.person_id, request.interaction_type, request.sentiment,
        request.notes, request.metadata
    )
    return {"interaction_id": interaction_id}


@router.get("/empathy/profile/{person_id}", dependencies=[Depends(verify_api_key)])
async def empathy_profile(person_id: str):
    if len(person_id) > 100:
        raise HTTPException(status_code=400, detail="person_id too long")
    os = get_shadow_os()
    profile = os.empathy.get_person_profile(person_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return profile.to_dict()


@router.get("/empathy/communication/{person_id}", dependencies=[Depends(verify_api_key)])
async def empathy_communication(person_id: str):
    if len(person_id) > 100:
        raise HTTPException(status_code=400, detail="person_id too long")
    os = get_shadow_os()
    style = os.empathy.get_communication_style(person_id)
    return style.to_dict()


@router.get("/empathy/empathy-score/{person_id}", dependencies=[Depends(verify_api_key)])
async def empathy_score(person_id: str):
    os = get_shadow_os()
    score = os.empathy.get_empathy_score(person_id)
    return {"person_id": person_id, "empathy_score": score}


@router.post("/empathy/predict", dependencies=[Depends(verify_api_key)])
async def empathy_predict(request: EmpathyPredictRequest):
    os = get_shadow_os()
    prediction = os.empathy.predict_reaction(request.person_id, request.topic, request.approach)
    return prediction.to_dict()


@router.get("/empathy/health/{person_id}", dependencies=[Depends(verify_api_key)])
async def empathy_health(person_id: str):
    os = get_shadow_os()
    health = os.get_relationship_health(person_id)
    return health


@router.get("/empathy/graph", dependencies=[Depends(verify_api_key)])
async def empathy_graph():
    os = get_shadow_os()
    return os.empathy.get_social_graph()


@router.get("/empathy/stats", dependencies=[Depends(verify_api_key)])
async def empathy_stats():
    os = get_shadow_os()
    return os.empathy.get_stats()


# ═══════════════════════════════════════════════════════════════════
# COGNITIVE TRINITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/trinity/analyze", dependencies=[Depends(verify_api_key)])
async def trinity_analyze(request: TrinityAnalyzeRequest):
    os = get_shadow_os()
    result = os.trinity.analyze(request.situation)
    return result.to_dict()


@router.get("/trinity/stats", dependencies=[Depends(verify_api_key)])
async def trinity_stats():
    os = get_shadow_os()
    return os.trinity.get_stats()


# ═══════════════════════════════════════════════════════════════════
# PRESENCE DETECTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/presence/device", dependencies=[Depends(verify_api_key)])
async def presence_register(request: PresenceDeviceRequest):
    os = get_shadow_os()
    device_id = os.presence.register_device(request.device_id, request.owner, request.device_type)
    return {"device_id": device_id}


@router.post("/presence/update", dependencies=[Depends(verify_api_key)])
async def presence_update(request: PresenceUpdateRequest):
    os = get_shadow_os()
    os.presence.update_presence(request.device_id, request.status, request.metadata)
    return {"status": "updated"}


@router.get("/presence/people", dependencies=[Depends(verify_api_key)])
async def presence_people():
    os = get_shadow_os()
    people = os.presence.get_present_people()
    return {"people": [p.to_dict() for p in people]}


@router.get("/presence/environment", dependencies=[Depends(verify_api_key)])
async def presence_environment():
    os = get_shadow_os()
    env = os.presence.detect_environment()
    return env.to_dict()


@router.get("/presence/privacy", dependencies=[Depends(verify_api_key)])
async def presence_privacy():
    os = get_shadow_os()
    mode = os.presence.get_privacy_mode()
    return {"privacy_mode": mode.value if hasattr(mode, 'value') else str(mode)}


# ═══════════════════════════════════════════════════════════════════
# DREAM STATE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/dream/memory", dependencies=[Depends(verify_api_key)])
async def dream_add_memory(request: DreamMemoryRequest):
    os = get_shadow_os()
    memory_id = os.dream.add_memory(request.content, request.domain, request.importance, request.tags)
    return {"memory_id": memory_id}


@router.post("/dream/consolidate", dependencies=[Depends(verify_api_key)])
async def dream_consolidate():
    os = get_shadow_os()
    result = os.consolidate_memories()
    return result


@router.get("/dream/insights", dependencies=[Depends(verify_api_key)])
async def dream_insights():
    os = get_shadow_os()
    insights = os.dream.get_insights()
    return {"insights": [i.to_dict() for i in insights]}


@router.get("/dream/patterns", dependencies=[Depends(verify_api_key)])
async def dream_patterns():
    os = get_shadow_os()
    patterns = os.dream.get_patterns()
    return {"patterns": [p.to_dict() for p in patterns] if patterns else []}


@router.get("/dream/recall", dependencies=[Depends(verify_api_key)])
async def dream_recall(query: str = "", domain: Optional[str] = None, limit: int = Query(default=10, ge=1, le=100)):
    limit = max(1, min(100, limit))
    os = get_shadow_os()
    memories = os.dream.recall(query, domain, limit)
    return {"memories": [m.to_dict() for m in memories]}


# ═════════════════════════════════════════════════════════════════════
# ADAPTIVE SKILL TREE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class SkillAddRequest(BaseModel):
    skill_id: str = Field(..., max_length=64)
    name: str = Field(..., max_length=100)
    domain: str = Field(..., max_length=50)
    dependencies: Optional[List[str]] = None
    initial_xp: float = Field(default=0.0, ge=0.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('dependencies', mode='before')
    @classmethod
    def _validate_deps(cls, v):
        if v is not None:
            return [d[:_MAX_ELEMENT_LENGTH] for d in v]
        return v


class SkillUseRequest(BaseModel):
    skill_id: str = Field(..., max_length=64)
    success: bool = Field(default=True)
    context: Optional[Dict[str, Any]] = None


@router.post("/skills/add", dependencies=[Depends(verify_api_key)])
async def skill_add(request: SkillAddRequest):
    os = get_shadow_os()
    try:
        node = os.skills.add_skill(
            skill_id=request.skill_id,
            name=request.name,
            domain=request.domain,
            dependencies=request.dependencies,
            initial_xp=request.initial_xp,
            metadata=request.metadata,
        )
        return node.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/skills/use", dependencies=[Depends(verify_api_key)])
async def skill_use(request: SkillUseRequest):
    os = get_shadow_os()
    try:
        node = os.skills.use_skill(request.skill_id, request.success, request.context)
        return node.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.get("/skills/{skill_id}", dependencies=[Depends(verify_api_key)])
async def skill_get(skill_id: str):
    os = get_shadow_os()
    skill = os.skills.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill.to_dict()


@router.get("/skills", dependencies=[Depends(verify_api_key)])
async def skill_list(domain: Optional[str] = None):
    os = get_shadow_os()
    skills = os.skills.list_skills(domain=domain)
    return {"skills": [s.to_dict() for s in skills]}


@router.get("/skills/recommendations", dependencies=[Depends(verify_api_key)])
async def skill_recommendations(limit: int = Query(default=10, ge=1, le=50)):
    os = get_shadow_os()
    recs = os.skills.get_recommendations(limit=limit)
    return {"recommendations": [r.to_dict() for r in recs]}


@router.get("/skills/domain-summary", dependencies=[Depends(verify_api_key)])
async def skill_domain_summary():
    os = get_shadow_os()
    return os.skills.get_domain_summary()


@router.get("/skills/level-ups", dependencies=[Depends(verify_api_key)])
async def skill_level_ups(skill_id: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    events = os.skills.get_level_ups(skill_id=skill_id, limit=limit)
    return {"level_ups": [e.to_dict() for e in events]}


@router.delete("/skills/{skill_id}", dependencies=[Depends(verify_api_key)])
async def skill_remove(skill_id: str):
    os = get_shadow_os()
    removed = os.skills.remove_skill(skill_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "removed", "skill_id": skill_id}


@router.get("/skill-tree", dependencies=[Depends(verify_api_key)])
async def skill_tree():
    os = get_shadow_os()
    return os.skills.get_skill_tree()


@router.get("/skills/stats", dependencies=[Depends(verify_api_key)])
async def skill_stats():
    os = get_shadow_os()
    return os.skills.get_stats()


# ═════════════════════════════════════════════════════════════════════
# NARRATIVE MEMORY ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class NarrativeEventRequest(BaseModel):
    content: str = Field(..., max_length=5000)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class NarrativeChapterRequest(BaseModel):
    title: str = Field(..., max_length=200)
    theme: str = Field(..., max_length=50)
    summary: str = Field(default="", max_length=1000)


class NarrativeChapterEventRequest(BaseModel):
    chapter_id: str = Field(..., max_length=64)
    event_id: str = Field(..., max_length=64)


class NarrativeArcRequest(BaseModel):
    title: str = Field(..., max_length=200)
    themes: Optional[List[str]] = None

    @field_validator('themes', mode='before')
    @classmethod
    def _validate_themes(cls, v):
        if v is not None:
            return [t[:_MAX_ELEMENT_LENGTH] for t in v]
        return v


class NarrativeArcChapterRequest(BaseModel):
    arc_id: str = Field(..., max_length=64)
    chapter_id: str = Field(..., max_length=64)


class NarrativeArcResolveRequest(BaseModel):
    arc_id: str = Field(..., max_length=64)
    resolution: str = Field(..., max_length=500)


class NarrativeSearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/narrative/event", dependencies=[Depends(verify_api_key)])
async def narrative_add_event(request: NarrativeEventRequest):
    os = get_shadow_os()
    event = os.narrative.add_event(request.content, request.importance, request.metadata)
    return event.to_dict()


@router.post("/narrative/chapter", dependencies=[Depends(verify_api_key)])
async def narrative_create_chapter(request: NarrativeChapterRequest):
    os = get_shadow_os()
    chapter = os.narrative.create_chapter(request.title, request.theme, request.summary)
    return chapter.to_dict()


@router.post("/narrative/chapter/add-event", dependencies=[Depends(verify_api_key)])
async def narrative_chapter_add_event(request: NarrativeChapterEventRequest):
    os = get_shadow_os()
    event = os.narrative.search_events(request.event_id, limit=1)
    if not event:
        event_obj = os.narrative.add_event(request.event_id, 0.5)
    else:
        event_obj = event[0]
    success = os.narrative.add_event_to_chapter(request.chapter_id, event_obj)
    if not success:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return {"status": "added", "event_id": event_obj.event_id}


@router.get("/narrative/chapter/{chapter_id}", dependencies=[Depends(verify_api_key)])
async def narrative_get_chapter(chapter_id: str):
    os = get_shadow_os()
    chapter = os.narrative.get_chapter(chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return chapter.to_dict()


@router.post("/narrative/arc", dependencies=[Depends(verify_api_key)])
async def narrative_create_arc(request: NarrativeArcRequest):
    os = get_shadow_os()
    arc = os.narrative.create_arc(request.title, request.themes)
    return arc.to_dict()


@router.post("/narrative/arc/add-chapter", dependencies=[Depends(verify_api_key)])
async def narrative_arc_add_chapter(request: NarrativeArcChapterRequest):
    os = get_shadow_os()
    success = os.narrative.add_chapter_to_arc(request.arc_id, request.chapter_id)
    if not success:
        raise HTTPException(status_code=404, detail="Arc or chapter not found")
    return {"status": "added"}


@router.post("/narrative/arc/resolve", dependencies=[Depends(verify_api_key)])
async def narrative_arc_resolve(request: NarrativeArcResolveRequest):
    os = get_shadow_os()
    success = os.narrative.resolve_arc(request.arc_id, request.resolution)
    if not success:
        raise HTTPException(status_code=404, detail="Arc not found")
    return {"status": "resolved"}


@router.get("/narrative/arc/{arc_id}", dependencies=[Depends(verify_api_key)])
async def narrative_get_arc(arc_id: str):
    os = get_shadow_os()
    arc = os.narrative.get_arc(arc_id)
    if not arc:
        raise HTTPException(status_code=404, detail="Arc not found")
    return arc.to_dict()


@router.get("/narrative/chapters", dependencies=[Depends(verify_api_key)])
async def narrative_list_chapters(theme: Optional[str] = None):
    os = get_shadow_os()
    chapters = os.narrative.list_chapters(theme=theme)
    return {"chapters": [c.to_dict() for c in chapters]}


@router.get("/narrative/arcs", dependencies=[Depends(verify_api_key)])
async def narrative_list_arcs():
    os = get_shadow_os()
    arcs = os.narrative.list_arcs()
    return {"arcs": [a.to_dict() for a in arcs]}


@router.get("/narrative/themes", dependencies=[Depends(verify_api_key)])
async def narrative_themes(limit: int = Query(default=10, ge=1, le=50)):
    os = get_shadow_os()
    themes = os.narrative.get_dominant_themes(limit=limit)
    return {"themes": [{"theme": t, "count": c} for t, c in themes]}


@router.post("/narrative/search", dependencies=[Depends(verify_api_key)])
async def narrative_search(request: NarrativeSearchRequest):
    os = get_shadow_os()
    events = os.narrative.search_events(request.query, limit=request.limit)
    return {"events": [e.to_dict() for e in events]}


@router.get("/narrative/summary", dependencies=[Depends(verify_api_key)])
async def narrative_summary():
    os = get_shadow_os()
    summary = os.narrative.generate_summary()
    return summary.to_dict()


@router.get("/narrative/stats", dependencies=[Depends(verify_api_key)])
async def narrative_stats():
    os = get_shadow_os()
    return os.narrative.get_stats()


# ═════════════════════════════════════════════════════════════════════
# PROCEDURAL MEMORY ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ProcedureAddRequest(BaseModel):
    name: str = Field(..., max_length=200)
    category: str = Field(..., max_length=50)
    steps: Optional[List[str]] = None

    @field_validator('steps', mode='before')
    @classmethod
    def _validate_steps(cls, v):
        if v is not None:
            return [s[:_MAX_ELEMENT_LENGTH] for s in v]
        return v


class ProcedurePracticeRequest(BaseModel):
    procedure_id: str = Field(..., max_length=64)
    success: bool = Field(default=True)


class HabitAddRequest(BaseModel):
    name: str = Field(..., max_length=200)
    cue: str = Field(..., max_length=200)
    routine: str = Field(..., max_length=200)
    reward: str = Field(..., max_length=200)


class HabitIdRequest(BaseModel):
    habit_id: str = Field(..., max_length=64)


class RoutineAddRequest(BaseModel):
    name: str = Field(..., max_length=200)
    frequency: str = Field(default="daily", max_length=20)


class RoutineIdRequest(BaseModel):
    routine_id: str = Field(..., max_length=64)


@router.post("/procedural/procedure", dependencies=[Depends(verify_api_key)])
async def procedural_add_procedure(request: ProcedureAddRequest):
    os = get_shadow_os()
    try:
        proc = os.procedural.add_procedure(request.name, request.category, request.steps)
        return proc.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/procedural/practice", dependencies=[Depends(verify_api_key)])
async def procedural_practice(request: ProcedurePracticeRequest):
    os = get_shadow_os()
    try:
        proc = os.procedural.practice_procedure(request.procedure_id, request.success)
        return proc.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.get("/procedural/procedure/{procedure_id}", dependencies=[Depends(verify_api_key)])
async def procedural_get_procedure(procedure_id: str):
    os = get_shadow_os()
    proc = os.procedural.get_procedure(procedure_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return proc.to_dict()


@router.get("/procedural/procedures", dependencies=[Depends(verify_api_key)])
async def procedural_list_procedures(category: Optional[str] = None):
    os = get_shadow_os()
    procs = os.procedural.list_procedures(category=category)
    return {"procedures": [p.to_dict() for p in procs]}


@router.get("/procedural/mastery", dependencies=[Depends(verify_api_key)])
async def procedural_mastery():
    os = get_shadow_os()
    return os.procedural.get_mastery_progress()


@router.post("/procedural/habit", dependencies=[Depends(verify_api_key)])
async def procedural_add_habit(request: HabitAddRequest):
    os = get_shadow_os()
    habit = os.procedural.add_habit(request.name, request.cue, request.routine, request.reward)
    return habit.to_dict()


@router.post("/procedural/habit/complete", dependencies=[Depends(verify_api_key)])
async def procedural_complete_habit(request: HabitIdRequest):
    os = get_shadow_os()
    try:
        habit = os.procedural.complete_habit(request.habit_id)
        return habit.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.post("/procedural/habit/miss", dependencies=[Depends(verify_api_key)])
async def procedural_miss_habit(request: HabitIdRequest):
    os = get_shadow_os()
    try:
        habit = os.procedural.miss_habit(request.habit_id)
        return habit.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.get("/procedural/habit/{habit_id}", dependencies=[Depends(verify_api_key)])
async def procedural_get_habit(habit_id: str):
    os = get_shadow_os()
    habit = os.procedural.get_habit(habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found")
    return habit.to_dict()


@router.get("/procedural/habits", dependencies=[Depends(verify_api_key)])
async def procedural_list_habits(stage: Optional[int] = Query(default=None)):
    os = get_shadow_os()
    habits = os.procedural.list_habits(stage=stage)
    return {"habits": [h.to_dict() for h in habits]}


@router.get("/procedural/streaks", dependencies=[Depends(verify_api_key)])
async def procedural_streaks():
    os = get_shadow_os()
    return os.procedural.get_habit_streaks()


@router.post("/procedural/routine", dependencies=[Depends(verify_api_key)])
async def procedural_add_routine(request: RoutineAddRequest):
    os = get_shadow_os()
    routine = os.procedural.add_routine(request.name, request.frequency)
    return routine.to_dict()


@router.post("/procedural/routine/complete", dependencies=[Depends(verify_api_key)])
async def procedural_complete_routine(request: RoutineIdRequest):
    os = get_shadow_os()
    try:
        routine = os.procedural.complete_routine(request.routine_id)
        return routine.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.get("/procedural/routine/{routine_id}", dependencies=[Depends(verify_api_key)])
async def procedural_get_routine(routine_id: str):
    os = get_shadow_os()
    routine = os.procedural.get_routine(routine_id)
    if not routine:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine.to_dict()


@router.get("/procedural/routines", dependencies=[Depends(verify_api_key)])
async def procedural_list_routines():
    os = get_shadow_os()
    routines = os.procedural.list_routines()
    return {"routines": [r.to_dict() for r in routines]}


@router.get("/procedural/stats", dependencies=[Depends(verify_api_key)])
async def procedural_stats():
    os = get_shadow_os()
    return os.procedural.get_stats()


# ═════════════════════════════════════════════════════════════════════
# CONFLICT RESOLUTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ConflictCreateRequest(BaseModel):
    title: str = Field(..., max_length=200)
    conflict_type: str = Field(..., max_length=50)
    intensity: int = Field(default=2, ge=1, le=4)
    description: str = Field(default="", max_length=2000)


class ConflictPerspectiveRequest(BaseModel):
    conflict_id: str = Field(..., max_length=64)
    party: str = Field(..., max_length=100)
    position: str = Field(..., max_length=300)
    interests: Optional[List[str]] = None
    concerns: Optional[List[str]] = None
    flexibility: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('interests', 'concerns', mode='before')
    @classmethod
    def _validate_list_elements(cls, v):
        if v is not None:
            return [item[:_MAX_ELEMENT_LENGTH] for item in v]
        return v


class ConflictIdRequest(BaseModel):
    conflict_id: str = Field(..., max_length=64)


class ConflictResolveRequest(BaseModel):
    conflict_id: str = Field(..., max_length=64)
    resolution: str = Field(..., max_length=500)


@router.post("/conflict/create", dependencies=[Depends(verify_api_key)])
async def conflict_create(request: ConflictCreateRequest):
    os = get_shadow_os()
    try:
        conflict = os.conflict.create_conflict(
            request.title, request.conflict_type, request.intensity, request.description
        )
        return conflict.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/conflict/perspective", dependencies=[Depends(verify_api_key)])
async def conflict_add_perspective(request: ConflictPerspectiveRequest):
    os = get_shadow_os()
    success = os.conflict.add_perspective(
        request.conflict_id, request.party, request.position,
        request.interests, request.concerns, request.flexibility
    )
    if not success:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return {"status": "added"}


@router.get("/conflict/{conflict_id}", dependencies=[Depends(verify_api_key)])
async def conflict_get(conflict_id: str):
    os = get_shadow_os()
    conflict = os.conflict.get_conflict(conflict_id)
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return conflict.to_dict()


@router.get("/conflict/{conflict_id}/strategies", dependencies=[Depends(verify_api_key)])
async def conflict_strategies(conflict_id: str):
    os = get_shadow_os()
    strategies = os.conflict.suggest_strategies(conflict_id)
    return {"strategies": [s.to_dict() for s in strategies]}


@router.get("/conflict/{conflict_id}/common-ground", dependencies=[Depends(verify_api_key)])
async def conflict_common_ground(conflict_id: str):
    os = get_shadow_os()
    analysis = os.conflict.analyze_common_ground(conflict_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Conflict not found or too few perspectives")
    return analysis


@router.post("/conflict/resolve", dependencies=[Depends(verify_api_key)])
async def conflict_resolve(request: ConflictResolveRequest):
    os = get_shadow_os()
    success = os.conflict.resolve_conflict(request.conflict_id, request.resolution)
    if not success:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return {"status": "resolved"}


@router.post("/conflict/escalate", dependencies=[Depends(verify_api_key)])
async def conflict_escalate(request: ConflictIdRequest):
    os = get_shadow_os()
    conflict = os.conflict.get_conflict(request.conflict_id)
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")
    if conflict.status.value == "resolved":
        raise HTTPException(status_code=409, detail="Cannot escalate a resolved conflict")
    success = os.conflict.escalate_conflict(request.conflict_id)
    if not success:
        raise HTTPException(status_code=500, detail="Escalation failed")
    return {"status": "escalated"}


@router.get("/conflicts", dependencies=[Depends(verify_api_key)])
async def conflict_list(status: Optional[str] = None):
    os = get_shadow_os()
    conflicts = os.conflict.list_conflicts(status=status)
    return {"conflicts": [c.to_dict() for c in conflicts]}


@router.get("/conflict/history", dependencies=[Depends(verify_api_key)])
async def conflict_history(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"history": os.conflict.get_resolution_history(limit)}


@router.get("/conflict/stats", dependencies=[Depends(verify_api_key)])
async def conflict_stats():
    os = get_shadow_os()
    return os.conflict.get_stats()

# ─── Emotional Intelligence Endpoints ───

class EmotionDetectRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    context: str = Field(default="", max_length=500)

class EmotionRegulationRequest(BaseModel):
    emotion: str = Field(..., max_length=50)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)

class PatternRecordRequest(BaseModel):
    pattern_type: str = Field(..., max_length=50)
    name: str = Field(..., max_length=100)
    emotions: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=500)


@router.post("/emotion/detect", dependencies=[Depends(verify_api_key)])
async def emotion_detect(request: EmotionDetectRequest):
    os = get_shadow_os()
    state = os.emotional_intel.detect_emotion(request.text, context=request.context)
    return state.to_dict()


@router.post("/emotion/regulation", dependencies=[Depends(verify_api_key)])
async def emotion_regulation(request: EmotionRegulationRequest):
    os = get_shadow_os()
    try:
        emotion = EmotionCategory(request.emotion)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid emotion: {request.emotion!r}")
    strategy = os.emotional_intel.suggest_regulation(emotion, request.intensity)
    return strategy.to_dict()


@router.post("/emotion/pattern", dependencies=[Depends(verify_api_key)])
async def emotion_pattern_record(request: PatternRecordRequest):
    os = get_shadow_os()
    try:
        pattern_type = PatternType(request.pattern_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid pattern type: {request.pattern_type!r}")
    emotions = []
    for e in request.emotions:
        try:
            emotions.append(EmotionCategory(e))
        except ValueError:
            pass
    pattern = os.emotional_intel.record_pattern(
        pattern_type, request.name, emotions, request.triggers, request.description
    )
    return pattern.to_dict()


@router.get("/emotion/patterns", dependencies=[Depends(verify_api_key)])
async def emotion_patterns(min_frequency: int = Query(default=2, ge=1)):
    os = get_shadow_os()
    patterns = os.emotional_intel.detect_patterns(min_frequency=min_frequency)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/emotion/trajectory", dependencies=[Depends(verify_api_key)])
async def emotion_trajectory(limit: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return os.emotional_intel.analyze_emotional_trajectory(limit=limit)


@router.get("/emotion/balance", dependencies=[Depends(verify_api_key)])
async def emotion_balance():
    os = get_shadow_os()
    return os.emotional_intel.get_emotional_balance()


@router.post("/emotion/empathy", dependencies=[Depends(verify_api_key)])
async def emotion_empathy(request: EmotionRegulationRequest):
    os = get_shadow_os()
    try:
        emotion = EmotionCategory(request.emotion)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid emotion: {request.emotion!r}")
    return os.emotional_intel.get_empathy_insight(emotion, request.intensity)


@router.get("/emotion/stats", dependencies=[Depends(verify_api_key)])
async def emotion_stats():
    os = get_shadow_os()
    return os.emotional_intel.get_stats()


# ─── Creative Synthesis Endpoints ───

class SynthesizeRequest(BaseModel):
    concepts: List[str] = Field(..., min_length=1, max_length=5)
    method: str = Field(default="combination", max_length=30)
    domain: str = Field(default="general", max_length=30)

class AssociationRequest(BaseModel):
    concept_a: str = Field(..., max_length=200)
    concept_b: str = Field(..., max_length=200)
    domain: str = Field(default="general", max_length=30)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)

class BrainstormRequest(BaseModel):
    seed_concept: str = Field(..., max_length=200)
    method: str = Field(default="cross_pollination", max_length=30)
    domain: str = Field(default="general", max_length=30)
    count: int = Field(default=5, ge=1, le=10)


@router.post("/creative/synthesize", dependencies=[Depends(verify_api_key)])
async def creative_synthesize(request: SynthesizeRequest):
    os = get_shadow_os()
    try:
        method = SynthesisMethod(request.method)
    except ValueError:
        method = SynthesisMethod.COMBINATION
    try:
        domain = CreativeDomain(request.domain)
    except ValueError:
        domain = CreativeDomain.GENERAL
    idea = os.creative.synthesize(request.concepts, method, domain)
    return idea.to_dict()


@router.post("/creative/associate", dependencies=[Depends(verify_api_key)])
async def creative_associate(request: AssociationRequest):
    os = get_shadow_os()
    try:
        domain = CreativeDomain(request.domain)
    except ValueError:
        domain = CreativeDomain.GENERAL
    pair = os.creative.add_association(
        request.concept_a, request.concept_b, domain, request.strength
    )
    return pair.to_dict()


@router.post("/creative/brainstorm", dependencies=[Depends(verify_api_key)])
async def creative_brainstorm(request: BrainstormRequest):
    os = get_shadow_os()
    try:
        method = SynthesisMethod(request.method)
    except ValueError:
        method = SynthesisMethod.CROSS_POLLINATION
    try:
        domain = CreativeDomain(request.domain)
    except ValueError:
        domain = CreativeDomain.GENERAL
    ideas = os.creative.brainstorm(request.seed_concept, method, domain, request.count)
    return {"ideas": [i.to_dict() for i in ideas]}


@router.get("/creative/ideas", dependencies=[Depends(verify_api_key)])
async def creative_ideas(
    domain: Optional[str] = None,
    min_novelty: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
):
    os = get_shadow_os()
    domain_enum = None
    if domain:
        try:
            domain_enum = CreativeDomain(domain)
        except ValueError:
            pass
    return {"ideas": os.creative.get_ideas(domain_enum, min_novelty, limit)}


@router.get("/creative/patterns", dependencies=[Depends(verify_api_key)])
async def creative_patterns(
    method: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
):
    os = get_shadow_os()
    method_enum = None
    if method:
        try:
            method_enum = SynthesisMethod(method)
        except ValueError:
            pass
    return {"patterns": os.creative.get_patterns(method_enum, limit)}


@router.get("/creative/associations/{concept}", dependencies=[Depends(verify_api_key)])
async def creative_associations(concept: str):
    os = get_shadow_os()
    return {"associations": os.creative.get_concept_associations(concept)}


@router.get("/creative/stats", dependencies=[Depends(verify_api_key)])
async def creative_stats():
    os = get_shadow_os()
    return os.creative.get_stats()


# ─── Metacognition Engine Endpoints ───

class StartThinkingRequest(BaseModel):
    reasoning_type: str = Field(default="systematic", max_length=30)
    topic: str = Field(..., max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cognitive_load: float = Field(default=0.3, ge=0.0, le=1.0)

class ResolveThinkingRequest(BaseModel):
    episode_id: str = Field(..., max_length=50)
    outcome: str = Field(..., max_length=1000)
    accuracy: float = Field(..., ge=0.0, le=1.0)

class DetectBiasRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    episode_id: Optional[str] = Field(default=None, max_length=50)


@router.post("/meta/think", dependencies=[Depends(verify_api_key)])
async def meta_start_thinking(request: StartThinkingRequest):
    os = get_shadow_os()
    try:
        rtype = ReasoningType(request.reasoning_type)
    except ValueError:
        rtype = ReasoningType.SYSTEMATIC
    episode = os.metacognition.start_thinking(rtype, request.topic, request.confidence, request.cognitive_load)
    return episode.to_dict()


@router.post("/meta/resolve", dependencies=[Depends(verify_api_key)])
async def meta_resolve_thinking(request: ResolveThinkingRequest):
    os = get_shadow_os()
    episode = os.metacognition.resolve_thinking(request.episode_id, request.outcome, request.accuracy)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode.to_dict()


@router.post("/meta/bias-detect", dependencies=[Depends(verify_api_key)])
async def meta_detect_bias(request: DetectBiasRequest):
    os = get_shadow_os()
    biases = os.metacognition.detect_biases(request.text, request.episode_id)
    return {"biases": [b.to_dict() for b in biases]}


@router.get("/meta/calibration", dependencies=[Depends(verify_api_key)])
async def meta_calibration():
    os = get_shadow_os()
    return os.metacognition.get_calibration_report()


@router.get("/meta/load", dependencies=[Depends(verify_api_key)])
async def meta_load():
    os = get_shadow_os()
    return os.metacognition.get_cognitive_load()


@router.get("/meta/biases", dependencies=[Depends(verify_api_key)])
async def meta_biases(bias_type: Optional[str] = None):
    os = get_shadow_os()
    bt = None
    if bias_type:
        try:
            bt = BiasType(bias_type)
        except ValueError:
            pass
    return {"biases": os.metacognition.get_biases(bt)}


@router.get("/meta/alerts", dependencies=[Depends(verify_api_key)])
async def meta_alerts(include_dismissed: bool = False, limit: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return {"alerts": os.metacognition.get_alerts(include_dismissed, limit)}


@router.get("/meta/stats", dependencies=[Depends(verify_api_key)])
async def meta_stats():
    os = get_shadow_os()
    return os.metacognition.get_stats()


# ─── Moral Reasoning Endpoints ───

class EvaluateActionRequest(BaseModel):
    action: str = Field(..., max_length=500)
    description: str = Field(default="", max_length=2000)
    stakeholders: Optional[List[Dict]] = Field(default=None)

class AddPrincipleRequest(BaseModel):
    name: str = Field(..., max_length=100)
    description: str = Field(default="", max_length=500)
    framework: str = Field(default="utilitarian", max_length=30)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    non_negotiable: bool = Field(default=False)

class ResolveDilemmaRequest(BaseModel):
    dilemma_id: str = Field(..., max_length=50)
    resolution: str = Field(..., max_length=1000)


@router.post("/moral/evaluate", dependencies=[Depends(verify_api_key)])
async def moral_evaluate(request: EvaluateActionRequest):
    os = get_shadow_os()
    stakeholders = []
    if request.stakeholders:
        for s in (request.stakeholders or [])[:10]:
            try:
                stakeholders.append(Stakeholder(
                    name=s.get("name", ""),
                    stakeholder_type=StakeholderType(s.get("type", "others")),
                    interests=s.get("interests", []),
                    impact=s.get("impact", 0.5),
                    vulnerability=s.get("vulnerability", 0.5),
                ))
            except (ValueError, TypeError):
                pass
    evaluation = os.moral.evaluate(request.action, request.description, stakeholders)
    return evaluation.to_dict()


@router.post("/moral/principle", dependencies=[Depends(verify_api_key)])
async def moral_add_principle(request: AddPrincipleRequest):
    os = get_shadow_os()
    try:
        framework = EthicalFramework(request.framework)
    except ValueError:
        framework = EthicalFramework.UTILITARIAN
    principle = os.moral.add_principle(request.name, request.description, framework, request.weight, request.non_negotiable)
    return principle.to_dict()


@router.post("/moral/resolve-dilemma", dependencies=[Depends(verify_api_key)])
async def moral_resolve_dilemma(request: ResolveDilemmaRequest):
    os = get_shadow_os()
    success = os.moral.resolve_dilemma(request.dilemma_id, request.resolution)
    if not success:
        raise HTTPException(status_code=404, detail="Dilemma not found")
    return {"resolved": True}


@router.get("/moral/dilemmas", dependencies=[Depends(verify_api_key)])
async def moral_dilemmas(status: Optional[str] = None):
    os = get_shadow_os()
    ds = None
    if status:
        try:
            ds = DilemmaStatus(status)
        except ValueError:
            pass
    return {"dilemmas": os.moral.get_dilemmas(ds)}


@router.get("/moral/principles", dependencies=[Depends(verify_api_key)])
async def moral_principles(framework: Optional[str] = None):
    os = get_shadow_os()
    fw = None
    if framework:
        try:
            fw = EthicalFramework(framework)
        except ValueError:
            pass
    return {"principles": os.moral.get_principles(fw)}


@router.get("/moral/evaluations", dependencies=[Depends(verify_api_key)])
async def moral_evaluations(min_score: float = Query(default=0.0, ge=0.0, le=1.0), limit: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return {"evaluations": os.moral.get_evaluations(min_score, limit)}


@router.get("/moral/stats", dependencies=[Depends(verify_api_key)])
async def moral_stats():
    os = get_shadow_os()
    return os.moral.get_stats()


# ─── Cognitive Pipeline Endpoints ───

class PipelineProcessRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    context: str = Field(default="", max_length=1000)


@router.post("/pipeline/process", dependencies=[Depends(verify_api_key)])
async def pipeline_process(request: PipelineProcessRequest):
    os = get_shadow_os()
    result = os.pipeline.process(request.text, request.context)
    return result


@router.get("/pipeline/stats", dependencies=[Depends(verify_api_key)])
async def pipeline_stats():
    os = get_shadow_os()
    return os.pipeline.get_pipeline_stats()


# ═════════════════════════════════════════════════════════════════════
# SOCIAL INTELLIGENCE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class SocialGroupRequest(BaseModel):
    name: str = Field(..., max_length=200)
    group_type: str = Field(..., max_length=50)
    description: str = Field(default="", max_length=1000)
    members: Optional[List[str]] = None

    @field_validator('members', mode='before')
    @classmethod
    def _validate_members(cls, v):
        if v is not None:
            return [m[:_MAX_ELEMENT_LENGTH] for m in v]
        return v


class SocialMemberRequest(BaseModel):
    group_id: str = Field(..., max_length=64)
    person_id: str = Field(..., max_length=64)


class InfluenceAddRequest(BaseModel):
    source: str = Field(..., max_length=64)
    target: str = Field(..., max_length=64)
    influence_type: str = Field(..., max_length=50)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = Field(default="", max_length=500)


class RoleAssignRequest(BaseModel):
    person_id: str = Field(..., max_length=64)
    group_id: str = Field(..., max_length=64)
    role_type: str = Field(..., max_length=50)
    influence_level: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/social/group", dependencies=[Depends(verify_api_key)])
async def social_create_group(request: SocialGroupRequest):
    os = get_shadow_os()
    try:
        gt = GroupType(request.group_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid group_type: {request.group_type}")
    group = os.social_intel.create_group(request.name, gt, request.description, request.members)
    return group.to_dict()


@router.post("/social/member", dependencies=[Depends(verify_api_key)])
async def social_add_member(request: SocialMemberRequest):
    os = get_shadow_os()
    success = os.social_intel.add_member(request.group_id, request.person_id)
    if not success:
        raise HTTPException(status_code=404, detail="Group not found or member limit reached")
    return {"status": "added"}


@router.post("/social/influence", dependencies=[Depends(verify_api_key)])
async def social_add_influence(request: InfluenceAddRequest):
    os = get_shadow_os()
    try:
        itype = InfluenceType(request.influence_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid influence_type: {request.influence_type}")
    rel = os.social_intel.add_influence(request.source, request.target, itype, request.strength, request.context)
    return rel.to_dict()


@router.post("/social/role", dependencies=[Depends(verify_api_key)])
async def social_assign_role(request: RoleAssignRequest):
    os = get_shadow_os()
    try:
        rtype = RoleType(request.role_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role_type: {request.role_type}")
    role = os.social_intel.assign_role(request.person_id, request.group_id, rtype, request.influence_level)
    return role.to_dict()


@router.get("/social/group/{group_id}", dependencies=[Depends(verify_api_key)])
async def social_get_group(group_id: str):
    os = get_shadow_os()
    group = os.social_intel.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group.to_dict()


@router.get("/social/groups", dependencies=[Depends(verify_api_key)])
async def social_list_groups(group_type: Optional[str] = None):
    os = get_shadow_os()
    gt = None
    if group_type:
        try:
            gt = GroupType(group_type)
        except ValueError:
            pass
    groups = os.social_intel.list_groups(gt)
    return {"groups": [g.to_dict() for g in groups]}


@router.get("/social/dynamics/{group_id}", dependencies=[Depends(verify_api_key)])
async def social_dynamics(group_id: str):
    os = get_shadow_os()
    dynamics = os.social_intel.get_group_dynamics(group_id)
    if not dynamics:
        raise HTTPException(status_code=404, detail="Group not found")
    return dynamics


@router.get("/social/influencers", dependencies=[Depends(verify_api_key)])
async def social_influencers(group_id: Optional[str] = None, min_influence: float = Query(default=0.7, ge=0.0, le=1.0)):
    os = get_shadow_os()
    gid = group_id if group_id else None
    return {"influencers": os.social_intel.find_influencers(gid, min_influence)}


@router.get("/social/network/{person_id}", dependencies=[Depends(verify_api_key)])
async def social_network(person_id: str):
    os = get_shadow_os()
    return os.social_intel.get_influence_network(person_id)


@router.get("/social/stats", dependencies=[Depends(verify_api_key)])
async def social_stats():
    os = get_shadow_os()
    return os.social_intel.get_stats()


# ═════════════════════════════════════════════════════════════════════
# DECISION JOURNAL ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class DecisionRecordRequest(BaseModel):
    title: str = Field(..., max_length=200)
    chosen: str = Field(..., max_length=200)
    alternatives: Optional[List[str]] = None
    reasoning: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    biases_flagged: Optional[List[str]] = None
    domain: str = Field(default="", max_length=100)
    decision_type: str = Field(default="routine", max_length=30)
    description: str = Field(default="", max_length=2000)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('alternatives', mode='before')
    @classmethod
    def _validate_alternatives(cls, v):
        if v is not None:
            return [a[:200] for a in v[:10]]
        return v

    @field_validator('biases_flagged', mode='before')
    @classmethod
    def _validate_biases(cls, v):
        if v is not None:
            return [b[:50] for b in v[:10]]
        return v


class DecisionOutcomeRequest(BaseModel):
    decision_id: str = Field(..., max_length=64)
    outcome: str = Field(..., max_length=2000)
    accuracy: float = Field(..., ge=0.0, le=1.0)


class DecisionStatusRequest(BaseModel):
    decision_id: str = Field(..., max_length=64)
    status: str = Field(..., max_length=20)


class RetrospectiveRequest(BaseModel):
    period_days: int = Field(default=30, ge=1, le=3650)


@router.post("/journal/decision", dependencies=[Depends(verify_api_key)])
async def journal_record_decision(request: DecisionRecordRequest):
    os = get_shadow_os()
    record = os.journal.record_decision(
        title=request.title,
        chosen=request.chosen,
        alternatives=request.alternatives,
        reasoning=request.reasoning,
        confidence=request.confidence,
        biases_flagged=request.biases_flagged,
        domain=request.domain,
        decision_type=request.decision_type,
        description=request.description,
        metadata=request.metadata,
    )
    return record.to_dict()


@router.post("/journal/outcome", dependencies=[Depends(verify_api_key)])
async def journal_record_outcome(request: DecisionOutcomeRequest):
    os = get_shadow_os()
    success = os.journal.record_outcome(request.decision_id, request.outcome, request.accuracy)
    if not success:
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"status": "recorded", "decision_id": request.decision_id}


@router.post("/journal/status", dependencies=[Depends(verify_api_key)])
async def journal_update_status(request: DecisionStatusRequest):
    os = get_shadow_os()
    success = os.journal.update_status(request.decision_id, request.status)
    if not success:
        raise HTTPException(status_code=404, detail="Decision not found or invalid status")
    return {"status": "updated", "decision_id": request.decision_id}


@router.post("/journal/retrospective", dependencies=[Depends(verify_api_key)])
async def journal_run_retrospective(request: RetrospectiveRequest):
    os = get_shadow_os()
    retro = os.journal.run_retrospective(period_days=request.period_days)
    return retro.to_dict()


@router.get("/journal/decision/{decision_id}", dependencies=[Depends(verify_api_key)])
async def journal_get_decision(decision_id: str):
    os = get_shadow_os()
    record = os.journal.get_decision(decision_id)
    if not record:
        raise HTTPException(status_code=404, detail="Decision not found")
    return record.to_dict()


@router.get("/journal/decisions", dependencies=[Depends(verify_api_key)])
async def journal_list_decisions(
    domain: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    decisions = os.journal.list_decisions(domain=domain, status=status, limit=limit)
    return {"decisions": [d.to_dict() for d in decisions]}


@router.get("/journal/accuracy", dependencies=[Depends(verify_api_key)])
async def journal_accuracy(domain: Optional[str] = None):
    os = get_shadow_os()
    return os.journal.get_decision_accuracy(domain=domain)


@router.get("/journal/overconfidence", dependencies=[Depends(verify_api_key)])
async def journal_overconfidence():
    os = get_shadow_os()
    return os.journal.get_overconfidence_report()


@router.get("/journal/patterns", dependencies=[Depends(verify_api_key)])
async def journal_patterns(min_frequency: int = Query(default=3, ge=1)):
    os = get_shadow_os()
    patterns = os.journal.detect_patterns(min_frequency=min_frequency)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/journal/stats", dependencies=[Depends(verify_api_key)])
async def journal_stats():
    os = get_shadow_os()
    return os.journal.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE REFLECTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ReflectionAddRequest(BaseModel):
    topic: str = Field(..., max_length=200)
    description: str = Field(default="", max_length=2000)
    reflection_type: str = Field(default="daily", max_length=30)
    what_went_well: Optional[List[str]] = None
    what_could_improve: Optional[List[str]] = None
    lessons_learned: Optional[List[str]] = None
    confidence_before: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_after: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: str = Field(default="", max_length=100)
    linked_decision_ids: Optional[List[str]] = None
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('what_went_well', 'what_could_improve', 'lessons_learned', mode='before')
    @classmethod
    def _validate_lists(cls, v):
        if v is not None:
            return [item[:200] for item in v[:10]]
        return v

    @field_validator('linked_decision_ids', mode='before')
    @classmethod
    def _validate_decisions(cls, v):
        if v is not None:
            return [d[:64] for d in v[:5]]
        return v


class ReflectionInsightRequest(BaseModel):
    reflection_id: str = Field(..., max_length=64)


@router.post("/reflection/add", dependencies=[Depends(verify_api_key)])
async def reflection_add(request: ReflectionAddRequest):
    os = get_shadow_os()
    entry = os.reflection.add_reflection(
        topic=request.topic,
        description=request.description,
        reflection_type=request.reflection_type,
        what_went_well=request.what_went_well,
        what_could_improve=request.what_could_improve,
        lessons_learned=request.lessons_learned,
        confidence_before=request.confidence_before,
        confidence_after=request.confidence_after,
        domain=request.domain,
        linked_decision_ids=request.linked_decision_ids,
        sentiment=request.sentiment,
        metadata=request.metadata,
    )
    return entry.to_dict()


@router.post("/reflection/insights/{reflection_id}", dependencies=[Depends(verify_api_key)])
async def reflection_extract_insights(reflection_id: str):
    os = get_shadow_os()
    insights = os.reflection.extract_insights(reflection_id)
    if insights is None:
        raise HTTPException(status_code=404, detail="Reflection not found")
    return {"insights": [i.to_dict() for i in insights]}


@router.post("/reflection/patterns", dependencies=[Depends(verify_api_key)])
async def reflection_detect_patterns(min_frequency: int = Query(default=2, ge=1)):
    os = get_shadow_os()
    patterns = os.reflection.detect_patterns(min_frequency=min_frequency)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/reflection/{entry_id}", dependencies=[Depends(verify_api_key)])
async def reflection_get(entry_id: str):
    os = get_shadow_os()
    entry = os.reflection.get_reflection(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Reflection not found")
    return entry.to_dict()


@router.get("/reflection/list", dependencies=[Depends(verify_api_key)])
async def reflection_list(
    reflection_type: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    entries = os.reflection.list_reflections(
        reflection_type=reflection_type, domain=domain, limit=limit
    )
    return {"reflections": [e.to_dict() for e in entries]}


@router.get("/reflection/confidence-trend", dependencies=[Depends(verify_api_key)])
async def reflection_confidence_trend(
    domain: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
):
    os = get_shadow_os()
    return os.reflection.get_confidence_trend(domain=domain, limit=limit)


@router.get("/reflection/insights", dependencies=[Depends(verify_api_key)])
async def reflection_insights(
    category: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=200),
):
    os = get_shadow_os()
    insights = os.reflection.get_learning_insights(category=category, limit=limit)
    return {"insights": [i.to_dict() for i in insights]}


@router.get("/reflection/stats", dependencies=[Depends(verify_api_key)])
async def reflection_stats():
    os = get_shadow_os()
    return os.reflection.get_stats()


# ─── Goal Reasoning Endpoints ─────────────────────────────────────

class GoalAssessRequest(BaseModel):
    goal_id: str = Field(..., max_length=64)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    deadline: Optional[str] = Field(default=None, max_length=50)
    effort_spent: float = Field(default=0.0, ge=0.0)
    effort_total: float = Field(default=1.0, ge=0.01)
    blockers: Optional[List[str]] = Field(default=None)
    context: Optional[Dict[str, Any]] = None

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, v):
        if v is not None:
            return [b[:200] for b in v[:10]]
        return v


class ObstacleDetectRequest(BaseModel):
    goal_id: str = Field(..., max_length=64)
    description: str = Field(..., max_length=500)
    severity: str = Field(default="medium", max_length=20)
    mitigation: str = Field(default="", max_length=500)
    metadata: Optional[Dict[str, Any]] = None


class ConflictAnalyzeRequest(BaseModel):
    goals: List[Dict[str, Any]] = Field(...)

    @field_validator("goals")
    @classmethod
    def validate_goals(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 goals per conflict analysis")
        return v


class ReplanSuggestRequest(BaseModel):
    goal_id: str = Field(..., max_length=64)
    reason: str = Field(..., max_length=500)
    current_state: Optional[Dict[str, Any]] = None


@router.post("/goal-reasoning/assess", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_assess(request: GoalAssessRequest):
    os = get_shadow_os()
    assessment = os.goal_reasoning.assess_goal(
        goal_id=request.goal_id,
        progress=request.progress,
        deadline=request.deadline,
        effort_spent=request.effort_spent,
        effort_total=request.effort_total,
        blockers=request.blockers,
        context=request.context,
    )
    return assessment.to_dict()


@router.post("/goal-reasoning/obstacle", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_obstacle(request: ObstacleDetectRequest):
    os = get_shadow_os()
    try:
        severity = ObstacleSeverity(request.severity)
    except ValueError:
        severity = ObstacleSeverity.MEDIUM
    obstacle = os.goal_reasoning.detect_obstacle(
        goal_id=request.goal_id,
        description=request.description,
        severity=severity,
        mitigation=request.mitigation,
        metadata=request.metadata,
    )
    return obstacle.to_dict()


@router.post("/goal-reasoning/obstacle/{obstacle_id}/resolve", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_resolve_obstacle(obstacle_id: str):
    os = get_shadow_os()
    if not os.goal_reasoning.resolve_obstacle(obstacle_id):
        raise HTTPException(status_code=404, detail="Obstacle not found")
    return {"status": "resolved"}


@router.post("/goal-reasoning/conflicts", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_conflicts(request: ConflictAnalyzeRequest):
    os = get_shadow_os()
    conflicts = os.goal_reasoning.analyze_conflicts(request.goals)
    return {"conflicts": [c.to_dict() for c in conflicts]}


@router.post("/goal-reasoning/replan", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_replan(request: ReplanSuggestRequest):
    os = get_shadow_os()
    action = os.goal_reasoning.suggest_replan(
        goal_id=request.goal_id,
        reason=request.reason,
        current_state=request.current_state,
    )
    return action.to_dict()


@router.post("/goal-reasoning/replan/{action_id}/apply", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_apply_replan(action_id: str):
    os = get_shadow_os()
    if not os.goal_reasoning.apply_replan(action_id):
        raise HTTPException(status_code=404, detail="Replan action not found or already applied")
    return {"status": "applied"}


@router.post("/goal-reasoning/prioritize", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_prioritize(request: ConflictAnalyzeRequest):
    os = get_shadow_os()
    result = os.goal_reasoning.prioritize_goals(request.goals)
    return {"prioritized_goals": result}


@router.get("/goal-reasoning/assessments", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_assessments(
    goal_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    return {"assessments": os.goal_reasoning.list_assessments(goal_id=goal_id, limit=limit)}


@router.get("/goal-reasoning/obstacles", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_obstacles(
    goal_id: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    return {"obstacles": os.goal_reasoning.list_obstacles(goal_id=goal_id, resolved=resolved, limit=limit)}


@router.get("/goal-reasoning/conflicts", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_conflicts_list(
    goal_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    return {"conflicts": os.goal_reasoning.list_conflicts(goal_id=goal_id, limit=limit)}


@router.get("/goal-reasoning/replan-actions", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_replan_actions(
    goal_id: Optional[str] = None,
    applied: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    return {"replan_actions": os.goal_reasoning.list_replan_actions(goal_id=goal_id, applied=applied, limit=limit)}


@router.get("/goal-reasoning/stats", dependencies=[Depends(verify_api_key)])
async def goal_reasoning_stats():
    os = get_shadow_os()
    return os.goal_reasoning.get_stats()


# ═════════════════════════════════════════════════════════════════
# Device Context - Multi-Device Passive Data Fusion
# ═════════════════════════════════════════════════════════════════

class DeviceIngestRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    data: Dict[str, Any] = Field(...)

class IPhoneIngestRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    data: Dict[str, Any] = Field(...)

class WatchIngestRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    data: Dict[str, Any] = Field(...)

class LaptopIngestRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    data: Dict[str, Any] = Field(...)

class RegisterDeviceRequest(BaseModel):
    device_id: str = Field(..., max_length=100)
    device_type: str = Field(..., max_length=50)
    name: str = Field(default="", max_length=100)
    capabilities: Optional[List[str]] = Field(default=None)

class RegisterWifiRequest(BaseModel):
    ssid: str = Field(..., max_length=100)
    location_type: str = Field(..., max_length=50)

class RegisterPersonRequest(BaseModel):
    bt_mac: str = Field(..., max_length=50)
    name: str = Field(..., max_length=100)


@router.post("/device/register", dependencies=[Depends(verify_api_key)])
async def device_register(req: RegisterDeviceRequest):
    os = get_shadow_os()
    try:
        device_type = DeviceType(req.device_type)
    except ValueError:
        device_type = DeviceType.UNKNOWN
    os.device_context.register_device(req.device_id, device_type, req.name, req.capabilities)
    return {"status": "registered", "device_id": req.device_id, "device_type": device_type.value}


@router.post("/device/iphone", dependencies=[Depends(verify_api_key)])
async def device_iphone_ingest(req: IPhoneIngestRequest):
    os = get_shadow_os()
    ctx = os.device_context.ingest_iphone(req.device_id, req.data)
    return ctx.to_dict() if ctx else {"status": "no_context"}


@router.post("/device/watch", dependencies=[Depends(verify_api_key)])
async def device_watch_ingest(req: WatchIngestRequest):
    os = get_shadow_os()
    ctx = os.device_context.ingest_watch(req.device_id, req.data)
    return ctx.to_dict() if ctx else {"status": "no_context"}


@router.post("/device/laptop", dependencies=[Depends(verify_api_key)])
async def device_laptop_ingest(req: LaptopIngestRequest):
    os = get_shadow_os()
    ctx = os.device_context.ingest_laptop(req.device_id, req.data)
    return ctx.to_dict() if ctx else {"status": "no_context"}


@router.get("/device/context", dependencies=[Depends(verify_api_key)])
async def device_context():
    os = get_shadow_os()
    ctx = os.device_context.get_context()
    return ctx.to_dict() if ctx else {"status": "no_data_yet"}


@router.get("/device/health-trends", dependencies=[Depends(verify_api_key)])
async def device_health_trends(minutes: int = Query(default=60, ge=1, le=1440)):
    os = get_shadow_os()
    return os.device_context.get_health_trends(minutes=minutes)


@router.get("/device/digital-wellbeing", dependencies=[Depends(verify_api_key)])
async def device_digital_wellbeing(hours: int = Query(default=24, ge=1, le=168)):
    os = get_shadow_os()
    return os.device_context.get_digital_wellbeing(hours=hours)


@router.get("/device/devices", dependencies=[Depends(verify_api_key)])
async def device_summary():
    os = get_shadow_os()
    return os.device_context.get_device_summary()


@router.get("/device/stats", dependencies=[Depends(verify_api_key)])
async def device_stats():
    os = get_shadow_os()
    return os.device_context.get_stats()


@router.post("/device/wifi-location", dependencies=[Depends(verify_api_key)])
async def device_register_wifi(req: RegisterWifiRequest):
    os = get_shadow_os()
    try:
        location_type = LocationType(req.location_type)
    except ValueError:
        location_type = LocationType.UNKNOWN
    os.device_context.register_wifi_location(req.ssid, location_type)
    return {"status": "registered", "ssid": req.ssid, "location_type": location_type.value}


@router.post("/device/person", dependencies=[Depends(verify_api_key)])
async def device_register_person(req: RegisterPersonRequest):
    os = get_shadow_os()
    os.device_context.register_person(req.bt_mac, req.name)
    return {"status": "registered", "name": req.name}


class SiriPushRequest(BaseModel):
    device_type: str = Field(default="iphone", max_length=20)
    device_id: Optional[str] = Field(default=None, max_length=100)
    data: Dict[str, Any] = Field(...)

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v):
        valid = {"iphone", "watch", "laptop"}
        if v and v.lower() not in valid:
            raise ValueError(f"device_type must be one of {valid}")
        return v.lower() if v else "iphone"


@router.post("/device/push", dependencies=[Depends(verify_api_key)])
async def device_siri_push(req: SiriPushRequest):
    """Simplified device data push endpoint for Siri Shortcuts.

    Accepts device_type (iphone/watch/laptop) and data dict.
    Auto-generates device_id if not provided.
    Routes to the appropriate ingest method.
    """
    os = get_shadow_os()
    device_id = req.device_id or f"siri_{req.device_type}"
    try:
        dt = DeviceType(req.device_type)
    except ValueError:
        dt = DeviceType.UNKNOWN
    os.device_context.register_device(device_id, dt, name=f"Siri {req.device_type.title()}")
    if req.device_type == "iphone":
        ctx = os.device_context.ingest_iphone(device_id, req.data)
    elif req.device_type == "watch":
        ctx = os.device_context.ingest_watch(device_id, req.data)
    else:
        ctx = os.device_context.ingest_laptop(device_id, req.data)
    return ctx.to_dict() if ctx else {"status": "pushed", "device_id": device_id}


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowArise — Universal Agent Creation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AriseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    objective: str = Field(..., min_length=1, max_length=2000)
    template_id: Optional[str] = Field(default=None, max_length=50)
    capabilities: Optional[List[str]] = Field(default=None, max_length=12)
    priority: Optional[str] = Field(default="medium", max_length=20)
    constraints: Optional[List[str]] = Field(default=None, max_length=10)
    success_criteria: Optional[List[str]] = Field(default=None, max_length=10)
    context_data: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        valid = {"critical", "high", "medium", "low", "background"}
        if v and v.lower() not in valid:
            raise ValueError(f"priority must be one of {valid}")
        return v.lower() if v else "medium"

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v):
        if v:
            valid = {e.value for e in CapabilityType}
            for cap in v:
                if cap not in valid:
                    raise ValueError(f"capability must be one of {valid}")
        return v


class AriseExecuteRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=12)
    description: str = Field(..., min_length=1, max_length=1000)
    capability: str = Field(..., min_length=1, max_length=30)
    input_data: Optional[Dict[str, Any]] = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_ms: int = Field(default=0, ge=0)


class AriseEvolveRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=12)
    feedback: str = Field(..., min_length=1, max_length=2000)
    adjustments: Optional[Dict[str, float]] = Field(default=None)
    new_capabilities: Optional[List[str]] = Field(default=None, max_length=12)

    @field_validator("new_capabilities")
    @classmethod
    def validate_capabilities(cls, v):
        if v:
            valid = {e.value for e in CapabilityType}
            for cap in v:
                if cap not in valid:
                    raise ValueError(f"capability must be one of {valid}")
        return v


class AriseCompleteRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=12)
    status: str = Field(default="completed", max_length=20)
    result: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        valid = {"completed", "failed"}
        if v.lower() not in valid:
            raise ValueError(f"status must be one of {valid}")
        return v.lower()


@router.post("/arise", dependencies=[Depends(verify_api_key)])
async def arise_create_agent(req: AriseRequest):
    os = get_shadow_os()
    return os.arise_agent(
        name=req.name, objective=req.objective,
        template_id=req.template_id, capabilities=req.capabilities,
        priority=req.priority, constraints=req.constraints,
        success_criteria=req.success_criteria, context_data=req.context_data,
    )


@router.post("/arise/execute", dependencies=[Depends(verify_api_key)])
async def arise_execute_step(req: AriseExecuteRequest):
    os = get_shadow_os()
    return os.execute_agent_step(
        agent_id=req.agent_id, description=req.description,
        capability=req.capability, input_data=req.input_data,
        confidence=req.confidence, duration_ms=req.duration_ms,
    )


@router.post("/arise/evolve", dependencies=[Depends(verify_api_key)])
async def arise_evolve_agent(req: AriseEvolveRequest):
    os = get_shadow_os()
    return os.evolve_agent(
        agent_id=req.agent_id, feedback=req.feedback,
        adjustments=req.adjustments, new_capabilities=req.new_capabilities,
    )


@router.post("/arise/complete", dependencies=[Depends(verify_api_key)])
async def arise_complete_agent(req: AriseCompleteRequest):
    os = get_shadow_os()
    try:
        status_val = AgentStatus(req.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {req.status}")
    agent = os.arise.complete(req.agent_id, result=req.result, status=status_val)
    return agent.to_dict() if agent else {"status": "error", "reason": "agent not found"}


@router.get("/arise", dependencies=[Depends(verify_api_key)])
async def arise_list_agents(
    status: Optional[str] = Query(default=None, max_length=20),
    lifecycle: Optional[str] = Query(default=None, max_length=20),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    try:
        st = AgentStatus(status) if status else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    try:
        lc = AgentLifecycle(lifecycle) if lifecycle else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid lifecycle: {lifecycle}")
    agents = os.arise.list_agents(status=st, lifecycle=lc, limit=limit)
    return [a.to_dict() for a in agents]


@router.get("/arise/templates", dependencies=[Depends(verify_api_key)])
async def arise_list_templates():
    os = get_shadow_os()
    templates = os.arise.list_templates()
    return [t.to_dict() for t in templates]


@router.get("/arise/stats", dependencies=[Depends(verify_api_key)])
async def arise_stats():
    os = get_shadow_os()
    return os.get_arise_stats()


@router.get("/arise/agent/{agent_id}", dependencies=[Depends(verify_api_key)])
async def arise_get_agent(agent_id: str):
    os = get_shadow_os()
    agent = os.arise.get_agent(agent_id)
    return agent.to_dict() if agent else {"status": "error", "reason": "agent not found"}


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowSummon — Voice Summon Router
# ═══════════════════════════════════════════════════════════════════════════════

class SummonVoiceRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = Field(default=None)


class SummonAbilityRequest(BaseModel):
    ability: str = Field(..., min_length=1, max_length=50)
    request: str = Field(..., min_length=1, max_length=5000)
    context: Optional[Dict[str, Any]] = Field(default=None)


@router.post("/summon/voice", dependencies=[Depends(verify_api_key)])
async def summon_voice(req: SummonVoiceRequest):
    os = get_shadow_os()
    return os.summon_voice(req.transcript, req.context)


@router.post("/summon/text", dependencies=[Depends(verify_api_key)])
async def summon_text(req: SummonVoiceRequest):
    os = get_shadow_os()
    return os.summon_voice(req.transcript, req.context)


@router.get("/summon/abilities", dependencies=[Depends(verify_api_key)])
async def summon_abilities():
    os = get_shadow_os()
    return os.get_summon_abilities()


@router.get("/summon/agent-templates", dependencies=[Depends(verify_api_key)])
async def summon_agent_templates():
    os = get_shadow_os()
    return os.get_summon_templates()


@router.get("/summon/history", dependencies=[Depends(verify_api_key)])
async def summon_history(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return os.get_summon_history(limit)


@router.get("/summon/stats", dependencies=[Depends(verify_api_key)])
async def summon_stats():
    os = get_shadow_os()
    return os.summon.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousDaemon — Always-On Processing + SecurityGuardian
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/daemon/status", dependencies=[Depends(verify_api_key)])
async def daemon_status():
    os = get_shadow_os()
    return os.get_daemon_status()


@router.get("/daemon/observations", dependencies=[Depends(verify_api_key)])
async def daemon_observations(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"observations": os.get_daemon_observations(limit)}


@router.get("/daemon/decisions", dependencies=[Depends(verify_api_key)])
async def daemon_decisions(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"decisions": os.get_daemon_decisions(limit)}


@router.get("/daemon/security-events", dependencies=[Depends(verify_api_key)])
async def daemon_security_events(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"events": os.get_security_events(limit)}


class DaemonWatchRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)

    @field_validator('path')
    @classmethod
    def validate_path_no_traversal(cls, v):
        import os as _os
        # Normalise both separator styles: requests may reference a path on a
        # different OS than the host processing them.
        normalized = _os.path.normpath(v).replace('\\', '/')
        if '..' in normalized.split('/'):
            raise ValueError('path contains directory traversal')
        restricted = ('/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/run/',
                      '/var/', '/boot/', '/bin/', '/sbin/', '/usr/')
        lowered = normalized.lower()
        for prefix in restricted:
            if lowered.startswith(prefix):
                raise ValueError(f'path in restricted directory: {prefix}')
        if lowered.startswith('/'):
            home = _os.path.expanduser('~').replace('\\', '/').lower()
            if not lowered.startswith(home):
                raise ValueError('absolute paths must be under home directory')
        return v


@router.post("/daemon/watch", dependencies=[Depends(verify_api_key)])
async def daemon_add_watch(req: DaemonWatchRequest):
    os = get_shadow_os()
    result = os.daemon_add_watch(req.path)
    if not result:
        raise HTTPException(status_code=400, detail="Path not found or not accessible")
    return {"watched": True, "path": req.path}


@router.delete("/daemon/watch", dependencies=[Depends(verify_api_key)])
async def daemon_remove_watch(req: DaemonWatchRequest):
    os = get_shadow_os()
    os.daemon_remove_watch(req.path)
    return {"removed": True, "path": req.path}


class DaemonObserveRequest(BaseModel):
    observation_type: str = Field(default="system_metrics", max_length=50, pattern=r'^[a-z_]+$')


@router.post("/daemon/observe", dependencies=[Depends(verify_api_key)])
async def daemon_force_observe(req: DaemonObserveRequest):
    os = get_shadow_os()
    return os.daemon_force_observe(req.observation_type)


@router.post("/daemon/kill-switch", dependencies=[Depends(verify_api_key)])
async def daemon_kill_switch():
    os = get_shadow_os()
    return os.daemon_kill_switch()


@router.post("/daemon/release-kill-switch", dependencies=[Depends(verify_api_key)])
async def daemon_release_kill_switch():
    os = get_shadow_os()
    return os.daemon_release_kill_switch()


class DaemonVitalsRequest(BaseModel):
    activity_state: str = Field(default="unknown", max_length=20, pattern=r'^[a-z_]+$')
    heart_rate: Optional[float] = Field(default=None, ge=0.0, le=300.0)
    hrv: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    step_count: Optional[int] = Field(default=None, ge=0)
    sleep_hours: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    stress_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    screen_on: Optional[bool] = None
    location_type: str = Field(default="", max_length=50, pattern=r'^[a-zA-Z0-9_\-]*$')


@router.post("/daemon/vitals", dependencies=[Depends(verify_api_key)])
async def daemon_ingest_vitals(req: DaemonVitalsRequest):
    from .autonomous_daemon import RhythmSample, UserActivityState
    os = get_shadow_os()
    try:
        state = UserActivityState(req.activity_state)
    except ValueError:
        state = UserActivityState.UNKNOWN
    sample = RhythmSample(
        activity_state=state,
        heart_rate=req.heart_rate,
        hrv=req.hrv,
        step_count=req.step_count,
        sleep_hours=req.sleep_hours,
        stress_level=req.stress_level,
        screen_on=req.screen_on,
        location_type=req.location_type,
    )
    os.ingest_user_vitals(sample)
    return {"status": "ingested", "activity_state": state.value}


@router.get("/daemon/vitals/baselines", dependencies=[Depends(verify_api_key)])
async def daemon_vitals_baselines():
    os = get_shadow_os()
    return os.get_user_rhythm_baselines()


@router.get("/daemon/vitals/state", dependencies=[Depends(verify_api_key)])
async def daemon_vitals_state():
    os = get_shadow_os()
    return {"user_state": os.get_user_activity_state().value}


class EvolutionObserveRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=65536)
    domain: str = Field(default="work", max_length=50)
    observation_type: str = Field(default="behavior", max_length=50)
    evidence: str = Field(default="", max_length=65536)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvolutionAdoptRequest(BaseModel):
    suggestion_id: str = Field(..., min_length=1, max_length=12)
    feedback: str = Field(default="", max_length=65536)


class EvolutionReviewRequest(BaseModel):
    suggestion_id: str = Field(..., min_length=1, max_length=12)
    outcome: str = Field(default="unknown", max_length=20)
    observed_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    feedback: str = Field(default="", max_length=65536)


@router.post("/evolution/observe", dependencies=[Depends(verify_api_key)])
async def evolution_observe(req: EvolutionObserveRequest):
    os = get_shadow_os()
    obs = os.evolution.add_observation(
        description=req.description,
        domain=req.domain,
        observation_type=req.observation_type,
        evidence=req.evidence,
        confidence=req.confidence,
    )
    return obs.to_dict()


@router.get("/evolution/friction", dependencies=[Depends(verify_api_key)])
async def evolution_friction(domain: str = Query(default=None, max_length=50)):
    os = get_shadow_os()
    fps = os.evolution.get_friction_points(domain=domain)
    return [fp.to_dict() for fp in fps]


@router.get("/evolution/suggestions", dependencies=[Depends(verify_api_key)])
async def evolution_suggestions(
    domain: str = Query(default=None, max_length=50),
    status: str = Query(default=None, max_length=20),
):
    os = get_shadow_os()
    sugs = os.evolution.get_suggestions(domain=domain, status=status)
    return [s.to_dict() for s in sugs]


@router.post("/evolution/generate", dependencies=[Depends(verify_api_key)])
async def evolution_generate(
    domain: str = Query(default=None, max_length=50),
    limit: int = Query(default=10, ge=1, le=50),
):
    os = get_shadow_os()
    sugs = os.evolution.generate_suggestions(domain=domain, limit=limit)
    return [s.to_dict() for s in sugs]


@router.post("/evolution/adopt", dependencies=[Depends(verify_api_key)])
async def evolution_adopt(req: EvolutionAdoptRequest):
    os = get_shadow_os()
    record = os.evolution.adopt_suggestion(req.suggestion_id, req.feedback)
    return record.to_dict()


@router.post("/evolution/reject", dependencies=[Depends(verify_api_key)])
async def evolution_reject(req: EvolutionAdoptRequest):
    os = get_shadow_os()
    result = os.evolution.reject_suggestion(req.suggestion_id, req.feedback)
    if not result:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return {"status": "rejected"}


@router.post("/evolution/review", dependencies=[Depends(verify_api_key)])
async def evolution_review(req: EvolutionReviewRequest):
    os = get_shadow_os()
    record = os.evolution.review_adoption(
        req.suggestion_id, req.outcome, req.observed_impact, req.feedback
    )
    if record is None:
        raise HTTPException(status_code=404, detail="No pending adoption found")
    return record.to_dict()


@router.get("/evolution/progress", dependencies=[Depends(verify_api_key)])
async def evolution_progress(domain: str = Query(default=None, max_length=50)):
    os = get_shadow_os()
    return os.evolution.get_domain_progress(domain=domain)


@router.get("/evolution/cross-domain", dependencies=[Depends(verify_api_key)])
async def evolution_cross_domain():
    os = get_shadow_os()
    return os.evolution.find_cross_domain_insights()


@router.get("/evolution/summary", dependencies=[Depends(verify_api_key)])
async def evolution_summary():
    os = get_shadow_os()
    return os.evolution.get_evolution_summary()


@router.get("/evolution/stats", dependencies=[Depends(verify_api_key)])
async def evolution_stats():
    os = get_shadow_os()
    return os.evolution.get_stats()


class OracleStateRequest(BaseModel):
    domain: str = Field(..., min_length=1, max_length=50)
    state_values: Dict = Field(default_factory=dict)


class OraclePredictRequest(BaseModel):
    target_date: str = Field(default="", max_length=20)
    confidence: str = Field(default="high", max_length=20)
    domains: List[str] = Field(default_factory=list)


@router.post("/oracle/state", dependencies=[Depends(verify_api_key)])
async def oracle_set_state(req: OracleStateRequest):
    os = get_shadow_os()
    snapshot = os.oracle.set_domain_state(req.domain, req.state_values)
    return snapshot.to_dict()


@router.post("/oracle/predict", dependencies=[Depends(verify_api_key)])
async def oracle_predict(req: OraclePredictRequest):
    os = get_shadow_os()
    domains = req.domains if req.domains else None
    fs = os.oracle.predict(
        target_date=req.target_date or None,
        confidence=req.confidence,
        domains=domains,
    )
    return fs.to_dict()


@router.get("/oracle/diagnose", dependencies=[Depends(verify_api_key)])
async def oracle_diagnose():
    os = get_shadow_os()
    return os.oracle.diagnose()


@router.get("/oracle/predictions", dependencies=[Depends(verify_api_key)])
async def oracle_predictions(
    domain: str = Query(default=None, max_length=50),
    limit: int = Query(default=20, ge=1, le=100),
):
    os = get_shadow_os()
    preds = os.oracle.get_predictions(domain=domain, limit=limit)
    return [p.to_dict() for p in preds]


@router.get("/oracle/forecasts", dependencies=[Depends(verify_api_key)])
async def oracle_forecasts(limit: int = Query(default=10, ge=1, le=50)):
    os = get_shadow_os()
    fcs = os.oracle.get_forecasts(limit=limit)
    return [f.to_dict() for f in fcs]


@router.get("/oracle/domains", dependencies=[Depends(verify_api_key)])
async def oracle_domains():
    os = get_shadow_os()
    return os.oracle.get_available_domains()


@router.get("/oracle/domain-states", dependencies=[Depends(verify_api_key)])
async def oracle_domain_states():
    os = get_shadow_os()
    return os.oracle.get_domain_states()


@router.get("/oracle/diagnoses", dependencies=[Depends(verify_api_key)])
async def oracle_diagnoses(limit: int = Query(default=10, ge=1, le=50)):
    os = get_shadow_os()
    return os.oracle.get_diagnoses(limit=limit)


@router.get("/oracle/stats", dependencies=[Depends(verify_api_key)])
async def oracle_stats():
    os = get_shadow_os()
    return os.oracle.get_stats()


@router.get("/oracle/daily", dependencies=[Depends(verify_api_key)])
async def oracle_daily():
    os = get_shadow_os()
    return os.oracle.get_daily_oracle()


@router.post("/oracle/compound-hazard", dependencies=[Depends(verify_api_key)])
async def oracle_compound_hazard(req: Optional[OraclePredictRequest] = None):
    os = get_shadow_os()
    domains = req.domains if req else None
    return os.oracle.assess_compound_hazard(domains=domains)


@router.get("/oracle/regime-shift", dependencies=[Depends(verify_api_key)])
async def oracle_regime_shift(domain: str = Query(default="economics", max_length=100)):
    os = get_shadow_os()
    return os.oracle.get_regime_shift_probability(domain=domain)


@router.get("/oracle/prophecy", dependencies=[Depends(verify_api_key)])
async def oracle_prophecy(horizon_days: int = Query(default=90, ge=1, le=365)):
    os = get_shadow_os()
    return os.oracle.get_prophecy(horizon_days=horizon_days)


# ─── FeedLoop Endpoints ──────────────────────────────────────────────────────

@router.get("/feedback/stats", dependencies=[Depends(verify_api_key)])
async def feedback_stats():
    os = get_shadow_os()
    return os.feed_loop.get_stats()

@router.get("/feedback/signals/{source_module}", dependencies=[Depends(verify_api_key)])
async def feedback_signals_by_source(source_module: str, limit: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return os.feed_loop.get_signals_by_source(source_module, limit=limit)

@router.get("/feedback/routes", dependencies=[Depends(verify_api_key)])
async def feedback_routes():
    os = get_shadow_os()
    return os.feed_loop.get_routes()

@router.get("/feedback/reverse-routes", dependencies=[Depends(verify_api_key)])
async def feedback_reverse_routes():
    os = get_shadow_os()
    return os.feed_loop.get_reverse_routes()

@router.get("/feedback/resonance", dependencies=[Depends(verify_api_key)])
async def feedback_resonance(
    min_signals: int = Query(default=3, ge=2, le=10),
    time_window: float = Query(default=300.0, ge=10.0, le=3600.0),
    min_intensity: float = Query(default=0.3, ge=0.0, le=1.0),
):
    os = get_shadow_os()
    return [e.to_dict() for e in os.feed_loop.detect_resonance(
        min_signals=min_signals, time_window_seconds=time_window, min_intensity=min_intensity
    )]

@router.get("/feedback/resonance/history", dependencies=[Depends(verify_api_key)])
async def feedback_resonance_history(limit: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return os.feed_loop.get_resonance_history(limit=limit)

@router.get("/feedback/active-loops", dependencies=[Depends(verify_api_key)])
async def feedback_active_loops():
    os = get_shadow_os()
    return os.feed_loop.get_active_loops()

@router.get("/feedback/signal-counts", dependencies=[Depends(verify_api_key)])
async def feedback_signal_counts():
    os = get_shadow_os()
    return os.feed_loop.get_signal_counts()

class FeedbackEmitRequest(BaseModel):
    source_module: str = Field(..., max_length=100)
    signal_type: str = Field(..., max_length=100)
    intensity: float = Field(..., ge=0.0, le=1.0)
    mode: str = Field(default="broadcast", max_length=50, pattern=r'^(broadcast|targeted|cascade)$')


@router.post("/feedback/emit", dependencies=[Depends(verify_api_key)])
async def feedback_emit(request: FeedbackEmitRequest):
    os = get_shadow_os()
    ids = os.feed_loop.emit(source_module=request.source_module, signal_type=request.signal_type,
                            intensity=request.intensity, mode=request.mode)
    return {"signal_ids": ids, "count": len(ids)}

@router.post("/feedback/poll/{target_module}", dependencies=[Depends(verify_api_key)])
async def feedback_poll(target_module: str, signal_type: Optional[str] = None, limit: int = Query(default=10, ge=1, le=100)):
    os = get_shadow_os()
    signals = os.feed_loop.poll(target_module=target_module, signal_type=signal_type, limit=limit)
    return [s.to_dict() for s in signals]


# --- Anticipatory Engine Endpoints ---

class AnticipateRequest(BaseModel):
    emotional_state: Optional[str] = Field(default=None, max_length=50)
    active_domains: List[str] = Field(default_factory=list, max_length=10)
    recent_actions: List[str] = Field(default_factory=list, max_length=20)
    goal_progress: Dict[str, float] = Field(default_factory=dict)
    time_of_day: int = Field(default=12, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)

    @field_validator("active_domains")
    @classmethod
    def validate_domains(cls, v):
        return [d[:50] for d in v]

    @field_validator("recent_actions")
    @classmethod
    def validate_actions(cls, v):
        return [a[:200] for a in v]


class ValidateAnticipationRequest(BaseModel):
    signal_id: str = Field(..., max_length=20)
    outcome: str = Field(..., pattern=r'^(correct|incorrect|partial)$')


class RecordSequenceRequest(BaseModel):
    actions: List[str] = Field(..., min_length=2, max_length=10)

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, v):
        return [a[:200] for a in v]


@router.post("/anticipate/predict", dependencies=[Depends(verify_api_key)])
async def anticipate_predict(req: AnticipateRequest):
    os = get_shadow_os()
    context = AnticipationContext(
        emotional_state=req.emotional_state,
        active_domains=req.active_domains,
        recent_actions=req.recent_actions,
        goal_progress=req.goal_progress,
        time_of_day=req.time_of_day,
        day_of_week=req.day_of_week,
    )
    signals = os.anticipatory.predict(context)
    return {"signals": [s.to_dict() for s in signals], "count": len(signals)}


@router.post("/anticipate/validate", dependencies=[Depends(verify_api_key)])
async def anticipate_validate(req: ValidateAnticipationRequest):
    os = get_shadow_os()
    success = os.anticipatory.validate(req.signal_id, req.outcome)
    if not success:
        raise HTTPException(status_code=404, detail="Signal not found")
    return {"validated": True, "signal_id": req.signal_id, "outcome": req.outcome}


@router.post("/anticipate/sequence", dependencies=[Depends(verify_api_key)])
async def anticipate_record_sequence(req: RecordSequenceRequest):
    os = get_shadow_os()
    os.anticipatory.record_sequence(req.actions)
    return {"recorded": True, "action_count": len(req.actions)}


@router.get("/anticipate/signals", dependencies=[Depends(verify_api_key)])
async def anticipate_signals(signal_type: Optional[str] = None, limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    signals = os.anticipatory.get_signals(signal_type=signal_type, limit=limit)
    return {"signals": [s.to_dict() for s in signals]}


@router.get("/anticipate/accuracy", dependencies=[Depends(verify_api_key)])
async def anticipate_accuracy(trigger: Optional[str] = None):
    os = get_shadow_os()
    return os.anticipatory.get_accuracy(trigger=trigger)


@router.get("/anticipate/history", dependencies=[Depends(verify_api_key)])
async def anticipate_history(limit: int = Query(default=100, ge=1, le=1000)):
    os = get_shadow_os()
    return os.anticipatory.get_history(limit=limit)


@router.get("/anticipate/stats", dependencies=[Depends(verify_api_key)])
async def anticipate_stats():
    os = get_shadow_os()
    return os.anticipatory.get_stats()


# --- TemporalSelf Endpoints ---

class MilestoneRequest(BaseModel):
    milestone_type: str = Field(..., max_length=50)
    title: str = Field(..., max_length=200)
    description: str = Field(default="", max_length=1000)
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = Field(default=None)
    tags: List[str] = Field(default_factory=list, max_length=10)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        return [t[:30] for t in v]

    @field_validator("context")
    @classmethod
    def validate_context(cls, v):
        if v is None:
            return {}
        return {str(k)[:50]: v for k, v in list(v.items())[:20]}


class IdentityMarkerRequest(BaseModel):
    category: str = Field(default="value", max_length=50)
    trait: str = Field(..., max_length=50)
    value: str = Field(..., max_length=200)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TrackDecisionRequest(BaseModel):
    context: str = Field(..., max_length=500)
    decision: str = Field(..., max_length=500)
    outcome: str = Field(default="", max_length=200)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list, max_length=5)


class ValidateMilestoneRequest(BaseModel):
    milestone_id: str = Field(..., max_length=20)


@router.post("/self/milestone", dependencies=[Depends(verify_api_key)])
async def record_milestone(req: MilestoneRequest):
    os = get_shadow_os()
    milestone = os.temporal_self.record_milestone(
        milestone_type=req.milestone_type,
        title=req.title,
        description=req.description,
        significance=req.significance,
        context=req.context,
        tags=req.tags,
    )
    return milestone.to_dict()


@router.get("/self/milestones", dependencies=[Depends(verify_api_key)])
async def get_milestones(milestone_type: Optional[str] = None, limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    milestones = os.temporal_self.get_milestones(milestone_type=milestone_type, limit=limit)
    return {"milestones": [m.to_dict() for m in milestones]}


@router.get("/self/milestone/{milestone_id}", dependencies=[Depends(verify_api_key)])
async def get_milestone(milestone_id: str):
    os = get_shadow_os()
    milestone = os.temporal_self.get_milestone(milestone_id)
    if not milestone:
        raise HTTPException(status_code=404, detail="Milestone not found")
    return milestone.to_dict()


@router.post("/self/identity", dependencies=[Depends(verify_api_key)])
async def add_identity_marker(req: IdentityMarkerRequest):
    os = get_shadow_os()
    marker = os.temporal_self.add_identity_marker(
        category=req.category,
        trait=req.trait,
        value=req.value,
        confidence=req.confidence,
    )
    return marker.to_dict()


@router.get("/self/identity", dependencies=[Depends(verify_api_key)])
async def get_identity():
    os = get_shadow_os()
    return os.temporal_self.get_identity()


@router.post("/self/decision", dependencies=[Depends(verify_api_key)])
async def track_decision(req: TrackDecisionRequest):
    os = get_shadow_os()
    decision_id = os.temporal_self.track_decision(
        context=req.context,
        decision=req.decision,
        outcome=req.outcome,
        confidence=req.confidence,
        tags=req.tags,
    )
    return {"decision_id": decision_id}


@router.get("/self/narrative", dependencies=[Depends(verify_api_key)])
async def generate_narrative(lookback: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    return os.temporal_self.generate_self_narrative(lookback=lookback)


@router.get("/self/coherence", dependencies=[Depends(verify_api_key)])
async def get_coherence():
    os = get_shadow_os()
    return {"coherence_score": os.temporal_self.get_coherence_score()}


@router.get("/self/epochs", dependencies=[Depends(verify_api_key)])
async def get_epochs(limit: int = Query(default=20, ge=1, le=50)):
    os = get_shadow_os()
    return {"epochs": [e.to_dict() for e in os.temporal_self.get_epochs(limit=limit)]}


@router.get("/self/stats", dependencies=[Depends(verify_api_key)])
async def temporal_self_stats():
    os = get_shadow_os()
    return os.temporal_self.get_stats()


# --- Health Dashboard Endpoints ---

class HealthModuleRequest(BaseModel):
    module_name: str = Field(..., max_length=50)
    initial_status: str = Field(default="healthy", max_length=30)


class HealthUpdateRequest(BaseModel):
    module_name: str = Field(..., max_length=50)
    latency_ms: float = Field(default=0.0, ge=0.0)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    error_count: int = Field(default=0, ge=0)
    status: Optional[str] = Field(default=None, max_length=20)
    details: Optional[Dict[str, Any]] = None


class ResolveAlertRequest(BaseModel):
    alert_id: str = Field(..., max_length=20)


@router.post("/health/register", dependencies=[Depends(verify_api_key)])
async def health_register_module(req: HealthModuleRequest):
    os = get_shadow_os()
    os.health_dashboard.register_module(req.module_name, req.initial_status)
    return {"registered": True, "module": req.module_name}


@router.post("/health/update", dependencies=[Depends(verify_api_key)])
async def health_update_module(req: HealthUpdateRequest):
    os = get_shadow_os()
    health = os.health_dashboard.update_module(
        module_name=req.module_name,
        latency_ms=req.latency_ms,
        success_rate=req.success_rate,
        error_count=req.error_count,
        status=req.status,
        details=req.details,
    )
    return health.to_dict()


@router.get("/health/modules", dependencies=[Depends(verify_api_key)])
async def health_get_modules():
    os = get_shadow_os()
    modules = os.health_dashboard.get_all_modules()
    return {"modules": [m.to_dict() for m in modules]}


@router.get("/health/module/{module_name}", dependencies=[Depends(verify_api_key)])
async def health_get_module(module_name: str):
    os = get_shadow_os()
    health = os.health_dashboard.get_module_health(module_name)
    if not health:
        raise HTTPException(status_code=404, detail="Module not found")
    return health.to_dict()


@router.get("/health/report", dependencies=[Depends(verify_api_key)])
async def health_report():
    os = get_shadow_os()
    report = os.health_dashboard.generate_report()
    return report.to_dict()


@router.get("/health/alerts", dependencies=[Depends(verify_api_key)])
async def health_alerts(severity: Optional[str] = None,
                        module_name: Optional[str] = None,
                        resolved: Optional[bool] = None,
                        limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    alerts = os.health_dashboard.get_alerts(
        severity=severity, module_name=module_name,
        resolved=resolved, limit=limit,
    )
    return {"alerts": [a.to_dict() for a in alerts]}


@router.post("/health/alert/{alert_id}/resolve", dependencies=[Depends(verify_api_key)])
async def health_resolve_alert(alert_id: str):
    os = get_shadow_os()
    result = os.health_dashboard.resolve_alert(alert_id)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"resolved": True, "alert_id": alert_id}


@router.get("/health/history", dependencies=[Depends(verify_api_key)])
async def health_history(limit: int = Query(default=100, ge=1, le=2000)):
    os = get_shadow_os()
    return {"history": os.health_dashboard.get_history(limit=limit)}


@router.get("/health/stats", dependencies=[Depends(verify_api_key)])
async def health_stats():
    os = get_shadow_os()
    return os.health_dashboard.get_stats()


@router.post("/health/refresh", dependencies=[Depends(verify_api_key)])
async def health_dashboard_refresh():
    os = get_shadow_os()
    return os.refresh_health_dashboard()


# ── Social Graph Propagation ───────────────────────────────────────────


class SocialRippleRequest(BaseModel):
    source: str = Field(..., max_length=64)
    ripple_type: str = Field(..., max_length=50)
    strength: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=500)
    person_id: Optional[str] = Field(default=None, max_length=64)
    group_id: Optional[str] = Field(default=None, max_length=64)
    conflict_id: Optional[str] = Field(default=None, max_length=64)
    metadata: Optional[Dict[str, str]] = None


class PropagationRuleRequest(BaseModel):
    source: str = Field(..., max_length=64)
    ripple_type: str = Field(..., max_length=50)
    target: str = Field(..., max_length=64)
    weight: float = Field(ge=0.0, le=2.0, default=0.5)
    strength_threshold: float = Field(ge=0.0, le=1.0, default=0.3)


class RuleToggleRequest(BaseModel):
    enabled: bool


@router.post("/social-propagation/ripple", dependencies=[Depends(verify_api_key)])
async def create_social_ripple(req: SocialRippleRequest):
    os = get_shadow_os()
    return os.propagate_social_ripple(
        source=req.source,
        ripple_type=req.ripple_type,
        strength=req.strength,
        reason=req.reason,
        person_id=req.person_id,
        group_id=req.group_id,
        conflict_id=req.conflict_id,
        metadata=req.metadata,
    )


@router.get("/social-propagation/ripples", dependencies=[Depends(verify_api_key)])
async def list_social_ripples(limit: int = Query(default=50, ge=1, le=500),
                               source: Optional[str] = None):
    os = get_shadow_os()
    return {"ripples": [r.to_dict() for r in os.social_propagation.get_ripples(limit=limit, source=source)]}


@router.get("/social-propagation/effects", dependencies=[Depends(verify_api_key)])
async def list_social_effects(limit: int = Query(default=50, ge=1, le=500),
                              target: Optional[str] = None):
    os = get_shadow_os()
    return {"effects": [e.to_dict() for e in os.social_propagation.get_effects(limit=limit, target=target)]}


@router.get("/social-propagation/rules", dependencies=[Depends(verify_api_key)])
async def list_propagation_rules():
    os = get_shadow_os()
    return {"rules": [r.to_dict() for r in os.social_propagation.get_rules()]}


@router.post("/social-propagation/rules", dependencies=[Depends(verify_api_key)])
async def add_propagation_rule(req: PropagationRuleRequest):
    os = get_shadow_os()
    rule = os.social_propagation.add_rule(
        source=req.source, ripple_type=req.ripple_type,
        target=req.target, weight=req.weight,
        strength_threshold=req.strength_threshold,
    )
    return rule.to_dict()


@router.post("/social-propagation/rules/{rule_id}/toggle", dependencies=[Depends(verify_api_key)])
async def toggle_propagation_rule(rule_id: str, req: RuleToggleRequest):
    os = get_shadow_os()
    success = os.social_propagation.toggle_rule(rule_id, req.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"rule_id": rule_id, "enabled": req.enabled}


@router.get("/social-propagation/graph", dependencies=[Depends(verify_api_key)])
async def get_social_graph():
    os = get_shadow_os()
    return os.build_social_graph()


@router.post("/social-propagation/graph/build", dependencies=[Depends(verify_api_key)])
async def build_social_graph():
    os = get_shadow_os()
    return os.build_social_graph()


@router.get("/social-propagation/summary", dependencies=[Depends(verify_api_key)])
async def social_propagation_summary():
    os = get_shadow_os()
    return os.social_propagation.get_propagation_summary()


@router.get("/social-propagation/stats", dependencies=[Depends(verify_api_key)])
async def social_propagation_stats():
    os = get_shadow_os()
    return os.social_propagation.get_stats()


# ═════════════════════════════════════════════════════════════════════
# NEURAL-SYMBOLIC BRIDGE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

from .neural_symbolic_bridge import NeuralSymbolicBridge, PatternType, RuleStrength, InferenceMode
from .quantum_decision_field import QuantumDecisionField
from .emergent_grammar import EmergentGrammar
from .holographic_memory import HolographicMemory
from .syncognitive_feedback import SyncognitiveFeedback, FeedbackType
from .morphic_resonance_engine import MorphicResonanceEngine
from .existential_calculus import ExistentialCalculus, SignificanceAxis
from .chrono_synthesis import ChronoSynthesis
from .phoenix_regenerator import PhoenixRegenerator, HealthStatus


class NeuralPatternRequest(BaseModel):
    pattern_type: str = Field(..., max_length=50)
    dimensions: Dict[str, float] = Field(default_factory=dict)
    symbol: str = Field(default="", max_length=100)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        return _validate_dict_keys(v)


class NeuralRuleRequest(BaseModel):
    premises: List[str] = Field(default_factory=list, max_length=20)
    conclusion: str = Field(..., max_length=500)
    strength: str = Field(default="moderate", max_length=20)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: str = Field(default="", max_length=100)

    @field_validator('premises')
    @classmethod
    def validate_premises(cls, v):
        if v is not None:
            return [p[:_MAX_ELEMENT_LENGTH] for p in v]
        return v


class NeuralMapRequest(BaseModel):
    pattern_id: str = Field(..., max_length=64)
    rule_id: str = Field(..., max_length=64)
    mode: str = Field(default="bidirectional", max_length=30)


class NeuroTranslateRequest(BaseModel):
    pattern_id: str = Field(..., max_length=64)


class SymbolicTranslateRequest(BaseModel):
    rule_id: str = Field(..., max_length=64)


class BidirectionalTranslateRequest(BaseModel):
    source_id: str = Field(..., max_length=64)
    source_type: str = Field(default="pattern", max_length=20)


@router.post("/neural-bridge/pattern", dependencies=[Depends(verify_api_key)])
async def neural_bridge_add_pattern(request: NeuralPatternRequest):
    os = get_shadow_os()
    try:
        ptype = PatternType(request.pattern_type)
    except ValueError:
        ptype = PatternType.ACTIVATION
    try:
        pattern = os.neural_bridge.add_pattern(
            pattern_type=ptype, dimensions=request.dimensions,
            symbol=request.symbol or None, confidence=request.confidence,
        )
        return pattern.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/neural-bridge/rule", dependencies=[Depends(verify_api_key)])
async def neural_bridge_add_rule(request: NeuralRuleRequest):
    os = get_shadow_os()
    try:
        strength = RuleStrength(request.strength)
    except ValueError:
        strength = RuleStrength.MODERATE
    rule = os.neural_bridge.add_rule(
        premises=request.premises, conclusion=request.conclusion,
        strength=strength, confidence=request.confidence, domain=request.domain,
    )
    return rule.to_dict()


@router.post("/neural-bridge/map", dependencies=[Depends(verify_api_key)])
async def neural_bridge_create_mapping(request: NeuralMapRequest):
    os = get_shadow_os()
    try:
        mode = InferenceMode(request.mode)
    except ValueError:
        mode = InferenceMode.BIDIRECTIONAL
    mapping = os.neural_bridge.create_mapping(
        neural_pattern_id=request.pattern_id,
        symbolic_rule_id=request.rule_id, mode=mode,
    )
    if not mapping:
        raise HTTPException(status_code=404, detail="Pattern or rule not found")
    return mapping.to_dict()


@router.post("/neural-bridge/translate/neuro-to-symbolic", dependencies=[Depends(verify_api_key)])
async def neural_bridge_neuro_to_symbolic(request: NeuroTranslateRequest):
    os = get_shadow_os()
    rules = os.neural_bridge.translate_neuro_to_symbolic(request.pattern_id)
    return {"rules": [r.to_dict() for r in rules]}


@router.post("/neural-bridge/translate/symbolic-to-neuro", dependencies=[Depends(verify_api_key)])
async def neural_bridge_symbolic_to_neuro(request: SymbolicTranslateRequest):
    os = get_shadow_os()
    patterns = os.neural_bridge.translate_symbolic_to_neuro(request.rule_id)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.post("/neural-bridge/translate/bidirectional", dependencies=[Depends(verify_api_key)])
async def neural_bridge_bidirectional(request: BidirectionalTranslateRequest):
    os = get_shadow_os()
    result = os.neural_bridge.translate_bidirectional(request.source_id, request.source_type)
    return result


@router.get("/neural-bridge/pattern/{pattern_id}", dependencies=[Depends(verify_api_key)])
async def neural_bridge_get_pattern(pattern_id: str):
    os = get_shadow_os()
    pattern = os.neural_bridge.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern.to_dict()


@router.get("/neural-bridge/rules", dependencies=[Depends(verify_api_key)])
async def neural_bridge_list_rules(domain: Optional[str] = Query(default=None, max_length=100)):
    os = get_shadow_os()
    if domain:
        rules = os.neural_bridge.find_rules_by_domain(domain)
    else:
        rules = os.neural_bridge.list_rules()
    return {"rules": [r.to_dict() for r in rules]}


@router.get("/neural-bridge/stats", dependencies=[Depends(verify_api_key)])
async def neural_bridge_stats():
    os = get_shadow_os()
    return os.neural_bridge.get_stats()


# ═════════════════════════════════════════════════════════════════════
# QUANTUM DECISION FIELD ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class QuantumDecisionRequest(BaseModel):
    options: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)
    context: str = Field(default="", max_length=500)


class QuantumObserveRequest(BaseModel):
    option_id: str = Field(..., max_length=64)


class QuantumEntangleRequest(BaseModel):
    decision_id_1: str = Field(..., max_length=64)
    decision_id_2: str = Field(..., max_length=64)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/quantum-field/decision", dependencies=[Depends(verify_api_key)])
async def quantum_field_create_decision(request: QuantumDecisionRequest):
    os = get_shadow_os()
    decision = os.quantum_field.create_decision(options=request.options, context=request.context or None)
    return decision.to_dict()


@router.post("/quantum-field/decision/{decision_id}/observe", dependencies=[Depends(verify_api_key)])
async def quantum_field_observe(decision_id: str, request: QuantumObserveRequest):
    os = get_shadow_os()
    option = os.quantum_field.observe(decision_id, request.option_id)
    if not option:
        raise HTTPException(status_code=404, detail="Decision or option not found")
    return option.to_dict()


@router.post("/quantum-field/decision/{decision_id}/collapse", dependencies=[Depends(verify_api_key)])
async def quantum_field_collapse(decision_id: str):
    os = get_shadow_os()
    decision = os.quantum_field.collapse(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision.to_dict()


@router.post("/quantum-field/entangle", dependencies=[Depends(verify_api_key)])
async def quantum_field_entangle(request: QuantumEntangleRequest):
    os = get_shadow_os()
    link = os.quantum_field.entangle(request.decision_id_1, request.decision_id_2, correlation=request.strength)
    return link.to_dict()


@router.get("/quantum-field/decision/{decision_id}", dependencies=[Depends(verify_api_key)])
async def quantum_field_get_decision(decision_id: str):
    os = get_shadow_os()
    decision = os.quantum_field.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision.to_dict()


@router.get("/quantum-field/decisions", dependencies=[Depends(verify_api_key)])
async def quantum_field_list_decisions(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    decisions = os.quantum_field.list_decisions(limit=limit)
    return {"decisions": [d.to_dict() for d in decisions]}


@router.get("/quantum-field/entanglements", dependencies=[Depends(verify_api_key)])
async def quantum_field_list_entanglements(decision_id: str = Query(default="", max_length=64)):
    os = get_shadow_os()
    if decision_id:
        links = os.quantum_field.get_entanglements(decision_id)
    else:
        links = os.quantum_field.get_entanglements("")
    return {"entanglements": [l.to_dict() for l in links]}


@router.get("/quantum-field/stats", dependencies=[Depends(verify_api_key)])
async def quantum_field_stats():
    os = get_shadow_os()
    return os.quantum_field.get_stats()


# ═════════════════════════════════════════════════════════════════════
# EMERGENT GRAMMAR ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class GrammarObserveRequest(BaseModel):
    pattern: str = Field(..., max_length=5000)
    context: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: str = Field(default="", max_length=100)


class GrammarComposeRequest(BaseModel):
    rule_ids: List[str] = Field(default_factory=list, max_length=20)


class GrammarAbstractRequest(BaseModel):
    symbol: str = Field(..., max_length=100)
    level: int = Field(default=1, ge=1, le=5)


class GrammarRefineRequest(BaseModel):
    rule_id: str = Field(..., max_length=64)
    pattern: str = Field(default="", max_length=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/grammar/observe", dependencies=[Depends(verify_api_key)])
async def grammar_observe(request: GrammarObserveRequest):
    os = get_shadow_os()
    sentence = os.grammar.observe(
        pattern=request.pattern, context=request.context,
        confidence=request.confidence, domain=request.domain,
    )
    return sentence.to_dict()


@router.post("/grammar/compose", dependencies=[Depends(verify_api_key)])
async def grammar_compose(request: GrammarComposeRequest):
    os = get_shadow_os()
    sentence = os.grammar.compose(symbols=request.rule_ids)
    return sentence.to_dict()


@router.post("/grammar/abstract", dependencies=[Depends(verify_api_key)])
async def grammar_abstract(request: GrammarAbstractRequest):
    os = get_shadow_os()
    symbol = os.grammar.abstract(pattern=request.symbol, level=request.level)
    return symbol.to_dict()


@router.post("/grammar/refine", dependencies=[Depends(verify_api_key)])
async def grammar_refine(request: GrammarRefineRequest):
    os = get_shadow_os()
    success = os.grammar.refine(rule_id=request.rule_id, new_confidence=request.confidence)
    return {"success": success, "rule_id": request.rule_id}


@router.get("/grammar/rules", dependencies=[Depends(verify_api_key)])
async def grammar_list_rules(domain: Optional[str] = Query(default=None, max_length=100)):
    os = get_shadow_os()
    domain_val = domain if domain else ""
    rules = os.grammar.find_rules(domain=domain_val)
    return {"rules": [r.to_dict() for r in rules]}


@router.get("/grammar/symbols", dependencies=[Depends(verify_api_key)])
async def grammar_list_symbols(min_usage: int = Query(default=0, ge=0)):
    os = get_shadow_os()
    symbols = os.grammar.find_symbols(min_usage=min_usage)
    return {"symbols": [s.to_dict() for s in symbols]}


@router.get("/grammar/stats", dependencies=[Depends(verify_api_key)])
async def grammar_stats():
    os = get_shadow_os()
    return os.grammar.get_stats()


# ═════════════════════════════════════════════════════════════════════
# HOLOGRAPHIC MEMORY ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class HolographicEncodeRequest(BaseModel):
    content: str = Field(..., max_length=65536)
    fragment_type: str = Field(default="partial", max_length=30)


class HolographicEncodeFullRequest(BaseModel):
    content: str = Field(..., max_length=65536)
    redundancy: int = Field(default=3, ge=1, le=10)


class HolographicReconstructRequest(BaseModel):
    fragment_ids: List[str] = Field(default_factory=list, max_length=20)


class HolographicPartialReconstructRequest(BaseModel):
    fragment_ids: List[str] = Field(default_factory=list, max_length=20)
    ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class HolographicVerifyRequest(BaseModel):
    fragment_ids: List[str] = Field(default_factory=list, max_length=20)


@router.post("/holographic/encode", dependencies=[Depends(verify_api_key)])
async def holographic_encode(request: HolographicEncodeRequest):
    os = get_shadow_os()
    fragment = os.holographic.encode(content=request.content, metadata={"fragment_type": request.fragment_type})
    return fragment.to_dict()


@router.post("/holographic/encode-full", dependencies=[Depends(verify_api_key)])
async def holographic_encode_full(request: HolographicEncodeFullRequest):
    os = get_shadow_os()
    fragments = os.holographic.encode_full(content=request.content, redundancy=request.redundancy)
    return {"fragments": [f.to_dict() for f in fragments]}


@router.get("/holographic/fragment/{fragment_id}", dependencies=[Depends(verify_api_key)])
async def holographic_retrieve(fragment_id: str):
    os = get_shadow_os()
    fragment = os.holographic.retrieve(fragment_id)
    if not fragment:
        raise HTTPException(status_code=404, detail="Fragment not found")
    return fragment.to_dict()


@router.post("/holographic/reconstruct", dependencies=[Depends(verify_api_key)])
async def holographic_reconstruct(request: HolographicReconstructRequest):
    os = get_shadow_os()
    result = os.holographic.reconstruct(fragment_ids=request.fragment_ids)
    return result.to_dict()


@router.post("/holographic/partial-reconstruct", dependencies=[Depends(verify_api_key)])
async def holographic_partial_reconstruct(request: HolographicPartialReconstructRequest):
    os = get_shadow_os()
    result = os.holographic.partial_reconstruct(
        fragment_ids=request.fragment_ids, min_quality=request.ratio,
    )
    return result.to_dict()


@router.get("/holographic/fragments", dependencies=[Depends(verify_api_key)])
async def holographic_find_fragments(content_query: str = Query(default="", max_length=500)):
    os = get_shadow_os()
    fragments = os.holographic.find_fragments(query=content_query)
    return {"fragments": [f.to_dict() for f in fragments]}


@router.post("/holographic/verify", dependencies=[Depends(verify_api_key)])
async def holographic_verify(request: HolographicVerifyRequest):
    os = get_shadow_os()
    results = {}
    for fid in request.fragment_ids:
        results[fid] = os.holographic.verify_integrity(fid)
    return {"results": results}


@router.get("/holographic/stats", dependencies=[Depends(verify_api_key)])
async def holographic_stats():
    os = get_shadow_os()
    return os.holographic.get_stats()


# ═════════════════════════════════════════════════════════════════════
# SYNCOGNITIVE FEEDBACK ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class SyncognitiveObserveRequest(BaseModel):
    process_name: str = Field(..., max_length=100)
    details: str = Field(..., max_length=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback_type: str = Field(default="self_correction", max_length=30)


class SyncognitiveLoopRequest(BaseModel):
    source: str = Field(..., max_length=100)
    target: str = Field(..., max_length=100)
    feedback_type: str = Field(default="self_correction", max_length=30)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class SyncognitivePropagateRequest(BaseModel):
    loop_id: str = Field(..., max_length=64)
    magnitude: float = Field(default=0.1, ge=-1.0, le=1.0)


class SyncognitiveCorrectRequest(BaseModel):
    process_name: str = Field(..., max_length=100)
    correction: str = Field(..., max_length=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SyncognitiveBiasRequest(BaseModel):
    process_name: str = Field(..., max_length=100)
    sample_size: int = Field(default=10, ge=1, le=100)


@router.post("/syncognitive/observe", dependencies=[Depends(verify_api_key)])
async def syncognitive_observe(request: SyncognitiveObserveRequest):
    os = get_shadow_os()
    try:
        ftype = FeedbackType(request.feedback_type)
    except ValueError:
        ftype = FeedbackType.SELF_CORRECTION
    observation = os.syncognitive.observe_process(
        process_name=request.process_name, details=request.details,
        confidence=request.confidence, observation_type=ftype,
    )
    return observation.to_dict()


@router.post("/syncognitive/loop", dependencies=[Depends(verify_api_key)])
async def syncognitive_create_loop(request: SyncognitiveLoopRequest):
    os = get_shadow_os()
    try:
        ftype = FeedbackType(request.feedback_type)
    except ValueError:
        ftype = FeedbackType.SELF_CORRECTION
    loop = os.syncognitive.create_feedback_loop(
        source=request.source, target=request.target,
        feedback_type=ftype, strength=request.strength,
    )
    return loop.to_dict()


@router.post("/syncognitive/propagate", dependencies=[Depends(verify_api_key)])
async def syncognitive_propagate(request: SyncognitivePropagateRequest):
    os = get_shadow_os()
    magnitude = request.magnitude
    loop_id = request.loop_id
    try:
        from .syncognitive_feedback import CognitiveObservation
        obs = CognitiveObservation(
            observation_id="propagate",
            process_name="propagate",
            observation_type=FeedbackType.CONFIDENCE_ADJUSTMENT,
            details="propagation trigger",
            confidence_before=0.5,
            confidence_after=min(1.0, max(0.0, 0.5 + magnitude)),
        )
        result = os.syncognitive.propagate_feedback(loop_id, obs)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/syncognitive/correct", dependencies=[Depends(verify_api_key)])
async def syncognitive_correct(request: SyncognitiveCorrectRequest):
    os = get_shadow_os()
    observation = os.syncognitive.correct_self(
        process_name=request.process_name, correction=request.correction,
        confidence_delta=request.confidence,
    )
    return observation.to_dict()


@router.post("/syncognitive/bias", dependencies=[Depends(verify_api_key)])
async def syncognitive_bias(request: SyncognitiveBiasRequest):
    os = get_shadow_os()
    observations = os.syncognitive.detect_bias(process_name=request.process_name)
    return {"bias_observations": [o.to_dict() for o in observations]}


@router.get("/syncognitive/profile/{process_name}", dependencies=[Depends(verify_api_key)])
async def syncognitive_get_profile(process_name: str):
    os = get_shadow_os()
    profile = os.syncognitive.get_process_profile(process_name)
    if not profile:
        raise HTTPException(status_code=404, detail="Process profile not found")
    return profile.to_dict()


@router.get("/syncognitive/observations", dependencies=[Depends(verify_api_key)])
async def syncognitive_list_observations(
    process_name: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    pname = process_name if process_name else ""
    observations = os.syncognitive.list_observations(process_name=pname, limit=limit)
    return {"observations": [o.to_dict() for o in observations]}


@router.get("/syncognitive/stats", dependencies=[Depends(verify_api_key)])
async def syncognitive_stats():
    os = get_shadow_os()
    return os.syncognitive.get_stats()


# ═════════════════════════════════════════════════════════════════════
# MORPHIC RESONANCE ENGINE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class MorphicPatternRequest(BaseModel):
    domain: str = Field(..., max_length=100)
    pattern: str = Field(..., max_length=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MorphicPropagateRequest(BaseModel):
    source_domain: str = Field(..., max_length=100)
    target_domain: str = Field(..., max_length=100)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class MorphicDetectRequest(BaseModel):
    domain_a: str = Field(..., max_length=100)
    domain_b: str = Field(..., max_length=5000)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class MorphicInfluenceRequest(BaseModel):
    influence_id: str = Field(..., max_length=64)


class MorphicDecayRequest(BaseModel):
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)


@router.post("/morphic/pattern", dependencies=[Depends(verify_api_key)])
async def morphic_register_pattern(request: MorphicPatternRequest):
    os = get_shadow_os()
    pattern = os.morphic.register_pattern(
        domain=request.domain, pattern=request.pattern, confidence=request.confidence,
    )
    return pattern.to_dict()


@router.post("/morphic/propagate", dependencies=[Depends(verify_api_key)])
async def morphic_propagate(request: MorphicPropagateRequest):
    os = get_shadow_os()
    influences = os.morphic.propagate(
        source_domain=request.source_domain, target_domain=request.target_domain,
        min_confidence=request.strength,
    )
    if not influences:
        return {"influence_id": "", "source_domain": request.source_domain, "target_domain": request.target_domain, "strength": request.strength}
    return influences[0].to_dict()


@router.post("/morphic/detect", dependencies=[Depends(verify_api_key)])
async def morphic_detect(request: MorphicDetectRequest):
    os = get_shadow_os()
    patterns = os.morphic.detect_resonance(domain=request.domain_a, pattern=request.domain_b)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.post("/morphic/influence", dependencies=[Depends(verify_api_key)])
async def morphic_apply_influence(request: MorphicInfluenceRequest):
    os = get_shadow_os()
    success = os.morphic.apply_influence(request.influence_id)
    if not success:
        raise HTTPException(status_code=404, detail="Influence not found")
    return {"applied": True, "influence_id": request.influence_id}


@router.post("/morphic/decay", dependencies=[Depends(verify_api_key)])
async def morphic_decay(request: MorphicDecayRequest):
    os = get_shadow_os()
    removed_count = os.morphic.decay_influences()
    return {"removed_count": removed_count}


@router.get("/morphic/field/{domain}", dependencies=[Depends(verify_api_key)])
async def morphic_get_field(domain: str):
    os = get_shadow_os()
    field = os.morphic.get_field(domain)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field.to_dict()


@router.get("/morphic/strength", dependencies=[Depends(verify_api_key)])
async def morphic_resonance_strength(
    domain_a: str = Query(..., max_length=100),
    domain_b: str = Query(..., max_length=100),
):
    os = get_shadow_os()
    patterns = os.morphic.get_field(domain_a)
    pid_a = ""
    if patterns and hasattr(patterns, 'to_dict'):
        d = patterns.to_dict()
        pats = d.get("patterns", [])
        if pats:
            pid_a = pats[0].get("pattern_id", "")
    strength = os.morphic.compute_resonance_strength(pid_a or domain_a, domain_b)
    return {"domain_a": domain_a, "domain_b": domain_b, "strength": strength}


@router.get("/morphic/stats", dependencies=[Depends(verify_api_key)])
async def morphic_stats():
    os = get_shadow_os()
    return os.morphic.get_stats()


# ═════════════════════════════════════════════════════════════════════
# EXISTENTIAL CALCULUS ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ExistentialEvaluateRequest(BaseModel):
    event_id: str = Field(..., max_length=100)
    description: str = Field(..., max_length=5000)
    axis_weights: Dict[str, float] = Field(default_factory=dict)

    @field_validator('axis_weights')
    @classmethod
    def validate_axis_weights(cls, v):
        return _validate_dict_keys(v)


class ExistentialAlignmentRequest(BaseModel):
    purpose: str = Field(..., max_length=1000)
    activities: List[str] = Field(default_factory=list, max_length=20)


class ExistentialThemesRequest(BaseModel):
    min_events: int = Field(default=2, ge=1, le=100)


class ExistentialFrameRequest(BaseModel):
    title: str = Field(..., max_length=200)
    axis_weights: Dict[str, float] = Field(default_factory=dict)

    @field_validator('axis_weights')
    @classmethod
    def validate_axis_weights(cls, v):
        return _validate_dict_keys(v)


@router.post("/existential/evaluate", dependencies=[Depends(verify_api_key)])
async def existential_evaluate(request: ExistentialEvaluateRequest):
    os = get_shadow_os()
    axis_weights = {}
    for k, v in request.axis_weights.items():
        try:
            axis = SignificanceAxis(k)
            axis_weights[axis] = v
        except ValueError:
            axis_weights[k] = v
    result = os.existential.evaluate_significance(
        event_id=request.event_id, description=request.description,
        axis_weights=axis_weights if axis_weights else None,
    )
    return result.to_dict()


@router.post("/existential/alignment", dependencies=[Depends(verify_api_key)])
async def existential_alignment(request: ExistentialAlignmentRequest):
    os = get_shadow_os()
    alignment = os.existential.compute_purpose_alignment(
        purpose=request.purpose, activities=request.activities,
    )
    return alignment.to_dict()


@router.post("/existential/themes", dependencies=[Depends(verify_api_key)])
async def existential_themes(request: ExistentialThemesRequest):
    os = get_shadow_os()
    themes = os.existential.detect_life_themes(min_frequency=request.min_events)
    return {"themes": [t.to_dict() for t in themes]}


@router.post("/existential/frame", dependencies=[Depends(verify_api_key)])
async def existential_add_frame(request: ExistentialFrameRequest):
    os = get_shadow_os()
    axis_scores = {}
    for k, v in request.axis_weights.items():
        axis_scores[k] = v
    frame = os.existential.add_meaning_frame(
        event_id="", axis_scores=axis_scores, narrative=request.title,
    )
    return frame.to_dict()


@router.get("/existential/frames", dependencies=[Depends(verify_api_key)])
async def existential_list_frames(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    results = os.existential.list_results(limit=limit)
    return {"frames": [r.to_dict() for r in results]}


@router.get("/existential/recommend", dependencies=[Depends(verify_api_key)])
async def existential_recommend(context: str = Query(..., max_length=500)):
    os = get_shadow_os()
    recommendation = os.existential.recommend_action(event_id=context)
    return {"recommendation": recommendation}


@router.get("/existential/stats", dependencies=[Depends(verify_api_key)])
async def existential_stats():
    os = get_shadow_os()
    return os.existential.get_stats()


# ═════════════════════════════════════════════════════════════════════
# CHRONO SYNTHESIS ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ChronoPastPatternRequest(BaseModel):
    event_type: str = Field(..., max_length=100)
    domain: str = Field(default="", max_length=100)
    frequency: int = Field(default=1, ge=1, le=100)


class ChronoFutureProjectionRequest(BaseModel):
    event_type: str = Field(..., max_length=100)
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    timeframe: str = Field(default="1h", max_length=50)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ChronoContextRequest(BaseModel):
    name: str = Field(..., max_length=200)
    past_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    future_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class ChronoSynthesizeRequest(BaseModel):
    context_id: str = Field(..., max_length=64)


class ChronoPastWeightRequest(BaseModel):
    pattern_id: str = Field(..., max_length=64)


class ChronoFutureWeightRequest(BaseModel):
    projection_id: str = Field(..., max_length=64)


@router.post("/chrono/past-pattern", dependencies=[Depends(verify_api_key)])
async def chrono_past_pattern(request: ChronoPastPatternRequest):
    os = get_shadow_os()
    pattern = os.chrono.record_past_pattern(
        event_type=request.event_type, domain=request.domain,
        frequency=request.frequency,
    )
    return pattern.to_dict()


@router.post("/chrono/future-projection", dependencies=[Depends(verify_api_key)])
async def chrono_future_projection(request: ChronoFutureProjectionRequest):
    os = get_shadow_os()
    projection = os.chrono.add_future_projection(
        event_type=request.event_type, probability=request.probability,
        timeframe=request.timeframe, confidence=request.confidence,
    )
    return projection.to_dict()


@router.post("/chrono/context", dependencies=[Depends(verify_api_key)])
async def chrono_create_context(request: ChronoContextRequest):
    os = get_shadow_os()
    context = os.chrono.create_temporal_context(
        past_pattern_ids=[], present_factors={"past_weight": request.past_weight, "future_weight": request.future_weight},
        future_projection_ids=[], domain=request.name, urgency=0.5,
    )
    return context.to_dict()


@router.post("/chrono/synthesize", dependencies=[Depends(verify_api_key)])
async def chrono_synthesize(request: ChronoSynthesizeRequest):
    os = get_shadow_os()
    decision = os.chrono.synthesize(context_id=request.context_id)
    return decision.to_dict()


@router.get("/chrono/past-patterns", dependencies=[Depends(verify_api_key)])
async def chrono_list_past_patterns(domain: Optional[str] = Query(default=None, max_length=100)):
    os = get_shadow_os()
    patterns = os.chrono.list_past_patterns(domain=domain)
    return {"patterns": [p.to_dict() for p in patterns]}


@router.get("/chrono/future-projections", dependencies=[Depends(verify_api_key)])
async def chrono_list_projections(event_type: Optional[str] = Query(default=None, max_length=100)):
    os = get_shadow_os()
    projections = os.chrono.list_future_projections(event_type=event_type)
    return {"projections": [p.to_dict() for p in projections]}


@router.get("/chrono/decisions", dependencies=[Depends(verify_api_key)])
async def chrono_list_decisions(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    decisions = os.chrono.list_decisions(limit=limit)
    return {"decisions": [d.to_dict() for d in decisions]}


@router.post("/chrono/past-weight", dependencies=[Depends(verify_api_key)])
async def chrono_compute_past_weight(request: ChronoPastWeightRequest):
    os = get_shadow_os()
    weight = os.chrono.compute_past_weight(request.pattern_id)
    return {"pattern_id": request.pattern_id, "weight": weight}


@router.post("/chrono/future-weight", dependencies=[Depends(verify_api_key)])
async def chrono_compute_future_weight(request: ChronoFutureWeightRequest):
    os = get_shadow_os()
    weight = os.chrono.compute_future_weight(request.projection_id)
    return {"projection_id": request.projection_id, "weight": weight}


@router.get("/chrono/stats", dependencies=[Depends(verify_api_key)])
async def chrono_stats():
    os = get_shadow_os()
    return os.chrono.get_stats()


# ═════════════════════════════════════════════════════════════════════
# PHOENIX REGENERATOR ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class PhoenixComponentRequest(BaseModel):
    component_id: str = Field(..., max_length=64)
    module_name: str = Field(..., max_length=200)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)


class PhoenixErrorRequest(BaseModel):
    component_id: str = Field(..., max_length=64)
    error_message: str = Field(..., max_length=5000)
    severity: str = Field(default="medium", max_length=20)


class PhoenixDiagnoseRequest(BaseModel):
    component_id: str = Field(..., max_length=64)


class PhoenixPlanRequest(BaseModel):
    component_id: str = Field(..., max_length=64)


class PhoenixExecuteRequest(BaseModel):
    plan_id: str = Field(..., max_length=64)


class PhoenixPreserveRequest(BaseModel):
    component_id: str = Field(..., max_length=64)


class PhoenixRestoreRequest(BaseModel):
    component_id: str = Field(..., max_length=64)


@router.post("/phoenix/component", dependencies=[Depends(verify_api_key)])
async def phoenix_register_component(request: PhoenixComponentRequest):
    os = get_shadow_os()
    health = os.phoenix.register_component(
        component_id=request.component_id, component_name=request.module_name,
        initial_health=request.health_score,
    )
    return health.to_dict()


@router.get("/phoenix/component/{component_id}", dependencies=[Depends(verify_api_key)])
async def phoenix_check_health(component_id: str):
    os = get_shadow_os()
    health = os.phoenix.check_health(component_id)
    if not health:
        raise HTTPException(status_code=404, detail="Component not found")
    return health.to_dict()


@router.post("/phoenix/error", dependencies=[Depends(verify_api_key)])
async def phoenix_report_error(request: PhoenixErrorRequest):
    os = get_shadow_os()
    diagnosis = os.phoenix.report_error(request.component_id, request.error_message)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Component not found")
    return diagnosis.to_dict()


@router.post("/phoenix/diagnose", dependencies=[Depends(verify_api_key)])
async def phoenix_diagnose(request: PhoenixDiagnoseRequest):
    os = get_shadow_os()
    diagnosis = os.phoenix.diagnose(request.component_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Component not found")
    return diagnosis.to_dict()


@router.post("/phoenix/plan", dependencies=[Depends(verify_api_key)])
async def phoenix_create_plan(request: PhoenixPlanRequest):
    os = get_shadow_os()
    plan = os.phoenix.create_regeneration_plan(request.component_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Component not found")
    return plan.to_dict()


@router.post("/phoenix/execute", dependencies=[Depends(verify_api_key)])
async def phoenix_execute(request: PhoenixExecuteRequest):
    os = get_shadow_os()
    result = os.phoenix.execute_regeneration(request.plan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plan not found")
    return result.to_dict()


@router.post("/phoenix/preserve", dependencies=[Depends(verify_api_key)])
async def phoenix_preserve(request: PhoenixPreserveRequest):
    os = get_shadow_os()
    learning = os.phoenix.preserve_learning(request.component_id)
    if not learning:
        raise HTTPException(status_code=404, detail="Component not found")
    return {"learning": learning}


@router.post("/phoenix/restore", dependencies=[Depends(verify_api_key)])
async def phoenix_restore(request: PhoenixRestoreRequest):
    os = get_shadow_os()
    success = os.phoenix.restore_learning(request.component_id, {})
    if not success:
        raise HTTPException(status_code=404, detail="Component not found")
    return {"restored": True, "component_id": request.component_id}


@router.get("/phoenix/components", dependencies=[Depends(verify_api_key)])
async def phoenix_list_components(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    components = os.phoenix.list_components(limit=limit)
    return {"components": [c.to_dict() for c in components]}


@router.get("/phoenix/diagnoses", dependencies=[Depends(verify_api_key)])
async def phoenix_list_diagnoses(
    component_id: Optional[str] = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    cid = component_id if component_id else ""
    diagnoses = os.phoenix.list_diagnoses(component_id=cid, limit=limit)
    return {"diagnoses": [d.to_dict() for d in diagnoses]}


@router.get("/phoenix/stats", dependencies=[Depends(verify_api_key)])
async def phoenix_stats():
    os = get_shadow_os()
    return os.phoenix.get_stats()


# ═════════════════════════════════════════════════════════════════════
# CAUSAL REASONING ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

from .causal_reasoning import CausalReasoning, RelationType, EvidenceStrength
from .quantum_memory_field import QuantumMemoryField, EntanglementType, AccessType
from .semantic_drift_detector import SemanticDriftDetector, ConceptStatus
from .cognitive_immune_system import CognitiveImmuneSystem, PathogenType
from .temporal_parallax import TemporalParallax


class CausalVariableRequest(BaseModel):
    name: str = Field(..., max_length=200)
    var_type: str = Field(default="", max_length=50)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class CausalCauseRequest(BaseModel):
    source_id: str = Field(..., max_length=64)
    target_id: str = Field(..., max_length=64)
    relation_type: str = Field(default="causes", max_length=30)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = Field(default="moderate", max_length=30)
    confounders: Optional[List[str]] = None

    @field_validator('confounders', mode='before')
    @classmethod
    def validate_confounders(cls, v):
        if v is not None:
            return [c[:_MAX_ELEMENT_LENGTH] for c in v]
        return v


class CausalInterveneRequest(BaseModel):
    variable_id: str = Field(..., max_length=64)
    do_value: str = Field(..., max_length=500)
    context: Optional[Dict[str, Any]] = None

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        return _validate_dict_keys(v)


class CausalCounterfactualRequest(BaseModel):
    premise: str = Field(..., max_length=2000)
    observed_outcome: str = Field(..., max_length=2000)
    alternative_outcome: str = Field(..., max_length=2000)


@router.post("/causal/variable", dependencies=[Depends(verify_api_key)])
async def causal_add_variable(request: CausalVariableRequest):
    os = get_shadow_os()
    try:
        variable = os.causal.add_variable(request.name, request.var_type, request.metadata)
        return variable.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}")


@router.post("/causal/cause", dependencies=[Depends(verify_api_key)])
async def causal_add_cause(request: CausalCauseRequest):
    os = get_shadow_os()
    try:
        rtype = RelationType(request.relation_type)
    except ValueError:
        rtype = RelationType.CAUSES
    try:
        evidence = EvidenceStrength(request.evidence)
    except ValueError:
        evidence = EvidenceStrength.MODERATE
    edge = os.causal.add_cause(
        request.source_id, request.target_id, rtype, request.strength, evidence, request.confounders
    )
    if not edge:
        raise HTTPException(status_code=400, detail="Invalid cause: source or target not found")
    return edge.to_dict()


@router.delete("/causal/cause/{edge_id}", dependencies=[Depends(verify_api_key)])
async def causal_remove_cause(edge_id: str):
    os = get_shadow_os()
    success = os.causal.remove_cause(edge_id)
    if not success:
        raise HTTPException(status_code=404, detail="Causal edge not found")
    return {"status": "removed", "edge_id": edge_id}


@router.get("/causal/causes/{variable_id}", dependencies=[Depends(verify_api_key)])
async def causal_discover_causes(variable_id: str, min_strength: float = Query(default=0.0, ge=0.0, le=1.0)):
    os = get_shadow_os()
    causes = os.causal.discover_causes(variable_id, min_strength=min_strength)
    return {"causes": [c.to_dict() for c in causes]}


@router.get("/causal/effects/{variable_id}", dependencies=[Depends(verify_api_key)])
async def causal_discover_effects(variable_id: str, min_strength: float = Query(default=0.0, ge=0.0, le=1.0)):
    os = get_shadow_os()
    effects = os.causal.discover_effects(variable_id, min_strength=min_strength)
    return {"effects": [e.to_dict() for e in effects]}


@router.post("/causal/intervene", dependencies=[Depends(verify_api_key)])
async def causal_intervene(request: CausalInterveneRequest):
    os = get_shadow_os()
    try:
        result = os.causal.intervene(request.variable_id, request.do_value, request.context)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}")


@router.post("/causal/counterfactual", dependencies=[Depends(verify_api_key)])
async def causal_counterfactual(request: CausalCounterfactualRequest):
    os = get_shadow_os()
    result = os.causal.counterfactual(request.premise, request.observed_outcome, request.alternative_outcome)
    return result.to_dict()


@router.get("/causal/confounders/{source_id}/{target_id}", dependencies=[Depends(verify_api_key)])
async def causal_find_confounders(source_id: str, target_id: str):
    os = get_shadow_os()
    confounders = os.causal.find_confounders(source_id, target_id)
    return {"confounders": confounders}


@router.get("/causal/mediators/{source_id}/{target_id}", dependencies=[Depends(verify_api_key)])
async def causal_find_mediators(source_id: str, target_id: str):
    os = get_shadow_os()
    mediators = os.causal.find_mediators(source_id, target_id)
    return {"mediators": mediators}


@router.get("/causal/paths/{source_id}/{target_id}", dependencies=[Depends(verify_api_key)])
async def causal_find_paths(source_id: str, target_id: str, max_depth: int = Query(default=5, ge=1, le=20)):
    os = get_shadow_os()
    paths = os.causal.find_causal_paths(source_id, target_id, max_depth=max_depth)
    return {"paths": [p.to_dict() for p in paths]}


@router.get("/causal/stats", dependencies=[Depends(verify_api_key)])
async def causal_stats():
    os = get_shadow_os()
    return os.causal.get_stats()


# ═════════════════════════════════════════════════════════════════════
# QUANTUM MEMORY FIELD ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class QuantumMemoryStoreRequest(BaseModel):
    content: str = Field(..., max_length=65536)
    domain: str = Field(default="", max_length=100)
    tags: Optional[List[str]] = None
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('tags', mode='before')
    @classmethod
    def validate_tags(cls, v):
        if v is not None:
            return [t[:_MAX_ELEMENT_LENGTH] for t in v]
        return v

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class QuantumMemoryObserveRequest(BaseModel):
    query: str = Field(..., max_length=5000)
    context: Optional[Dict[str, Any]] = None
    access_type: str = Field(default="observe", max_length=30)

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        return _validate_dict_keys(v)


class QuantumMemoryEntangleRequest(BaseModel):
    memory_id_1: str = Field(..., max_length=64)
    memory_id_2: str = Field(..., max_length=64)
    entanglement_type: str = Field(default="semantic", max_length=30)
    correlation: float = Field(default=0.5, ge=0.0, le=1.0)


class QuantumMemoryAssociateRequest(BaseModel):
    query: str = Field(..., max_length=5000)
    max_results: int = Field(default=10, ge=1, le=100)


class QuantumMemoryPredictRequest(BaseModel):
    context: str = Field(..., max_length=5000)
    domain: str = Field(default="", max_length=100)


class QuantumMemoryDecohereRequest(BaseModel):
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)


@router.post("/quantum-memory/store", dependencies=[Depends(verify_api_key)])
async def quantum_memory_store(request: QuantumMemoryStoreRequest):
    os = get_shadow_os()
    memory = os.quantum_memory.store(
        request.content, request.domain, request.tags, request.probability, request.metadata
    )
    return memory.to_dict()


@router.post("/quantum-memory/observe", dependencies=[Depends(verify_api_key)])
async def quantum_memory_observe(request: QuantumMemoryObserveRequest):
    os = get_shadow_os()
    try:
        atype = AccessType(request.access_type)
    except ValueError:
        atype = AccessType.OBSERVE
    result = os.quantum_memory.observe(request.query, request.context, atype)
    return result.to_dict()


@router.post("/quantum-memory/entangle", dependencies=[Depends(verify_api_key)])
async def quantum_memory_entangle(request: QuantumMemoryEntangleRequest):
    os = get_shadow_os()
    try:
        etype = EntanglementType(request.entanglement_type)
    except ValueError:
        etype = EntanglementType.SEMANTIC
    entanglement = os.quantum_memory.entangle(
        request.memory_id_1, request.memory_id_2, etype, request.correlation
    )
    if not entanglement:
        raise HTTPException(status_code=404, detail="One or both memories not found")
    return entanglement.to_dict()


@router.delete("/quantum-memory/entangle/{entanglement_id}", dependencies=[Depends(verify_api_key)])
async def quantum_memory_disentangle(entanglement_id: str):
    os = get_shadow_os()
    success = os.quantum_memory.disentangle(entanglement_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entanglement not found")
    return {"status": "disentangled", "entanglement_id": entanglement_id}


@router.get("/quantum-memory/recall/{memory_id}", dependencies=[Depends(verify_api_key)])
async def quantum_memory_recall(memory_id: str):
    os = get_shadow_os()
    memory = os.quantum_memory.recall(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory.to_dict()


@router.post("/quantum-memory/associate", dependencies=[Depends(verify_api_key)])
async def quantum_memory_associate(request: QuantumMemoryAssociateRequest):
    os = get_shadow_os()
    results = os.quantum_memory.associate(request.query, request.max_results)
    return {"associations": results, "count": len(results)}


@router.post("/quantum-memory/predict", dependencies=[Depends(verify_api_key)])
async def quantum_memory_predict(request: QuantumMemoryPredictRequest):
    os = get_shadow_os()
    result = os.quantum_memory.predict_next(request.context, request.domain)
    if not result:
        return {"prediction": None}
    return result


@router.post("/quantum-memory/decohere", dependencies=[Depends(verify_api_key)])
async def quantum_memory_decohere(request: QuantumMemoryDecohereRequest):
    os = get_shadow_os()
    removed = os.quantum_memory.decohere(request.decay_rate)
    return {"removed_count": removed}


@router.get("/quantum-memory/field-state", dependencies=[Depends(verify_api_key)])
async def quantum_memory_field_state():
    os = get_shadow_os()
    state = os.quantum_memory.get_field_state()
    return state.to_dict()


@router.get("/quantum-memory/entanglements/{memory_id}", dependencies=[Depends(verify_api_key)])
async def quantum_memory_entanglements(memory_id: str):
    os = get_shadow_os()
    entanglements = os.quantum_memory.get_entanglements(memory_id)
    return {"entanglements": [e.to_dict() for e in entanglements]}


@router.get("/quantum-memory/stats", dependencies=[Depends(verify_api_key)])
async def quantum_memory_stats():
    os = get_shadow_os()
    return os.quantum_memory.get_stats()


# ═════════════════════════════════════════════════════════════════════
# SEMANTIC DRIFT DETECTOR ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class DriftConceptRequest(BaseModel):
    concept: str = Field(..., max_length=200)
    definition: str = Field(..., max_length=5000)
    context: str = Field(default="", max_length=500)
    associations: Optional[List[str]] = None

    @field_validator('associations', mode='before')
    @classmethod
    def validate_associations(cls, v):
        if v is not None:
            return [a[:_MAX_ELEMENT_LENGTH] for a in v]
        return v


class DriftUsageRequest(BaseModel):
    concept: str = Field(..., max_length=200)
    context: str = Field(default="", max_length=500)
    definition_hint: str = Field(default="", max_length=2000)


class DriftCompareRequest(BaseModel):
    concept_a: str = Field(..., max_length=200)
    concept_b: str = Field(..., max_length=200)


class DriftMergeRequest(BaseModel):
    concept_a: str = Field(..., max_length=200)
    concept_b: str = Field(..., max_length=200)
    new_definition: str = Field(..., max_length=5000)


class DriftSplitRequest(BaseModel):
    concept: str = Field(..., max_length=200)
    definitions: List[str] = Field(..., min_length=2, max_length=10)

    @field_validator('definitions', mode='before')
    @classmethod
    def validate_definitions(cls, v):
        if v is not None:
            return [d[:2000] for d in v]
        return v


@router.post("/drift/concept", dependencies=[Depends(verify_api_key)])
async def drift_register_concept(request: DriftConceptRequest):
    os = get_shadow_os()
    snapshot = os.drift_detector.register_concept(
        request.concept, request.definition, request.context, request.associations
    )
    return snapshot.to_dict()


@router.post("/drift/usage", dependencies=[Depends(verify_api_key)])
async def drift_record_usage(request: DriftUsageRequest):
    os = get_shadow_os()
    snapshot = os.drift_detector.record_usage(
        request.concept, request.context, request.definition_hint
    )
    if not snapshot:
        raise HTTPException(status_code=404, detail="Concept not found")
    return snapshot.to_dict()


@router.post("/drift/detect/{concept}", dependencies=[Depends(verify_api_key)])
async def drift_detect(concept: str, min_magnitude: float = Query(default=0.1, ge=0.0, le=1.0)):
    os = get_shadow_os()
    event = os.drift_detector.detect_drift(concept, min_magnitude=min_magnitude)
    if not event:
        return {"drift": None, "status": "no_drift_detected"}
    return event.to_dict()


@router.get("/drift/lineage/{concept}", dependencies=[Depends(verify_api_key)])
async def drift_lineage(concept: str):
    os = get_shadow_os()
    lineage = os.drift_detector.get_lineage(concept)
    if not lineage:
        raise HTTPException(status_code=404, detail="Concept not found")
    return lineage.to_dict()


@router.post("/drift/compare", dependencies=[Depends(verify_api_key)])
async def drift_compare(request: DriftCompareRequest):
    os = get_shadow_os()
    result = os.drift_detector.compare_concepts(request.concept_a, request.concept_b)
    return result


@router.get("/drift/report", dependencies=[Depends(verify_api_key)])
async def drift_report():
    os = get_shadow_os()
    report = os.drift_detector.get_drift_report()
    return report.to_dict()


@router.post("/drift/merge", dependencies=[Depends(verify_api_key)])
async def drift_merge(request: DriftMergeRequest):
    os = get_shadow_os()
    snapshot = os.drift_detector.merge_concepts(
        request.concept_a, request.concept_b, request.new_definition
    )
    if not snapshot:
        raise HTTPException(status_code=404, detail="One or both concepts not found")
    return snapshot.to_dict()


@router.post("/drift/split", dependencies=[Depends(verify_api_key)])
async def drift_split(request: DriftSplitRequest):
    os = get_shadow_os()
    snapshots = os.drift_detector.split_concept(request.concept, request.definitions)
    return {"snapshots": [s.to_dict() for s in snapshots]}


@router.get("/drift/concepts", dependencies=[Depends(verify_api_key)])
async def drift_list_concepts(status: Optional[str] = Query(default=None, max_length=30)):
    os = get_shadow_os()
    status_enum = None
    if status:
        try:
            status_enum = ConceptStatus(status)
        except ValueError:
            pass
    concepts = os.drift_detector.list_concepts(status=status_enum)
    return {"concepts": concepts}


@router.get("/drift/stats", dependencies=[Depends(verify_api_key)])
async def drift_stats():
    os = get_shadow_os()
    return os.drift_detector.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE IMMUNE SYSTEM ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ImmuneDetectRequest(BaseModel):
    content: str = Field(..., max_length=10000)
    source: str = Field(default="", max_length=200)
    context: Optional[Dict[str, Any]] = None

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        return _validate_dict_keys(v)


class ImmuneAntibodyRequest(BaseModel):
    pathogen_id: str = Field(..., max_length=64)
    counter_argument: str = Field(..., max_length=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ImmuneInoculateRequest(BaseModel):
    pathogen_type: str = Field(..., max_length=30)
    weak_content: str = Field(..., max_length=10000)


class ImmuneCheckRequest(BaseModel):
    content: str = Field(..., max_length=10000)
    pathogen_type: str = Field(..., max_length=30)


class ImmuneEvaluateRequest(BaseModel):
    content: str = Field(..., max_length=10000)


@router.post("/immune/detect", dependencies=[Depends(verify_api_key)])
async def immune_detect(request: ImmuneDetectRequest):
    os = get_shadow_os()
    pathogen = os.immune.detect_pathogen(request.content, request.source, request.context)
    if not pathogen:
        return {"pathogen": None, "status": "clean"}
    return pathogen.to_dict()


@router.post("/immune/antibody", dependencies=[Depends(verify_api_key)])
async def immune_generate_antibody(request: ImmuneAntibodyRequest):
    os = get_shadow_os()
    antibody = os.immune.generate_antibody(
        request.pathogen_id, request.counter_argument, request.confidence
    )
    if not antibody:
        raise HTTPException(status_code=404, detail="Pathogen not found")
    return antibody.to_dict()


@router.post("/immune/neutralize/{pathogen_id}", dependencies=[Depends(verify_api_key)])
async def immune_neutralize(pathogen_id: str):
    os = get_shadow_os()
    response = os.immune.neutralize(pathogen_id)
    if not response:
        raise HTTPException(status_code=404, detail="Pathogen not found or already neutralized")
    return response.to_dict()


@router.post("/immune/quarantine/{pathogen_id}", dependencies=[Depends(verify_api_key)])
async def immune_quarantine(pathogen_id: str):
    os = get_shadow_os()
    response = os.immune.quarantine(pathogen_id)
    if not response:
        raise HTTPException(status_code=404, detail="Pathogen not found or already quarantined")
    return response.to_dict()


@router.post("/immune/inoculate", dependencies=[Depends(verify_api_key)])
async def immune_inoculate(request: ImmuneInoculateRequest):
    os = get_shadow_os()
    try:
        ptype = PathogenType(request.pathogen_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid pathogen_type: {request.pathogen_type}")
    cell = os.immune.inoculate(ptype, request.weak_content)
    return cell.to_dict()


@router.post("/immune/check-immunity", dependencies=[Depends(verify_api_key)])
async def immune_check_immunity(request: ImmuneCheckRequest):
    os = get_shadow_os()
    try:
        ptype = PathogenType(request.pathogen_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid pathogen_type: {request.pathogen_type}")
    score = os.immune.check_immunity(request.content, ptype)
    return {"immunity_score": score, "pathogen_type": ptype.value}


@router.get("/immune/pathogens", dependencies=[Depends(verify_api_key)])
async def immune_list_pathogens(threat_level: Optional[int] = Query(default=None, ge=1, le=5)):
    os = get_shadow_os()
    pathogens = os.immune.get_pathogens(threat_level=threat_level)
    return {"pathogens": [p.to_dict() for p in pathogens]}


@router.get("/immune/antibodies/{pathogen_id}", dependencies=[Depends(verify_api_key)])
async def immune_list_antibodies(pathogen_id: str):
    os = get_shadow_os()
    antibodies = os.immune.get_antibodies(pathogen_id)
    return {"antibodies": [a.to_dict() for a in antibodies]}


@router.get("/immune/memory", dependencies=[Depends(verify_api_key)])
async def immune_memory():
    os = get_shadow_os()
    cells = os.immune.get_immune_memory()
    return {"memory_cells": [c.to_dict() for c in cells]}


@router.post("/immune/evaluate", dependencies=[Depends(verify_api_key)])
async def immune_evaluate(request: ImmuneEvaluateRequest):
    os = get_shadow_os()
    result = os.immune.evaluate_threat(request.content)
    return result


@router.get("/immune/stats", dependencies=[Depends(verify_api_key)])
async def immune_stats():
    os = get_shadow_os()
    return os.immune.get_stats()


# ═════════════════════════════════════════════════════════════════════
# TEMPORAL PARALLAX ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

class ParallaxEventRequest(BaseModel):
    event_type: str = Field(..., max_length=100)
    description: str = Field(..., max_length=5000)
    domain: str = Field(default="", max_length=100)
    raw_impact: float = Field(default=0.5, ge=0.0, le=1.0)


class ParallaxCompareRequest(BaseModel):
    event_id: str = Field(..., max_length=64)
    distance_a: int = Field(..., ge=1, le=3650)
    distance_b: int = Field(..., ge=1, le=3650)


@router.post("/parallax/event", dependencies=[Depends(verify_api_key)])
async def parallax_register_event(request: ParallaxEventRequest):
    os = get_shadow_os()
    event = os.parallax.register_event(
        request.event_type, request.description, request.domain, request.raw_impact
    )
    return event.to_dict()


@router.post("/parallax/compute/{event_id}", dependencies=[Depends(verify_api_key)])
async def parallax_compute(event_id: str):
    os = get_shadow_os()
    measurement = os.parallax.compute_parallax(event_id)
    if not measurement:
        raise HTTPException(status_code=404, detail="Event not found")
    return measurement.to_dict()


@router.post("/parallax/compare", dependencies=[Depends(verify_api_key)])
async def parallax_compare(request: ParallaxCompareRequest):
    os = get_shadow_os()
    result = os.parallax.compare_perspectives(
        request.event_id, request.distance_a, request.distance_b
    )
    if not result:
        raise HTTPException(status_code=404, detail="Event not found")
    return result


@router.get("/parallax/shifts", dependencies=[Depends(verify_api_key)])
async def parallax_shifts(min_parallax_score: float = Query(default=0.2, ge=0.0, le=1.0)):
    os = get_shadow_os()
    shifts = os.parallax.identify_shifts(min_parallax_score=min_parallax_score)
    return {"shifts": shifts}


@router.get("/parallax/dominant/{event_id}", dependencies=[Depends(verify_api_key)])
async def parallax_dominant(event_id: str):
    os = get_shadow_os()
    perspective = os.parallax.get_dominant_perspective(event_id)
    if not perspective:
        raise HTTPException(status_code=404, detail="Event not found")
    return perspective


@router.get("/parallax/emergent", dependencies=[Depends(verify_api_key)])
async def parallax_emergent(min_impact: float = Query(default=0.5, ge=0.0, le=1.0)):
    os = get_shadow_os()
    events = os.parallax.find_emergent_significance(min_impact=min_impact)
    return {"events": events}


@router.get("/parallax/diminishing", dependencies=[Depends(verify_api_key)])
async def parallax_diminishing(min_impact: float = Query(default=0.5, ge=0.0, le=1.0)):
    os = get_shadow_os()
    events = os.parallax.find_diminishing_events(min_impact=min_impact)
    return {"events": events}


@router.get("/parallax/event/{event_id}", dependencies=[Depends(verify_api_key)])
async def parallax_get_event(event_id: str):
    os = get_shadow_os()
    event = os.parallax.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.get("/parallax/events", dependencies=[Depends(verify_api_key)])
async def parallax_list_events(domain: str = Query(default="", max_length=100), limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    events = os.parallax.list_events(domain=domain, limit=limit)
    return {"events": events}


@router.get("/parallax/stats", dependencies=[Depends(verify_api_key)])
async def parallax_stats():
    os = get_shadow_os()
    return os.parallax.get_stats()


# ═════════════════════════════════════════════════════════════════════
# ADAPTIVE BOUNDARY ENDPOINTS (Sprint 227)
# ═════════════════════════════════════════════════════════════════════

from .adaptive_boundary import AdaptiveBoundary, BoundaryType


class BoundaryRuleRequest(BaseModel):
    domain_a: str = Field(..., max_length=100)
    domain_b: str = Field(..., max_length=100)
    boundary_type: str = Field(default="flexible", max_length=30)
    permeability: float = Field(default=0.3, ge=0.0, le=1.0)
    energy_cost: float = Field(default=0.1, ge=0.0, le=10.0)


class BoundaryCheckRequest(BaseModel):
    source_domain: str = Field(..., max_length=100)
    target_domain: str = Field(..., max_length=100)
    content: str = Field(default="", max_length=5000)


class BoundaryAllocateRequest(BaseModel):
    domain: str = Field(..., max_length=100)
    amount: float = Field(..., ge=0.0)


class BoundarySpendRequest(BaseModel):
    domain: str = Field(..., max_length=100)
    amount: float = Field(..., ge=0.0)


class BoundaryViolationRequest(BaseModel):
    source_domain: str = Field(..., max_length=100)
    target_domain: str = Field(..., max_length=100)
    content: str = Field(default="", max_length=5000)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/boundary/rule", dependencies=[Depends(verify_api_key)])
async def boundary_set_rule(request: BoundaryRuleRequest):
    os = get_shadow_os()
    try:
        btype = BoundaryType(request.boundary_type)
    except ValueError:
        btype = BoundaryType.FLEXIBLE
    rule = os.boundary.set_boundary(
        request.domain_a, request.domain_b, btype.value,
        request.permeability, request.energy_cost,
    )
    return rule.to_dict()


@router.delete("/boundary/rule/{rule_id}", dependencies=[Depends(verify_api_key)])
async def boundary_remove_rule(rule_id: str):
    os = get_shadow_os()
    success = os.boundary.remove_boundary(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"status": "removed", "rule_id": rule_id}


@router.post("/boundary/check", dependencies=[Depends(verify_api_key)])
async def boundary_check(request: BoundaryCheckRequest):
    os = get_shadow_os()
    result = os.boundary.check_boundary(request.source_domain, request.target_domain, request.content)
    return result


@router.post("/boundary/allocate", dependencies=[Depends(verify_api_key)])
async def boundary_allocate(request: BoundaryAllocateRequest):
    os = get_shadow_os()
    budget = os.boundary.allocate_energy(request.domain, request.amount)
    if not budget:
        raise HTTPException(status_code=400, detail="Allocation failed")
    return budget.to_dict()


@router.post("/boundary/spend", dependencies=[Depends(verify_api_key)])
async def boundary_spend(request: BoundarySpendRequest):
    os = get_shadow_os()
    budget = os.boundary.spend_energy(request.domain, request.amount)
    if not budget:
        raise HTTPException(status_code=400, detail="Spend failed (insufficient energy or domain not found)")
    return budget.to_dict()


@router.get("/boundary/profile", dependencies=[Depends(verify_api_key)])
async def boundary_profile():
    os = get_shadow_os()
    profile = os.boundary.get_energy_profile()
    return profile.to_dict()


@router.post("/boundary/violation", dependencies=[Depends(verify_api_key)])
async def boundary_report_violation(request: BoundaryViolationRequest):
    os = get_shadow_os()
    violation = os.boundary.report_violation(
        request.source_domain, request.target_domain, request.content, request.severity,
    )
    return violation.to_dict()


@router.get("/boundary/violations", dependencies=[Depends(verify_api_key)])
async def boundary_violations(
    domain: str = Query(default="", max_length=100),
    limit: int = Query(default=50, ge=1, le=500),
):
    os = get_shadow_os()
    violations = os.boundary.get_violations(domain=domain, limit=limit)
    return {"violations": violations, "count": len(violations)}


@router.get("/boundary/recovery", dependencies=[Depends(verify_api_key)])
async def boundary_recovery():
    os = get_shadow_os()
    return os.boundary.suggest_recovery()


@router.get("/boundary/rules/{domain}", dependencies=[Depends(verify_api_key)])
async def boundary_rules(domain: str):
    os = get_shadow_os()
    rules = os.boundary.get_boundary_rules(domain=domain)
    return {"rules": rules, "count": len(rules)}


@router.get("/boundary/stats", dependencies=[Depends(verify_api_key)])
async def boundary_stats():
    os = get_shadow_os()
    return os.boundary.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE RESONANCE ENGINE ENDPOINTS (Sprint 228)
# ═════════════════════════════════════════════════════════════════════

from .cognitive_resonance_engine import CognitiveResonanceEngine


class ResonancePatternRequest(BaseModel):
    domain: str = Field(..., max_length=100)
    frequency: float = Field(..., ge=0.0)
    amplitude: float = Field(default=1.0, ge=0.0)
    phase: float = Field(default=0.0, ge=0.0, le=6.283185307)
    content: str = Field(default="", max_length=5000)


class ResonanceDetectRequest(BaseModel):
    min_amplitude: float = Field(default=0.3, ge=0.0, le=1.0)
    min_coherence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResonanceHarmonicsRequest(BaseModel):
    domain: str = Field(..., max_length=100)


class ResonanceAmplifyRequest(BaseModel):
    pattern_id: str = Field(..., max_length=64)
    factor: float = Field(default=1.5, ge=0.01, le=10.0)


class ResonanceDampenRequest(BaseModel):
    pattern_id: str = Field(..., max_length=64)
    factor: float = Field(default=0.5, ge=0.01, le=1.0)


class ResonanceSynthesizeRequest(BaseModel):
    pattern_ids: List[str] = Field(..., max_length=20)

    @field_validator('pattern_ids', mode='before')
    @classmethod
    def validate_pattern_ids(cls, v):
        if v is not None:
            return [p[:64] for p in v]
        return v


@router.post("/resonance-engine/pattern", dependencies=[Depends(verify_api_key)])
async def resonance_emit_pattern(request: ResonancePatternRequest):
    os = get_shadow_os()
    pattern = os.resonance_engine.emit_pattern(
        request.domain, request.frequency, request.amplitude, request.phase, request.content,
    )
    return pattern.to_dict()


@router.post("/resonance-engine/detect", dependencies=[Depends(verify_api_key)])
async def resonance_detect(request: ResonanceDetectRequest):
    os = get_shadow_os()
    events = os.resonance_engine.detect_resonance(
        min_amplitude=request.min_amplitude, min_coherence=request.min_coherence,
    )
    return {"events": [e.to_dict() for e in events], "count": len(events)}


@router.post("/resonance-engine/harmonics", dependencies=[Depends(verify_api_key)])
async def resonance_harmonics(request: ResonanceHarmonicsRequest):
    os = get_shadow_os()
    spectrum = os.resonance_engine.compute_harmonics(request.domain)
    return spectrum.to_dict()


@router.get("/resonance-engine/constructive", dependencies=[Depends(verify_api_key)])
async def resonance_constructive(min_coherence: float = Query(default=0.6, ge=0.0, le=1.0)):
    os = get_shadow_os()
    events = os.resonance_engine.find_constructive(min_coherence=min_coherence)
    return {"events": [e.to_dict() for e in events], "count": len(events)}


@router.get("/resonance-engine/destructive", dependencies=[Depends(verify_api_key)])
async def resonance_destructive():
    os = get_shadow_os()
    events = os.resonance_engine.find_destructive()
    return {"events": [e.to_dict() for e in events], "count": len(events)}


@router.post("/resonance-engine/amplify", dependencies=[Depends(verify_api_key)])
async def resonance_amplify(request: ResonanceAmplifyRequest):
    os = get_shadow_os()
    pattern = os.resonance_engine.amplify(request.pattern_id, request.factor)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern.to_dict()


@router.post("/resonance-engine/dampen", dependencies=[Depends(verify_api_key)])
async def resonance_dampen(request: ResonanceDampenRequest):
    os = get_shadow_os()
    pattern = os.resonance_engine.dampen(request.pattern_id, request.factor)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern.to_dict()


@router.post("/resonance-engine/synthesize", dependencies=[Depends(verify_api_key)])
async def resonance_synthesize(request: ResonanceSynthesizeRequest):
    os = get_shadow_os()
    insight = os.resonance_engine.synthesize_insight(request.pattern_ids)
    if not insight:
        raise HTTPException(status_code=400, detail="Synthesis failed — check pattern IDs")
    return insight.to_dict()


@router.get("/resonance-engine/events", dependencies=[Depends(verify_api_key)])
async def resonance_events(
    domain: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=500),
):
    os = get_shadow_os()
    events = os.resonance_engine.get_resonance_events(domain=domain, limit=limit)
    return {"events": [e.to_dict() for e in events], "count": len(events)}


@router.get("/resonance-engine/insights", dependencies=[Depends(verify_api_key)])
async def resonance_insights(
    domain: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=500),
):
    os = get_shadow_os()
    insights = os.resonance_engine.get_insights(domain=domain, limit=limit)
    return {"insights": [i.to_dict() for i in insights], "count": len(insights)}


@router.get("/resonance-engine/stats", dependencies=[Depends(verify_api_key)])
async def resonance_engine_stats():
    os = get_shadow_os()
    return os.resonance_engine.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE ECONOMY ENDPOINTS (Sprint 229)
# ═════════════════════════════════════════════════════════════════════

from .cognitive_economy import CognitiveEconomy


class EconomyBudgetRequest(BaseModel):
    resource_type: str = Field(..., max_length=100)
    total_budget: float = Field(..., ge=0.0)


class EconomyAllocationRequest(BaseModel):
    activity: str = Field(..., max_length=200)
    resource_type: str = Field(..., max_length=100)
    amount: float = Field(..., ge=0.0)
    priority: int = Field(default=2, ge=0, le=10)
    justification: str = Field(default="", max_length=1000)


class EconomySpendRequest(BaseModel):
    amount: float = Field(..., ge=0.0)


class EconomyOpportunityCostRequest(BaseModel):
    chosen_activity: str = Field(..., max_length=200)
    foregone_activity: str = Field(..., max_length=200)
    resource_type: str = Field(..., max_length=100)


@router.post("/economy/budget", dependencies=[Depends(verify_api_key)])
async def economy_set_budget(request: EconomyBudgetRequest):
    os = get_shadow_os()
    resource = os.economy.set_budget(request.resource_type, request.total_budget)
    return resource.to_dict()


@router.post("/economy/request", dependencies=[Depends(verify_api_key)])
async def economy_request_allocation(request: EconomyAllocationRequest):
    os = get_shadow_os()
    alloc = os.economy.request_allocation(
        request.activity, request.resource_type, request.amount,
        request.priority, request.justification,
    )
    return alloc.to_dict()


@router.post("/economy/approve/{request_id}", dependencies=[Depends(verify_api_key)])
async def economy_approve(request_id: str):
    os = get_shadow_os()
    alloc = os.economy.approve_allocation(request_id)
    if not alloc:
        raise HTTPException(status_code=404, detail="Request not found or already approved")
    return alloc.to_dict()


@router.post("/economy/spend/{request_id}", dependencies=[Depends(verify_api_key)])
async def economy_spend(request_id: str, request: EconomySpendRequest):
    os = get_shadow_os()
    alloc = os.economy.spend_allocation(request_id, request.amount)
    if not alloc:
        raise HTTPException(status_code=404, detail="Allocation not found or insufficient funds")
    return alloc.to_dict()


@router.delete("/economy/release/{request_id}", dependencies=[Depends(verify_api_key)])
async def economy_release(request_id: str):
    os = get_shadow_os()
    alloc = os.economy.release_allocation(request_id)
    if not alloc:
        raise HTTPException(status_code=404, detail="Allocation not found")
    return alloc.to_dict()


@router.post("/economy/opportunity-cost", dependencies=[Depends(verify_api_key)])
async def economy_opportunity_cost(request: EconomyOpportunityCostRequest):
    os = get_shadow_os()
    cost = os.economy.calculate_opportunity_cost(
        request.chosen_activity, request.foregone_activity, request.resource_type,
    )
    return cost.to_dict()


@router.get("/economy/budget/{resource_type}", dependencies=[Depends(verify_api_key)])
async def economy_get_budget(resource_type: str):
    os = get_shadow_os()
    budget = os.economy.get_budget(resource_type)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    return budget.to_dict()


@router.get("/economy/allocations", dependencies=[Depends(verify_api_key)])
async def economy_allocations(
    status: Optional[str] = Query(default=None, max_length=30),
    resource_type: Optional[str] = Query(default=None, max_length=100),
):
    os = get_shadow_os()
    allocations = os.economy.get_allocations(status=status, resource_type=resource_type)
    return {"allocations": allocations, "count": len(allocations)}


@router.get("/economy/opportunity-costs", dependencies=[Depends(verify_api_key)])
async def economy_opportunity_costs(
    resource_type: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=100, ge=1, le=500),
):
    os = get_shadow_os()
    costs = os.economy.get_opportunity_costs(resource_type=resource_type, limit=limit)
    return {"opportunity_costs": costs, "count": len(costs)}


@router.get("/economy/report", dependencies=[Depends(verify_api_key)])
async def economy_report():
    os = get_shadow_os()
    report = os.economy.generate_report()
    return report.to_dict()


@router.get("/economy/rebalancing", dependencies=[Depends(verify_api_key)])
async def economy_rebalancing():
    os = get_shadow_os()
    result = os.economy.optimize_allocation()
    return result


@router.get("/economy/stats", dependencies=[Depends(verify_api_key)])
async def economy_stats():
    os = get_shadow_os()
    return os.economy.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE TOPOLOGY ENDPOINTS (Sprint 230)
# ═════════════════════════════════════════════════════════════════════

from .cognitive_topology import CognitiveTopology, NodeType, EdgeType


class TopologyNodeRequest(BaseModel):
    name: str = Field(..., max_length=200)
    node_type: str = Field(default="concept", max_length=30)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


class TopologyEdgeRequest(BaseModel):
    source_id: str = Field(..., max_length=64)
    target_id: str = Field(..., max_length=64)
    edge_type: str = Field(default="enhances", max_length=30)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        return _validate_dict_keys(v)


@router.post("/topology/node", dependencies=[Depends(verify_api_key)])
async def topology_add_node(request: TopologyNodeRequest):
    os = get_shadow_os()
    try:
        ntype = NodeType(request.node_type)
    except ValueError:
        ntype = NodeType.CONCEPT
    node = os.topology.add_node(request.name, ntype, request.metadata)
    return node.to_dict()


@router.post("/topology/edge", dependencies=[Depends(verify_api_key)])
async def topology_add_edge(request: TopologyEdgeRequest):
    os = get_shadow_os()
    try:
        etype = EdgeType(request.edge_type)
    except ValueError:
        etype = EdgeType.ENHANCES
    edge = os.topology.add_edge(
        request.source_id, request.target_id, etype, request.weight, request.metadata,
    )
    if not edge:
        raise HTTPException(status_code=400, detail="Source or target node not found")
    return edge.to_dict()


@router.delete("/topology/node/{node_id}", dependencies=[Depends(verify_api_key)])
async def topology_remove_node(node_id: str):
    os = get_shadow_os()
    success = os.topology.remove_node(node_id)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"status": "removed", "node_id": node_id}


@router.delete("/topology/edge/{edge_id}", dependencies=[Depends(verify_api_key)])
async def topology_remove_edge(edge_id: str):
    os = get_shadow_os()
    success = os.topology.remove_edge(edge_id)
    if not success:
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"status": "removed", "edge_id": edge_id}


@router.post("/topology/analyze", dependencies=[Depends(verify_api_key)])
async def topology_analyze():
    os = get_shadow_os()
    analysis = os.topology.analyze_topology()
    return analysis.to_dict()


@router.get("/topology/clusters", dependencies=[Depends(verify_api_key)])
async def topology_clusters(min_density: float = Query(default=0.3, ge=0.0, le=1.0)):
    os = get_shadow_os()
    clusters = os.topology.find_clusters(min_density=min_density)
    return {"clusters": [c.to_dict() for c in clusters], "count": len(clusters)}


@router.get("/topology/bridges", dependencies=[Depends(verify_api_key)])
async def topology_bridges():
    os = get_shadow_os()
    bridges = os.topology.find_bridges()
    return {"bridges": bridges, "count": len(bridges)}


@router.get("/topology/bottlenecks", dependencies=[Depends(verify_api_key)])
async def topology_bottlenecks():
    os = get_shadow_os()
    bottlenecks = os.topology.find_bottlenecks()
    return {"bottlenecks": bottlenecks, "count": len(bottlenecks)}


@router.get("/topology/isolated", dependencies=[Depends(verify_api_key)])
async def topology_isolated():
    os = get_shadow_os()
    isolated = os.topology.find_isolated_nodes()
    return {"isolated_nodes": isolated, "count": len(isolated)}


@router.get("/topology/suggestions", dependencies=[Depends(verify_api_key)])
async def topology_suggestions(max_suggestions: int = Query(default=20, ge=1, le=100)):
    os = get_shadow_os()
    suggestions = os.topology.suggest_connections(max_suggestions=max_suggestions)
    return {"suggestions": suggestions, "count": len(suggestions)}


@router.get("/topology/path/{source_id}/{target_id}", dependencies=[Depends(verify_api_key)])
async def topology_path(source_id: str, target_id: str):
    os = get_shadow_os()
    path = os.topology.get_metrics()
    return {"source_id": source_id, "target_id": target_id, "topology_metrics": path}


@router.get("/topology/metrics", dependencies=[Depends(verify_api_key)])
async def topology_metrics():
    os = get_shadow_os()
    return os.topology.get_metrics()


@router.get("/topology/stats", dependencies=[Depends(verify_api_key)])
async def topology_stats():
    os = get_shadow_os()
    return os.topology.get_stats()


# ═════════════════════════════════════════════════════════════════════
# COGNITIVE SYMBIOSIS ENDPOINTS (Sprint 231)
# ═════════════════════════════════════════════════════════════════════

from .cognitive_symbiosis import CognitiveSymbiosis, SymbiosisType, InteractionMode


class SymbiosisPairRequest(BaseModel):
    module_a: str = Field(..., max_length=100)
    module_b: str = Field(..., max_length=100)
    symbiosis_type: str = Field(..., max_length=30)
    interaction_mode: str = Field(..., max_length=30)
    strength_a_to_b: float = Field(default=0.5, ge=0.0, le=1.0)
    strength_b_to_a: float = Field(default=0.5, ge=0.0, le=1.0)


class SymbiosisInteractionRequest(BaseModel):
    pair_id: str = Field(..., max_length=64)
    direction: str = Field(..., max_length=30)
    content: str = Field(..., max_length=5000)
    outcome: str = Field(..., max_length=500)
    benefit_a: float = Field(default=0.5, ge=0.0, le=1.0)
    benefit_b: float = Field(default=0.5, ge=0.0, le=1.0)


class SymbiosisDetectParasitismRequest(BaseModel):
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class SymbiosisConvertRequest(BaseModel):
    pair_id: str = Field(..., max_length=64)
    new_type: str = Field(..., max_length=30)


@router.post("/symbiosis/pair", dependencies=[Depends(verify_api_key)])
async def symbiosis_register_pair(request: SymbiosisPairRequest):
    os = get_shadow_os()
    try:
        stype = SymbiosisType(request.symbiosis_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbiosis_type: {request.symbiosis_type}")
    try:
        imode = InteractionMode(request.interaction_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid interaction_mode: {request.interaction_mode}")
    pair = os.symbiosis.register_pair(
        request.module_a, request.module_b, stype, imode,
        request.strength_a_to_b, request.strength_b_to_a,
    )
    return pair.to_dict()


@router.delete("/symbiosis/pair/{pair_id}", dependencies=[Depends(verify_api_key)])
async def symbiosis_remove_pair(pair_id: str):
    os = get_shadow_os()
    success = os.symbiosis.remove_pair(pair_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pair not found")
    return {"status": "removed", "pair_id": pair_id}


@router.post("/symbiosis/interaction", dependencies=[Depends(verify_api_key)])
async def symbiosis_record_interaction(request: SymbiosisInteractionRequest):
    os = get_shadow_os()
    interaction = os.symbiosis.record_interaction(
        request.pair_id, request.direction, request.content,
        request.outcome, request.benefit_a, request.benefit_b,
    )
    if not interaction:
        raise HTTPException(status_code=404, detail="Pair not found")
    return interaction.to_dict()


@router.post("/symbiosis/detect-parasitism", dependencies=[Depends(verify_api_key)])
async def symbiosis_detect_parasitism(request: SymbiosisDetectParasitismRequest):
    os = get_shadow_os()
    detections = os.symbiosis.detect_parasitism(min_confidence=request.min_confidence)
    return {"detections": [d.to_dict() for d in detections], "count": len(detections)}


@router.post("/symbiosis/convert", dependencies=[Depends(verify_api_key)])
async def symbiosis_convert(request: SymbiosisConvertRequest):
    os = get_shadow_os()
    try:
        ntype = SymbiosisType(request.new_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid new_type: {request.new_type}")
    pair = os.symbiosis.convert_pair(request.pair_id, ntype)
    if not pair:
        raise HTTPException(status_code=404, detail="Pair not found")
    return pair.to_dict()


@router.post("/symbiosis/optimize", dependencies=[Depends(verify_api_key)])
async def symbiosis_optimize():
    os = get_shadow_os()
    suggestions = os.symbiosis.optimize_mutualism()
    return {"suggestions": suggestions, "count": len(suggestions)}


@router.get("/symbiosis/pair/{pair_id}", dependencies=[Depends(verify_api_key)])
async def symbiosis_get_pair(pair_id: str):
    os = get_shadow_os()
    pair = os.symbiosis.get_pair(pair_id)
    if not pair:
        raise HTTPException(status_code=404, detail="Pair not found")
    return pair.to_dict()


@router.get("/symbiosis/pairs", dependencies=[Depends(verify_api_key)])
async def symbiosis_list_pairs(
    module: str = Query(default="", max_length=100),
    symbiosis_type: Optional[str] = Query(default=None, max_length=30),
):
    os = get_shadow_os()
    stype = None
    if symbiosis_type:
        try:
            stype = SymbiosisType(symbiosis_type)
        except ValueError:
            pass
    pairs = os.symbiosis.get_pairs(module=module, symbiosis_type=stype)
    return {"pairs": [p.to_dict() for p in pairs], "count": len(pairs)}


@router.get("/symbiosis/interactions/{pair_id}", dependencies=[Depends(verify_api_key)])
async def symbiosis_interactions(
    pair_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    os = get_shadow_os()
    interactions = os.symbiosis.get_interactions(pair_id, limit=limit)
    return {"interactions": [i.to_dict() for i in interactions], "count": len(interactions)}


@router.get("/symbiosis/report", dependencies=[Depends(verify_api_key)])
async def symbiosis_report():
    os = get_shadow_os()
    report = os.symbiosis.generate_report()
    return report.to_dict()


@router.get("/symbiosis/benefit", dependencies=[Depends(verify_api_key)])
async def symbiosis_benefit():
    os = get_shadow_os()
    score = os.symbiosis.get_system_benefit()
    return {"system_benefit_score": score}


@router.get("/symbiosis/suggestions", dependencies=[Depends(verify_api_key)])
async def symbiosis_suggestions():
    os = get_shadow_os()
    suggestions = os.symbiosis.suggest_optimizations()
    return {"suggestions": suggestions, "count": len(suggestions)}


@router.get("/symbiosis/stats", dependencies=[Depends(verify_api_key)])
async def symbiosis_stats():
    os = get_shadow_os()
    return os.symbiosis.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard compatibility aliases
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/briefing/generate", dependencies=[Depends(verify_api_key)])
async def briefing_generate():
    os = get_shadow_os()
    briefing = await os.briefing.generate_briefing(
        device_context=os.device_context,
        evolution_guide=getattr(os, "evolution", None),
    )
    return briefing.to_dict()


@router.get("/arise/list", dependencies=[Depends(verify_api_key)])
async def arise_list(limit: int = Query(default=20, ge=1, le=200)):
    os = get_shadow_os()
    agents = os.arise.list_agents(limit=limit)
    return {"agents": agents}


@router.get("/device/summary", dependencies=[Depends(verify_api_key)])
async def device_summary_alias():
    os = get_shadow_os()
    return os.device_context.get_device_summary()


# === ShadowPowers Endpoints ===


class PowerApplyRequest(BaseModel):
    power_name: str = Field(..., max_length=100)
    context: str = Field(..., max_length=5000)
    result: str = Field(default="", max_length=5000)
    success: bool = Field(default=True)


class PowerAwakenRequest(BaseModel):
    context: str = Field(..., max_length=5000)


class PowerSynthesizeRequest(BaseModel):
    power_names: list = Field(..., max_length=5)
    context: str = Field(default="", max_length=5000)


@router.post("/powers/apply", dependencies=[Depends(verify_api_key)])
async def powers_apply(req: PowerApplyRequest):
    os = get_shadow_os()
    app = os.shadow_powers.apply_power(
        power_name=req.power_name,
        context=req.context,
        result=req.result,
        success=req.success,
    )
    if not app:
        raise HTTPException(status_code=404, detail="Power not found")
    return app.to_dict()


@router.post("/powers/recover", dependencies=[Depends(verify_api_key)])
async def powers_recover():
    os = get_shadow_os()
    os.shadow_powers.recover_fatigue()
    return {"recovered": True}


@router.post("/powers/awaken", dependencies=[Depends(verify_api_key)])
async def powers_awaken(req: PowerAwakenRequest):
    os = get_shadow_os()
    awakened = os.shadow_powers.awaken_dormant(context=req.context)
    return {"awakened": awakened}


@router.post("/powers/synthesize", dependencies=[Depends(verify_api_key)])
async def powers_synthesize(req: PowerSynthesizeRequest):
    os = get_shadow_os()
    result = os.shadow_powers.synthesize_powers(
        power_names=req.power_names,
        context=req.context,
    )
    return result.to_dict() if result else {"synthesized": None}


@router.get("/powers/{name}", dependencies=[Depends(verify_api_key)])
async def powers_get(name: str):
    os = get_shadow_os()
    power = os.shadow_powers.get_power(name)
    if not power:
        raise HTTPException(status_code=404, detail="Power not found")
    return power


@router.get("/powers", dependencies=[Depends(verify_api_key)])
async def powers_list(category: Optional[str] = Query(default=None), state: Optional[str] = Query(default=None)):
    os = get_shadow_os()
    powers = os.shadow_powers.list_powers(category=category, state=state)
    return {"powers": powers}


@router.get("/powers/tree", dependencies=[Depends(verify_api_key)])
async def powers_tree():
    os = get_shadow_os()
    return os.shadow_powers.get_power_tree()


@router.get("/powers/applications", dependencies=[Depends(verify_api_key)])
async def powers_applications(power_name: Optional[str] = Query(default=None), limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    apps = os.shadow_powers.get_applications(power_name=power_name, limit=limit)
    return {"applications": [a.to_dict() for a in apps]}


@router.get("/powers/evolution-history", dependencies=[Depends(verify_api_key)])
async def powers_evolution_history(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"history": os.shadow_powers.get_evolution_history(limit=limit)}


@router.get("/powers/recommendations", dependencies=[Depends(verify_api_key)])
async def powers_recommendations():
    os = get_shadow_os()
    recs = os.shadow_powers.get_recommendations()
    return {"recommendations": recs}


@router.get("/powers/stats", dependencies=[Depends(verify_api_key)])
async def powers_stats():
    os = get_shadow_os()
    return os.shadow_powers.get_stats()


# --- Cross-Module Synthesis Engine Endpoints ---

class IngestSignalRequest(BaseModel):
    source_module: str = Field(..., max_length=50)
    category: str = Field(..., max_length=30)
    signal_type: str = Field(..., max_length=50)
    content: str = Field(..., max_length=5000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    domains: List[str] = Field(default_factory=list, max_length=10)
    entities: List[str] = Field(default_factory=list, max_length=20)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, v):
        return [d[:30] for d in v]

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v):
        return [e[:50] for e in v]


class SynthesizeRequest(BaseModel):
    modules: Optional[List[str]] = Field(default=None, max_length=10)
    categories: Optional[List[str]] = Field(default=None, max_length=10)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_syntheses: int = Field(default=5, ge=1, le=20)


class ValidateSynthesisRequest(BaseModel):
    synthesis_id: str = Field(..., max_length=20)
    outcome: str = Field(..., pattern=r'^(confirmed|partially_confirmed|refuted)$')
    ground_truth_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ResolveContradictionRequest(BaseModel):
    synthesis_id_a: str = Field(..., max_length=20)
    synthesis_id_b: str = Field(..., max_length=20)
    resolution: str = Field(..., max_length=5000)


@router.post("/synthesis/ingest", dependencies=[Depends(verify_api_key)])
async def synthesis_ingest(req: IngestSignalRequest):
    from .cross_module_synthesis import ModuleSignal
    os = get_shadow_os()
    signal = ModuleSignal(
        source_module=req.source_module,
        category=req.category,
        signal_type=req.signal_type,
        content=req.content,
        confidence=req.confidence,
        domains=req.domains,
        entities=req.entities,
        metadata=req.metadata,
    )
    signal_id = os.synthesis_engine.ingest_signal(signal)
    return {"signal_id": signal_id, "status": "ingested"}


@router.post("/synthesis/synthesize", dependencies=[Depends(verify_api_key)])
async def synthesis_synthesize(req: SynthesizeRequest):
    os = get_shadow_os()
    results = os.synthesis_engine.synthesize(
        modules=req.modules,
        categories=req.categories,
        min_confidence=req.min_confidence,
        max_syntheses=req.max_syntheses,
    )
    return [r.to_dict() for r in results]


@router.get("/synthesis/history", dependencies=[Depends(verify_api_key)])
async def synthesis_history(
    limit: int = Query(default=20, ge=1, le=200),
    domain: Optional[str] = Query(default=None, max_length=30),
    active_only: bool = Query(default=False),
):
    os = get_shadow_os()
    return os.synthesis_engine.get_synthesis_history(
        limit=limit, domain=domain, active_only=active_only
    )


@router.get("/synthesis/detail/{synthesis_id}", dependencies=[Depends(verify_api_key)])
async def synthesis_detail(synthesis_id: str):
    os = get_shadow_os()
    result = os.synthesis_engine.get_synthesis_detail(synthesis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Synthesis not found")
    return result


@router.post("/synthesis/validate", dependencies=[Depends(verify_api_key)])
async def synthesis_validate(req: ValidateSynthesisRequest):
    os = get_shadow_os()
    result = os.synthesis_engine.validate_synthesis(
        req.synthesis_id, req.outcome, req.ground_truth_confidence
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Synthesis not found")
    return result.to_dict()


@router.post("/synthesis/contradictions", dependencies=[Depends(verify_api_key)])
async def synthesis_contradictions(min_severity: float = Query(default=0.3, ge=0.0, le=1.0)):
    os = get_shadow_os()
    return os.synthesis_engine.detect_contradictions(min_severity=min_severity)


@router.post("/synthesis/resolve-contradiction", dependencies=[Depends(verify_api_key)])
async def synthesis_resolve_contradiction(req: ResolveContradictionRequest):
    os = get_shadow_os()
    result = os.synthesis_engine.resolve_contradiction(
        req.synthesis_id_a, req.synthesis_id_b, req.resolution
    )
    if result is None:
        raise HTTPException(status_code=404, detail="One or both syntheses not found")
    return result.to_dict()


@router.get("/synthesis/emergent", dependencies=[Depends(verify_api_key)])
async def synthesis_emergent(min_confidence: float = Query(default=0.4, ge=0.0, le=1.0)):
    os = get_shadow_os()
    results = os.synthesis_engine.detect_emergent_properties(min_confidence=min_confidence)
    return [r.to_dict() for r in results]


@router.get("/synthesis/trends", dependencies=[Depends(verify_api_key)])
async def synthesis_trends(min_forecast_confidence: float = Query(default=0.3, ge=0.0, le=1.0)):
    os = get_shadow_os()
    return os.synthesis_engine.get_trends(min_forecast_confidence=min_forecast_confidence)


@router.get("/synthesis/clusters", dependencies=[Depends(verify_api_key)])
async def synthesis_clusters(max_clusters: int = Query(default=10, ge=1, le=100)):
    os = get_shadow_os()
    return os.synthesis_engine.get_topic_clusters(max_clusters=max_clusters)


@router.get("/synthesis/stats", dependencies=[Depends(verify_api_key)])
async def synthesis_stats():
    os = get_shadow_os()
    return os.synthesis_engine.get_stats()


# --- Self-Modification Engine Endpoints ---

class ProposeModificationRequest(BaseModel):
    filepath: str = Field(..., max_length=500)
    modified_code: str = Field(..., max_length=300000)
    modification_type: str = Field(default="fix_bug", pattern=r'^(fix_bug|add_feature|optimize|refactor|add_test|fix_security|improve_perf|add_docs)$')
    description: str = Field(default="", max_length=5000)
    auto_snapshot: bool = Field(default=True)

    @field_validator('filepath')
    @classmethod
    def validate_filepath_safe(cls, v):
        import os as _os
        normalized = _os.path.normpath(v)
        if '..' in normalized:
            raise ValueError('filepath contains directory traversal')
        restricted = ['/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/run/', '/var/']
        for prefix in restricted:
            if normalized.startswith(prefix):
                raise ValueError(f'filepath in restricted directory: {prefix}')
        if not normalized.endswith('.py'):
            raise ValueError('only .py files can be modified')
        return v


class ValidateModificationRequest(BaseModel):
    modification_id: str = Field(..., max_length=20)


class ApplyModificationRequest(BaseModel):
    modification_id: str = Field(..., max_length=20)


@router.post("/self-mod/snapshot", dependencies=[Depends(verify_api_key)])
async def self_mod_snapshot(filepath: str = Query(..., max_length=500)):
    import os as _os
    normalized = _os.path.normpath(filepath)
    if '..' in normalized:
        raise HTTPException(status_code=400, detail="filepath contains directory traversal")
    restricted = ['/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/run/', '/var/']
    for prefix in restricted:
        if normalized.startswith(prefix):
            raise HTTPException(status_code=400, detail=f"filepath in restricted directory: {prefix}")
    if not normalized.endswith('.py'):
        raise HTTPException(status_code=400, detail="only .py files can be snapshotted")
    os = get_shadow_os()
    sid = os.self_mod.snapshot_file(filepath)
    if sid is None:
        raise HTTPException(status_code=400, detail="Failed to snapshot file")
    return {"snapshot_id": sid}


@router.post("/self-mod/propose", dependencies=[Depends(verify_api_key)])
async def self_mod_propose(req: ProposeModificationRequest):
    os = get_shadow_os()
    mod = os.self_mod.propose_modification(
        filepath=req.filepath,
        modified_code=req.modified_code,
        modification_type=req.modification_type,
        description=req.description,
        auto_snapshot=req.auto_snapshot,
    )
    if mod is None:
        raise HTTPException(status_code=400, detail="Failed to create modification")
    return mod.to_dict()


@router.post("/self-mod/validate", dependencies=[Depends(verify_api_key)])
async def self_mod_validate(req: ValidateModificationRequest):
    os = get_shadow_os()
    results = os.self_mod.validate_modification(req.modification_id)
    if "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    return results


@router.post("/self-mod/apply", dependencies=[Depends(verify_api_key)])
async def self_mod_apply(req: ApplyModificationRequest):
    os = get_shadow_os()
    success = os.self_mod.apply_modification(req.modification_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to apply modification")
    return {"status": "applied"}


@router.post("/self-mod/rollback", dependencies=[Depends(verify_api_key)])
async def self_mod_rollback(modification_id: str = Query(..., max_length=20)):
    os = get_shadow_os()
    success = os.self_mod.rollback_modification(modification_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to roll back")
    return {"status": "rolled_back"}


@router.post("/self-mod/reject", dependencies=[Depends(verify_api_key)])
async def self_mod_reject(modification_id: str = Query(..., max_length=20)):
    os = get_shadow_os()
    success = os.self_mod.reject_modification(modification_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to reject")
    return {"status": "rejected"}


@router.get("/self-mod/analyze", dependencies=[Depends(verify_api_key)])
async def self_mod_analyze():
    os = get_shadow_os()
    return os.self_mod.analyze_codebase()


@router.get("/self-mod/history", dependencies=[Depends(verify_api_key)])
async def self_mod_history(
    limit: int = Query(default=20, ge=1, le=200),
    status: Optional[str] = Query(default=None, max_length=20),
):
    os = get_shadow_os()
    return os.self_mod.get_modification_history(limit=limit, status=status)


@router.get("/self-mod/detail/{modification_id}", dependencies=[Depends(verify_api_key)])
async def self_mod_detail(modification_id: str):
    os = get_shadow_os()
    result = os.self_mod.get_modification_detail(modification_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Modification not found")
    return result


@router.get("/self-mod/stats", dependencies=[Depends(verify_api_key)])
async def self_mod_stats():
    os = get_shadow_os()
    return os.self_mod.get_stats()


# --- Cognitive Visualizer Endpoints ---

class RecordInteractionRequest(BaseModel):
    source_module: str = Field(..., max_length=100)
    target_module: str = Field(..., max_length=100)
    event_type: str = Field(..., max_length=50)
    data_size: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@router.post("/visualizer/interaction", dependencies=[Depends(verify_api_key)])
async def visualizer_record_interaction(req: RecordInteractionRequest):
    os = get_shadow_os()
    event_id = os.visualizer.record_interaction(
        source_module=req.source_module,
        target_module=req.target_module,
        event_type=req.event_type,
        data_size=req.data_size,
        confidence=req.confidence,
    )
    return {"event_id": event_id}


@router.get("/visualizer/graph", dependencies=[Depends(verify_api_key)])
async def visualizer_graph(fmt: str = Query(default="json", pattern=r'^(json|dot|mermaid|html)$')):
    os = get_shadow_os()
    return os.visualizer.generate_dependency_graph(fmt)


@router.get("/visualizer/matrix", dependencies=[Depends(verify_api_key)])
async def visualizer_matrix():
    os = get_shadow_os()
    return os.visualizer.generate_interaction_matrix()


@router.get("/visualizer/timeline", dependencies=[Depends(verify_api_key)])
async def visualizer_timeline(limit: int = Query(default=200, ge=1, le=5000)):
    os = get_shadow_os()
    return os.visualizer.generate_timeline(limit=limit)


@router.get("/visualizer/heatmap", dependencies=[Depends(verify_api_key)])
async def visualizer_heatmap():
    os = get_shadow_os()
    return os.visualizer.generate_resource_heatmap()


@router.get("/visualizer/modules", dependencies=[Depends(verify_api_key)])
async def visualizer_modules():
    os = get_shadow_os()
    return os.visualizer.get_module_overview()


@router.get("/visualizer/stats", dependencies=[Depends(verify_api_key)])
async def visualizer_stats():
    os = get_shadow_os()
    return os.visualizer.get_stats()


# --- Performance Profiler Endpoints ---

class RecordTimingRequest(BaseModel):
    module_name: str = Field(..., max_length=100)
    operation: str = Field(..., max_length=100)
    duration_ms: float = Field(..., ge=0.0)
    memory_bytes: int = Field(default=0, ge=0)
    error: bool = Field(default=False)
    cache_hit: bool = Field(default=False)


@router.post("/profiler/record", dependencies=[Depends(verify_api_key)])
async def profiler_record(req: RecordTimingRequest):
    os = get_shadow_os()
    sample_id = os.profiler.record(
        module_name=req.module_name,
        operation=req.operation,
        duration_ms=req.duration_ms,
        memory_bytes=req.memory_bytes,
        error=req.error,
        cache_hit=req.cache_hit,
    )
    return {"sample_id": sample_id}


@router.get("/profiler/module/{module_name}", dependencies=[Depends(verify_api_key)])
async def profiler_module(module_name: str):
    os = get_shadow_os()
    profile = os.profiler.get_module_profile(module_name)
    if profile is None:
        raise HTTPException(status_code=404, detail="Module not found")
    return profile.to_dict()


@router.get("/profiler/profiles", dependencies=[Depends(verify_api_key)])
async def profiler_profiles():
    os = get_shadow_os()
    return os.profiler.get_all_profiles()


@router.post("/profiler/recommendations", dependencies=[Depends(verify_api_key)])
async def profiler_generate_recommendations():
    os = get_shadow_os()
    return os.profiler.generate_recommendations()


@router.get("/profiler/recommendations", dependencies=[Depends(verify_api_key)])
async def profiler_recommendations(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return os.profiler.get_recommendations(limit=limit)


@router.get("/profiler/slow-modules", dependencies=[Depends(verify_api_key)])
async def profiler_slow_modules(
    threshold_ms: float = Query(default=50, ge=0.0),
    top_n: int = Query(default=10, ge=1, le=50),
):
    os = get_shadow_os()
    return os.profiler.get_slow_modules(threshold_ms=threshold_ms, top_n=top_n)


@router.get("/profiler/error-prone", dependencies=[Depends(verify_api_key)])
async def profiler_error_prone(
    threshold: float = Query(default=0.01, ge=0.0, le=1.0),
    top_n: int = Query(default=10, ge=1, le=50),
):
    os = get_shadow_os()
    return os.profiler.get_error_prone_modules(threshold=threshold, top_n=top_n)


@router.get("/profiler/cache-inefficient", dependencies=[Depends(verify_api_key)])
async def profiler_cache_inefficient(
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    top_n: int = Query(default=10, ge=1, le=50),
):
    os = get_shadow_os()
    return os.profiler.get_cache_inefficient_modules(threshold=threshold, top_n=top_n)


@router.get("/profiler/stats", dependencies=[Depends(verify_api_key)])
async def profiler_stats():
    os = get_shadow_os()
    return os.profiler.get_stats()


# --- Intelligence Report Endpoints ---

@router.post("/intel-report/generate", dependencies=[Depends(verify_api_key)])
async def intel_report_generate(report_type: str = Query(default="hourly", pattern=r'^(hourly|daily|weekly|on_demand)$')):
    os = get_shadow_os()
    report = os.intel_report.generate_report(shadow_os=os, report_type=report_type)
    return report.to_dict()


@router.get("/intel-report/latest", dependencies=[Depends(verify_api_key)])
async def intel_report_latest():
    os = get_shadow_os()
    report = os.intel_report.get_latest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No reports available")
    return report


@router.get("/intel-report/history", dependencies=[Depends(verify_api_key)])
async def intel_report_history(limit: int = Query(default=20, ge=1, le=200)):
    os = get_shadow_os()
    return os.intel_report.get_report_history(limit=limit)


@router.get("/intel-report/actions", dependencies=[Depends(verify_api_key)])
async def intel_report_actions(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return os.intel_report.get_pending_actions(limit=limit)


@router.post("/intel-report/resolve/{action_id}", dependencies=[Depends(verify_api_key)])
async def intel_report_resolve(action_id: str):
    os = get_shadow_os()
    success = os.intel_report.resolve_action(action_id)
    if not success:
        raise HTTPException(status_code=404, detail="Action not found")
    return {"status": "resolved"}


@router.get("/intel-report/stats", dependencies=[Depends(verify_api_key)])
async def intel_report_stats():
    os = get_shadow_os()
    return os.intel_report.get_stats()


# --- Anomaly Correlation Engine Endpoints ---

class RecordAnomalyRequest(BaseModel):
    module_name: str = Field(..., max_length=100)
    category: str = Field(..., max_length=50)
    description: str = Field(..., max_length=2000)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/anomaly-corr/record", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_record(req: RecordAnomalyRequest):
    os = get_shadow_os()
    aid = os.anomaly_corr.record_anomaly(
        module_name=req.module_name,
        category=req.category,
        description=req.description,
        severity=req.severity,
    )
    return {"anomaly_id": aid}


@router.post("/anomaly-corr/detect", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_detect():
    os = get_shadow_os()
    correlations = os.anomaly_corr.detect_correlations()
    return [c.to_dict() for c in correlations]


@router.get("/anomaly-corr/cascades", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_cascades(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return os.anomaly_corr.get_cascades(limit=limit)


@router.get("/anomaly-corr/bursts", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_bursts(
    window_seconds: float = Query(default=300, ge=1, le=3600),
    min_count: int = Query(default=3, ge=2, le=20),
):
    os = get_shadow_os()
    return os.anomaly_corr.detect_bursts(
        window_seconds=window_seconds, min_count=min_count
    )


@router.get("/anomaly-corr/recent", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_recent(
    limit: int = Query(default=50, ge=1, le=500),
    category: Optional[str] = Query(default=None, max_length=50),
):
    os = get_shadow_os()
    return os.anomaly_corr.get_recent_anomalies(limit=limit, category=category)


@router.get("/anomaly-corr/related/{anomaly_id}", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_related(
    anomaly_id: str,
    min_score: float = Query(default=0.3, ge=0.0, le=1.0),
):
    os = get_shadow_os()
    return os.anomaly_corr.get_correlated_anomalies(
        anomaly_id=anomaly_id, min_score=min_score
    )


@router.get("/anomaly-corr/stats", dependencies=[Depends(verify_api_key)])
async def anomaly_corr_stats():
    os = get_shadow_os()
    return os.anomaly_corr.get_stats()


# --- Module Capability Registry Endpoints ---

class RegisterModuleRequest(BaseModel):
    module_name: str = Field(..., max_length=100)
    capabilities: List[str] = Field(..., min_length=1, max_length=20)
    dependencies: Optional[List[str]] = Field(default=None, max_length=20)
    display_name: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1000)


@router.post("/registry/register", dependencies=[Depends(verify_api_key)])
async def registry_register(req: RegisterModuleRequest):
    os = get_shadow_os()
    module_id = os.module_registry.register_module(
        module_name=req.module_name,
        capabilities=req.capabilities,
        dependencies=req.dependencies,
        display_name=req.display_name,
        description=req.description,
    )
    if module_id is None:
        raise HTTPException(status_code=400, detail="Circular dependency detected")
    return {"module_id": module_id, "status": "registered"}


@router.post("/registry/status/{module_name}", dependencies=[Depends(verify_api_key)])
async def registry_set_status(module_name: str, status: str = Query(..., pattern=r'^(active|inactive|degraded|error)$')):
    os = get_shadow_os()
    success = os.module_registry.set_status(module_name, status)
    if not success:
        raise HTTPException(status_code=404, detail="Module not found")
    return {"module_name": module_name, "status": status}


@router.post("/registry/health/{module_name}", dependencies=[Depends(verify_api_key)])
async def registry_set_health(module_name: str, health_score: float = Query(..., ge=0.0, le=1.0)):
    os = get_shadow_os()
    success = os.module_registry.set_health(module_name, health_score)
    if not success:
        raise HTTPException(status_code=404, detail="Module not found")
    return {"module_name": module_name, "health_score": health_score}


@router.get("/registry/find/capability/{capability}", dependencies=[Depends(verify_api_key)])
async def registry_find_by_capability(capability: str):
    os = get_shadow_os()
    return os.module_registry.find_by_capability(capability)


@router.get("/registry/find/depends/{dep_name}", dependencies=[Depends(verify_api_key)])
async def registry_find_by_dependency(dep_name: str):
    os = get_shadow_os()
    return os.module_registry.find_by_dependency(dep_name)


@router.get("/registry/module/{module_name}", dependencies=[Depends(verify_api_key)])
async def registry_get_module(module_name: str):
    os = get_shadow_os()
    result = os.module_registry.get_module(module_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Module not found")
    return result


@router.get("/registry/modules", dependencies=[Depends(verify_api_key)])
async def registry_list_modules(status: Optional[str] = Query(default=None, pattern=r'^(active|inactive|degraded|error|initializing|initializing|shutting_down)$')):
    os = get_shadow_os()
    return os.module_registry.list_modules(status=status)


@router.get("/registry/graph", dependencies=[Depends(verify_api_key)])
async def registry_dependency_graph():
    os = get_shadow_os()
    return os.module_registry.get_dependency_graph()


@router.get("/registry/compatible/{module_name}", dependencies=[Depends(verify_api_key)])
async def registry_compatible(module_name: str):
    os = get_shadow_os()
    return os.module_registry.get_compatible_modules(module_name)


@router.get("/registry/stats", dependencies=[Depends(verify_api_key)])
async def registry_stats():
    os = get_shadow_os()
    return os.module_registry.get_stats()


# --- Decision Arbiter Endpoints ---

class SubmitClaimRequest(BaseModel):
    source_module: str = Field(..., max_length=100)
    statement: str = Field(..., max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[List[str]] = Field(default=None, max_length=10)
    counter_evidence: Optional[List[str]] = Field(default=None, max_length=10)
    domain: str = Field(default="general", max_length=50)

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, v):
        return [e[:500] for e in (v or [])] if v else []

    @field_validator("counter_evidence")
    @classmethod
    def validate_counter_evidence(cls, v):
        return [e[:500] for e in (v or [])] if v else []


@router.post("/arbiter/submit", dependencies=[Depends(verify_api_key)])
async def arbiter_submit_claim(req: SubmitClaimRequest):
    os = get_shadow_os()
    claim_id = os.arbiter.submit_claim(
        source_module=req.source_module,
        statement=req.statement,
        confidence=req.confidence,
        evidence=req.evidence,
        counter_evidence=req.counter_evidence,
        domain=req.domain,
    )
    return {"claim_id": claim_id, "status": "submitted"}


@router.post("/arbiter/adjudicate/{claim_id}", dependencies=[Depends(verify_api_key)])
async def arbiter_adjudicate(claim_id: str):
    os = get_shadow_os()
    judgment = os.arbiter.adjudicate(claim_id)
    if judgment is None:
        raise HTTPException(status_code=404, detail="Claim not found or already resolved")
    return judgment.to_dict()


@router.post("/arbiter/record-outcome/{module_name}", dependencies=[Depends(verify_api_key)])
async def arbiter_record_outcome(module_name: str, correct: bool = Query(...)):
    os = get_shadow_os()
    os.arbiter.record_source_outcome(module_name, correct)
    return {"module_name": module_name, "correct": correct, "status": "recorded"}


@router.get("/arbiter/judgment/{claim_id}", dependencies=[Depends(verify_api_key)])
async def arbiter_get_judgment(claim_id: str):
    os = get_shadow_os()
    result = os.arbiter.get_judgment(claim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Judgment not found")
    return result


@router.get("/arbiter/claims", dependencies=[Depends(verify_api_key)])
async def arbiter_get_claims(
    resolved: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    return os.arbiter.get_claims(resolved=resolved, limit=limit)


@router.get("/arbiter/judgments", dependencies=[Depends(verify_api_key)])
async def arbiter_get_judgments(limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return os.arbiter.get_judgments(limit=limit)


@router.get("/arbiter/trust", dependencies=[Depends(verify_api_key)])
async def arbiter_get_trust():
    os = get_shadow_os()
    return os.arbiter.get_source_trust()


@router.get("/arbiter/stats", dependencies=[Depends(verify_api_key)])
async def arbiter_stats():
    os = get_shadow_os()
    return os.arbiter.get_stats()


# ═══════════════════════════════════════════════════════════════════
# VISION PERCEPTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class VisionAnalyzeRequest(BaseModel):
    description: str = Field(default="", max_length=5000)
    source: str = Field(default="api", max_length=100)


class VisionSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)


class VisionFileRequest(BaseModel):
    filepath: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=5000)

    @field_validator('filepath')
    @classmethod
    def validate_filepath_safe(cls, v):
        import os as _os
        normalized = _os.path.normpath(v).replace('\\', '/')
        if '..' in normalized.split('/'):
            raise ValueError('filepath contains directory traversal')
        restricted = ('/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/run/',
                      '/var/', '/boot/', '/bin/', '/sbin/', '/usr/')
        lowered = normalized.lower()
        for prefix in restricted:
            if lowered.startswith(prefix):
                raise ValueError(f'filepath in restricted directory: {prefix}')
        return v


@router.post("/vision/analyze", dependencies=[Depends(verify_api_key)])
async def vision_analyze(request: VisionAnalyzeRequest):
    os = get_shadow_os()
    result = os.vision.capture_and_analyze(description=request.description, source=request.source)
    return result.to_dict()


@router.post("/vision/analyze-file", dependencies=[Depends(verify_api_key)])
async def vision_analyze_file(request: VisionFileRequest):
    os = get_shadow_os()
    try:
        result = os.vision.analyze_from_file(request.filepath, description=request.description)
        return result.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=type(e).__name__)


@router.post("/vision/search", dependencies=[Depends(verify_api_key)])
async def vision_search(request: VisionSearchRequest):
    os = get_shadow_os()
    results = os.vision.search_memory(query=request.query, limit=request.limit)
    return {"results": results, "count": len(results)}


@router.get("/vision/recent", dependencies=[Depends(verify_api_key)])
async def vision_recent(limit: int = Query(default=10, ge=1, le=100)):
    os = get_shadow_os()
    results = os.vision.get_recent(limit=limit)
    return {"results": results, "count": len(results)}


@router.get("/vision/stats", dependencies=[Depends(verify_api_key)])
async def vision_stats():
    os = get_shadow_os()
    return os.vision.get_stats()


@router.post("/vision/compare", dependencies=[Depends(verify_api_key)])
async def vision_compare(request: VisionSearchRequest):
    os = get_shadow_os()
    memories = os.vision.search_memory(query=request.query, limit=2)
    if len(memories) < 2:
        return {"comparison": None, "message": "Need at least 2 memories to compare"}
    m1 = memories[0]
    m2 = memories[1]
    similarity = 0.0
    if "similarity_to_previous" in m2:
        similarity = m2["similarity_to_previous"]
    change = m2.get("change_from_previous")
    return {"comparison": {"memory_1_id": m1.get("memory_id"), "memory_2_id": m2.get("memory_id"), "similarity": similarity, "change": change}, "count": 2}


# ═══════════════════════════════════════════════════════════════════
# NOTIFICATION BUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class PushNotifyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field(default="normal", max_length=20)
    source_module: str = Field(default="shadow_os", max_length=100)
    category: str = Field(default="general", max_length=50)


class MarkReadRequest(BaseModel):
    notification_id: str = Field(..., max_length=20)


@router.post("/notifications/push", dependencies=[Depends(verify_api_key)])
async def push_notification(request: PushNotifyRequest):
    os = get_shadow_os()
    nid = os.notifications.notify(
        title=request.title, body=request.body, priority=request.priority,
        source_module=request.source_module, category=request.category,
    )
    return {"notification_id": nid}


@router.post("/notifications/push-critical", dependencies=[Depends(verify_api_key)])
async def push_critical(request: PushNotifyRequest):
    os = get_shadow_os()
    nid = os.notifications.notify_critical(
        title=request.title, body=request.body,
        source_module=request.source_module, category=request.category,
    )
    return {"notification_id": nid}


@router.get("/notifications/list", dependencies=[Depends(verify_api_key)])
async def list_notifications(
    priority: Optional[str] = Query(default=None),
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    result = os.notifications.get_notifications(priority=priority, unread_only=unread_only, limit=limit)
    return {"notifications": result, "count": len(result)}


@router.get("/notifications/unread-count", dependencies=[Depends(verify_api_key)])
async def unread_notification_count():
    os = get_shadow_os()
    return {"unread_count": os.notifications.get_unread_count()}


@router.post("/notifications/mark-read", dependencies=[Depends(verify_api_key)])
async def mark_notification_read(request: MarkReadRequest):
    os = get_shadow_os()
    success = os.notifications.mark_read(request.notification_id)
    return {"success": success}


@router.get("/notifications/digest", dependencies=[Depends(verify_api_key)])
async def notification_digest():
    os = get_shadow_os()
    digest = os.notifications.generate_digest()
    return digest.to_dict()


@router.get("/notifications/stats", dependencies=[Depends(verify_api_key)])
async def notification_stats():
    os = get_shadow_os()
    return os.notifications.get_stats()


# ═══════════════════════════════════════════════════════════════════
# VISUAL OBSERVATION BRIDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class VisualProcessRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=5000)
    scene_category: str = Field(default="", max_length=50)
    objects: Optional[List[str]] = Field(default=None, max_length=50)
    faces: int = Field(default=0, ge=0, le=100)
    has_text: bool = Field(default=False)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualStreamRequest(BaseModel):
    interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    max_frames: int = Field(default=3600, ge=10, le=86400)


@router.post("/vision/process", dependencies=[Depends(verify_api_key)])
async def vision_process_observation(request: VisualProcessRequest):
    os = get_shadow_os()
    result = os.visual_bridge.process_visual_observation(
        description=request.description,
        scene_category=request.scene_category,
        objects=request.objects,
        faces=request.faces,
        has_text=request.has_text,
        importance=request.importance,
    )
    return result.to_dict()


@router.get("/vision/bridge/events", dependencies=[Depends(verify_api_key)])
async def visual_bridge_events(limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    return {"events": os.visual_bridge.get_recent_events(limit=limit)}


@router.get("/vision/bridge/stats", dependencies=[Depends(verify_api_key)])
async def visual_bridge_stats():
    os = get_shadow_os()
    return os.visual_bridge.get_stats()


@router.post("/vision/stream/start", dependencies=[Depends(verify_api_key)])
async def vision_stream_start(request: VisualStreamRequest):
    os = get_shadow_os()
    started = os.vision.start_streaming(request.interval_seconds, request.max_frames)
    return {"streaming": started, "status": os.vision.get_streaming_status()}


@router.post("/vision/stream/stop", dependencies=[Depends(verify_api_key)])
async def vision_stream_stop():
    os = get_shadow_os()
    frames = os.vision.stop_streaming()
    return {"frames_captured": frames}



# ═══════════════════════════════════════════════════════════════════
# SHADOW DEPLOY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class ProvisionShadowRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    device_id: str = Field(default="", max_length=100)
    mode: str = Field(default="sentry", max_length=20, pattern=r'^(sentry|patrol|listen)$')
    location: str = Field(default="", max_length=500)
    camera_interval: float = Field(default=10.0, ge=1.0, le=300.0)
    mic_sample: float = Field(default=5.0, ge=1.0, le=30.0)
    report_interval: float = Field(default=5.0, ge=1.0, le=60.0)
    alert_on_faces: bool = Field(default=True)
    alert_on_motion: bool = Field(default=True)
    alert_on_sound: bool = Field(default=True)


class ShadowReportRequest(BaseModel):
    shadow_id: str = Field(..., max_length=20)
    auth_token: str = Field(..., max_length=100)
    visual_observation: Optional[str] = Field(default=None, max_length=5000)
    scene_category: Optional[str] = Field(default=None, max_length=50)
    faces_detected: int = Field(default=0, ge=0, le=100)
    motion_detected: bool = Field(default=False)
    sound_event: Optional[str] = Field(default=None, max_length=2000)
    sound_level_db: float = Field(default=0.0, ge=0.0, le=120.0)
    light_level: float = Field(default=0.0, ge=0.0, le=1.0)
    battery_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    is_charging: bool = Field(default=False)
    location_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    location_lng: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    signal_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class ShadowHeartbeatRequest(BaseModel):
    shadow_id: str = Field(..., max_length=20)
    auth_token: str = Field(..., max_length=100)


class ShadowConfigRequest(BaseModel):
    shadow_id: str = Field(..., max_length=20)
    auth_token: str = Field(..., max_length=100)
    camera_interval: Optional[float] = Field(default=None, ge=1.0, le=300.0)
    mic_sample: Optional[float] = Field(default=None, ge=1.0, le=30.0)
    report_interval: Optional[float] = Field(default=None, ge=1.0, le=60.0)
    alert_on_faces: Optional[bool] = None
    alert_on_sound: Optional[bool] = None


class ShadowDeprovisionRequest(BaseModel):
    shadow_id: str = Field(..., max_length=20)


@router.post("/deploy/provision", dependencies=[Depends(verify_api_key)])
async def deploy_provision(request: ProvisionShadowRequest):
    os = get_shadow_os()
    shadow = os.shadow_deploy.provision_shadow(
        name=request.name,
        device_id=request.device_id,
        mode=request.mode,
        location=request.location,
        camera_interval=request.camera_interval,
        mic_sample=request.mic_sample,
        report_interval=request.report_interval,
        alert_on_faces=request.alert_on_faces,
        alert_on_motion=request.alert_on_motion,
        alert_on_sound=request.alert_on_sound,
    )
    # Return auth token only at provision time
    result = shadow.to_dict()
    result["auth_token"] = shadow.auth_token
    return result


@router.post("/deploy/deprovision", dependencies=[Depends(verify_api_key)])
async def deploy_deprovision(request: ShadowDeprovisionRequest):
    os = get_shadow_os()
    success = os.shadow_deploy.deprovision_shadow(request.shadow_id)
    return {"deprovisioned": success}


@router.get("/deploy/shadows", dependencies=[Depends(verify_api_key)])
async def deploy_list_shadows(status: Optional[str] = None, mode: Optional[str] = None):
    os = get_shadow_os()
    return {"shadows": os.shadow_deploy.list_shadows(status=status, mode=mode)}


@router.get("/deploy/shadow/{shadow_id}", dependencies=[Depends(verify_api_key)])
async def deploy_get_shadow(shadow_id: str):
    os = get_shadow_os()
    result = os.shadow_deploy.get_shadow(shadow_id)
    if not result:
        raise HTTPException(status_code=404, detail="Shadow not found")
    return result


@router.post("/deploy/heartbeat")
async def deploy_heartbeat(request: ShadowHeartbeatRequest):
    os = get_shadow_os()
    ok = os.shadow_deploy.heartbeat(request.shadow_id, request.auth_token)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid shadow credentials")
    return {"alive": True}


@router.post("/deploy/report")
async def deploy_submit_report(request: ShadowReportRequest):
    os = get_shadow_os()
    report = os.shadow_deploy.submit_report(
        shadow_id=request.shadow_id,
        auth_token=request.auth_token,
        visual_observation=request.visual_observation,
        scene_category=request.scene_category,
        faces_detected=request.faces_detected,
        motion_detected=request.motion_detected,
        sound_event=request.sound_event,
        sound_level_db=request.sound_level_db,
        light_level=request.light_level,
        battery_pct=request.battery_pct,
        is_charging=request.is_charging,
        location_lat=request.location_lat,
        location_lng=request.location_lng,
        signal_strength=request.signal_strength,
    )
    if not report:
        raise HTTPException(status_code=401, detail="Invalid shadow credentials")
    return {"report_id": report.report_id, "alert": report.alert_triggered}


@router.get("/deploy/dashboard")
async def deploy_dashboard(shadow_id: str = Query(...), auth_token: str = Query(...)):
    os = get_shadow_os()
    result = os.shadow_deploy.get_shadow_dashboard(shadow_id, auth_token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid shadow credentials")
    return result


@router.post("/deploy/config")
async def deploy_update_config(request: ShadowConfigRequest):
    os = get_shadow_os()
    success = os.shadow_deploy.update_config(
        shadow_id=request.shadow_id,
        auth_token=request.auth_token,
        camera_interval=request.camera_interval,
        mic_sample=request.mic_sample,
        report_interval=request.report_interval,
        alert_on_faces=request.alert_on_faces,
        alert_on_sound=request.alert_on_sound,
    )
    return {"updated": success}


@router.post("/deploy/idle")
async def deploy_mark_idle(request: ShadowHeartbeatRequest):
    os = get_shadow_os()
    ok = os.shadow_deploy.mark_shadow_idle(request.shadow_id, request.auth_token)
    return {"idle": ok}


@router.get("/deploy/reports", dependencies=[Depends(verify_api_key)])
async def deploy_reports(shadow_id: Optional[str] = None, alerts_only: bool = False,
                         limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    return {"reports": os.shadow_deploy.get_reports(
        shadow_id=shadow_id, limit=limit, alerts_only=alerts_only
    )}




# ═══════════════════════════════════════════════════════════════════
# SHADOW JOURNAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class JournalRecordRequest(BaseModel):
    entry_type: str = Field(..., max_length=20)
    content: str = Field(..., min_length=1, max_length=10000)
    category: str = Field(default="general", max_length=50)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: Optional[List[str]] = Field(default=None, max_length=20)
    summary: str = Field(default="", max_length=2000)


@router.post("/journal/record", dependencies=[Depends(verify_api_key)])
async def journal_record(request: JournalRecordRequest):
    os = get_shadow_os()
    eid = os.journal.record(
        entry_type=request.entry_type, content=request.content,
        category=request.category, importance=request.importance,
        tags=request.tags, summary=request.summary,
    )
    return {"entry_id": eid}


@router.get("/journal/entries", dependencies=[Depends(verify_api_key)])
async def journal_entries(
    date: Optional[str] = Query(default=None),
    entry_type: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    os = get_shadow_os()
    result = os.journal.get_entries(date=date, entry_type=entry_type,
                                    category=category, limit=limit)
    return {"entries": result, "count": len(result)}


@router.get("/journal/entry/{entry_id}", dependencies=[Depends(verify_api_key)])
async def journal_get_entry(entry_id: str):
    os = get_shadow_os()
    entry = os.journal.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.get("/journal/search", dependencies=[Depends(verify_api_key)])
async def journal_search(query: str = Query(..., min_length=1, max_length=500),
                         limit: int = Query(default=50, ge=1, le=500)):
    os = get_shadow_os()
    result = os.journal.search(query=query, limit=limit)
    return {"results": result, "count": len(result)}


@router.get("/journal/daily-summary", dependencies=[Depends(verify_api_key)])
async def journal_daily_summary(date: Optional[str] = Query(default=None)):
    os = get_shadow_os()
    summary = os.journal.generate_daily_summary(date=date)
    return summary.to_dict()


@router.get("/journal/weekly-digest", dependencies=[Depends(verify_api_key)])
async def journal_weekly_digest():
    os = get_shadow_os()
    return os.journal.get_weekly_digest()


@router.get("/journal/export", dependencies=[Depends(verify_api_key)])
async def journal_export(start_date: Optional[str] = None, end_date: Optional[str] = None,
                         fmt: str = "json"):
    os = get_shadow_os()
    data = os.journal.export_journal(start_date=start_date, end_date=end_date, fmt=fmt)
    if fmt == "json":
        return FastAPIResponse(content=data, media_type="application/json")
    return FastAPIResponse(content=data, media_type="text/plain")


# (Duplicate `/journal/stats` definition removed — original is at line ~2384.)


@router.get("/dashboard/full", dependencies=[Depends(verify_api_key)])
async def dashboard_full(force: bool = True):
    os = get_shadow_os()
    return os.dashboard_agg.get_snapshot(force=force).to_dict()



# ═══════════════════════════════════════════════════════════════════
# PREDICTIVE ALERTS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/predictive/analyze", dependencies=[Depends(verify_api_key)])
async def predictive_analyze():
    os = get_shadow_os()
    alerts = os.predictive.analyze_and_predict()
    return {"alerts": [a.to_dict() for a in alerts], "count": len(alerts)}


@router.get("/predictive/alerts", dependencies=[Depends(verify_api_key)])
async def predictive_get_alerts(
    category: Optional[str] = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    unacknowledged_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
):
    os = get_shadow_os()
    result = os.predictive.get_alerts(
        category=category, min_confidence=min_confidence,
        unacknowledged_only=unacknowledged_only, limit=limit
    )
    return {"alerts": result, "count": len(result)}


@router.get("/predictive/alert/{alert_id}", dependencies=[Depends(verify_api_key)])
async def predictive_get_alert(alert_id: str):
    os = get_shadow_os()
    alert = os.predictive.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/predictive/acknowledge", dependencies=[Depends(verify_api_key)])
async def predictive_acknowledge(alert_id: str = Query(..., max_length=20)):
    os = get_shadow_os()
    success = os.predictive.acknowledge(alert_id)
    return {"acknowledged": success}


@router.post("/predictive/resolve", dependencies=[Depends(verify_api_key)])
async def predictive_resolve(alert_id: str = Query(..., max_length=20)):
    os = get_shadow_os()
    success = os.predictive.resolve(alert_id)
    return {"resolved": success}


@router.get("/predictive/stats", dependencies=[Depends(verify_api_key)])
async def predictive_stats():
    os = get_shadow_os()
    return os.predictive.get_stats()



# ═══════════════════════════════════════════════════════════════════
# SHADOW POOL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class PoolTaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field(default="normal", max_length=20)
    chain_name: Optional[str] = Field(default=None, max_length=50)


@router.post("/pool/submit", dependencies=[Depends(verify_api_key)])
async def pool_submit_task(request: PoolTaskRequest):
    os = get_shadow_os()
    task = os.shadow_pool.submit_task(
        title=request.title, description=request.description,
        priority=request.priority, chain_name=request.chain_name,
    )
    return {"task": task.to_dict(), "pool_stats": os.shadow_pool.get_stats()}


@router.get("/pool/status", dependencies=[Depends(verify_api_key)])
async def pool_status():
    os = get_shadow_os()
    return os.get_pool_status()


@router.get("/pool/tasks", dependencies=[Depends(verify_api_key)])
async def pool_tasks(status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)):
    os = get_shadow_os()
    return {"tasks": os.shadow_pool.get_tasks(status=status, limit=limit)}


@router.get("/pool/task/{task_id}", dependencies=[Depends(verify_api_key)])
async def pool_get_task(task_id: str):
    os = get_shadow_os()
    task = os.shadow_pool.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/pool/task/{task_id}/cancel", dependencies=[Depends(verify_api_key)])
async def pool_cancel_task(task_id: str):
    os = get_shadow_os()
    success = os.shadow_pool.cancel_task(task_id)
    return {"cancelled": success}


@router.get("/pool/workers", dependencies=[Depends(verify_api_key)])
async def pool_workers(role: Optional[str] = None):
    os = get_shadow_os()
    return {"workers": os.shadow_pool.get_workers(role=role)}


@router.get("/pool/worker/{worker_id}", dependencies=[Depends(verify_api_key)])
async def pool_get_worker(worker_id: str):
    os = get_shadow_os()
    worker = os.shadow_pool.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker


@router.post("/pool/worker/{worker_id}/deploy", dependencies=[Depends(verify_api_key)])
async def pool_deploy_worker(worker_id: str, device_id: str = Query(...),
                              location: str = Query(default="")):
    os = get_shadow_os()
    result = os.shadow_pool.deploy_worker_to_remote(
        worker_id=worker_id, device_id=device_id, location=location
    )
    if not result:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"deployment": result}


@router.get("/pool/chains", dependencies=[Depends(verify_api_key)])
async def pool_chains():
    os = get_shadow_os()
    return {"chains": os.shadow_pool.get_available_chains()}


@router.get("/pool/stats", dependencies=[Depends(verify_api_key)])
async def pool_stats():
    os = get_shadow_os()
    return os.shadow_pool.get_stats()

@router.get("/deploy/stats", dependencies=[Depends(verify_api_key)])
async def deploy_stats():
    os = get_shadow_os()
    return os.shadow_deploy.get_stats()



@router.post("/vision/consolidate", dependencies=[Depends(verify_api_key)])
async def vision_consolidate(limit: int = 50):
    os = get_shadow_os()
    if hasattr(os, 'dream') and os.dream:
        result = os.vision.consolidate_to_dream(dream_state=os.dream, limit=limit)
        return result
    return {"consolidated": 0, "skipped": 0, "error": "DreamState not available"}

@router.get("/vision/stream/status", dependencies=[Depends(verify_api_key)])
async def vision_stream_status():
    os = get_shadow_os()
    return os.vision.get_streaming_status()


@router.post("/compact", dependencies=[Depends(verify_api_key)])
async def compact_memory():
    """Manually trigger memory compaction. Evicts low-importance atoms, generates summaries."""
    os = get_shadow_os()
    evicted = await os.memory.compact()
    stats = os.memory.get_stats()
    return {"evicted": evicted, "remaining": stats["total_atoms"], "stats": stats}


@router.get("/compact/status", dependencies=[Depends(verify_api_key)])
async def compact_status():
    """Get memory compaction/usage status."""
    os = get_shadow_os()
    stats = os.memory.get_stats()
    return {
        "total_atoms": stats["total_atoms"],
        "compactions": stats["stats"]["compactions"],
        "atoms_evicted": stats["stats"]["atoms_evicted"],
        "auto_compactions": stats["stats"].get("auto_compactions", 0),
        "summaries_generated": stats["stats"].get("summaries_generated", 0),
        "working_count": stats.get("by_tier", {}).get("working", 0),
        "archival_count": stats.get("by_tier", {}).get("archival", 0),
    }
