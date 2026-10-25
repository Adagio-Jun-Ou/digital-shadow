#!/usr/bin/env python3
"""Phone Integration API v2 - Fixed"""

import os
import sys
import json
import hashlib
import secrets
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("shadowd.phone_api")

PHONE_V2_CORS_ORIGINS = os.environ.get(
    "PHONE_CORS_ORIGINS",
    "http://localhost:8765,http://localhost:3000"
).split(",")

app = FastAPI(title="Shadow AI Phone API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=PHONE_V2_CORS_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Models
class LocationData(BaseModel):
    latitude: float
    longitude: float
    accuracy: float = 10.0
    altitude: Optional[float] = None
    speed: Optional[float] = None
    timestamp: datetime
    device_id: str

class BluetoothData(BaseModel):
    device_id: str
    timestamp: datetime
    devices: List[Dict]

class WifiData(BaseModel):
    device_id: str
    timestamp: datetime
    networks: List[Dict]

# Storage
paired_devices: Dict[str, Dict] = {}
observations: List[Dict] = []
MAX_OBSERVATIONS = 10000
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)) -> bool:
    if not paired_devices:
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    for device_id, device in paired_devices.items():
        if device.get("key_hash") == key_hash:
            return True
    raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"name": "Shadow AI Phone API", "version": "2.0.0", "status": "running"}

@app.on_event("startup")
async def startup():
    logger.info("Phone API v2.0.0 started")
    Path.home().joinpath("digital-shadow/data/phone").mkdir(parents=True, exist_ok=True)

class PairRequest(BaseModel):
    device_id: str
    device_name: str
    device_type: str

@app.post("/pair")
async def pair_device(request: PairRequest):
    api_key = secrets.token_urlsafe(32)
    paired_devices[request.device_id] = {
        "name": request.device_name, "type": request.device_type,
        "paired_at": datetime.now().isoformat(),
        "key_hash": hashlib.sha256(api_key.encode()).hexdigest()
    }
    return {"success": True, "device_id": request.device_id, "api_key": api_key}

@app.post("/observe/location")
async def obs_location(data: LocationData, authorized: bool = Depends(verify_api_key)):
    obs = {"type": "location", **data.dict(), "received": datetime.now().isoformat()}
    observations.append(obs)
    if len(observations) > MAX_OBSERVATIONS:
        del observations[:len(observations) - MAX_OBSERVATIONS]
    return {"status": "observed", "id": len(observations)}

@app.post("/observe/bluetooth")
async def obs_bluetooth(data: BluetoothData, authorized: bool = Depends(verify_api_key)):
    if len(data.devices) > 100:
        raise HTTPException(status_code=400, detail="Too many devices in single observation")
    obs = {"type": "bluetooth", **data.dict(), "received": datetime.now().isoformat()}
    observations.append(obs)
    if len(observations) > MAX_OBSERVATIONS:
        del observations[:len(observations) - MAX_OBSERVATIONS]
    return {"status": "observed", "devices": len(data.devices)}

@app.post("/observe/wifi")
async def obs_wifi(data: WifiData, authorized: bool = Depends(verify_api_key)):
    if len(data.networks) > 100:
        raise HTTPException(status_code=400, detail="Too many networks in single observation")
    obs = {"type": "wifi", **data.dict(), "received": datetime.now().isoformat()}
    observations.append(obs)
    if len(observations) > MAX_OBSERVATIONS:
        del observations[:len(observations) - MAX_OBSERVATIONS]
    return {"status": "observed", "networks": len(data.networks)}

@app.get("/devices")
async def list_devices():
    return {"devices": paired_devices, "total": len(paired_devices)}

@app.get("/observations")
async def get_observations(limit: int = 100):
    return {"observations": observations[-limit:], "total": len(observations)}

@app.get("/health")
async def health():
    return {"status": "healthy", "devices": len(paired_devices), "observations": len(observations)}

if __name__ == "__main__":
    host = os.environ.get("PHONE_HOST", "127.0.0.1")
    port = int(os.environ.get("PHONE_PORT", "8766"))
    uvicorn.run(app, host=host, port=port)
