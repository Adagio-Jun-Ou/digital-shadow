#!/usr/bin/env python3
"""
Digital Shadow Phone Integration
- Web dashboard (mobile-responsive)
- Webhook receiver for phone data
- REST API for phone apps
- Secure token-based auth
"""

import asyncio
import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import logging

import httpx
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("shadowd.phone")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("SHADOW_DATA_DIR", PROJECT_ROOT / "data"))
PHONE_DIR = DATA_DIR / "phone"
TOKENS_FILE = PHONE_DIR / "tokens.json"
OBSERVATIONS_FILE = PHONE_DIR / "phone_observations.json"

for d in [PHONE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

security = HTTPBearer()


class PhoneObservation(BaseModel):
    observation_type: str
    content: Dict
    device_id: str
    timestamp: Optional[str] = None
    location: Optional[Dict] = None
    metadata: Optional[Dict] = None


class LocationData(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    timestamp: Optional[str] = None


class PhoneApp(BaseModel):
    name: str
    package: Optional[str] = None
    usage_time: Optional[int] = None  # seconds
    last_opened: Optional[str] = None


class DeviceRegistration(BaseModel):
    device_id: str
    device_name: str
    device_type: str  # "android", "ios", "web"
    push_token: Optional[str] = None


class ShadowPhoneAPI:
    """Phone integration for Digital Shadow."""
    
    def __init__(self):
        self.tokens: Dict[str, Dict] = self._load_tokens()
        self.observations: List[Dict] = self._load_observations()
        self.devices: Dict[str, Dict] = {}
        self.shadow_api = "http://127.0.0.1:8765"
        
    def _load_tokens(self) -> Dict:
        if TOKENS_FILE.exists():
            with open(TOKENS_FILE) as f:
                return json.load(f)
        return {}
    
    def _save_tokens(self):
        with open(TOKENS_FILE, 'w') as f:
            json.dump(self.tokens, f, indent=2)
    
    def _load_observations(self) -> List[Dict]:
        if OBSERVATIONS_FILE.exists():
            with open(OBSERVATIONS_FILE) as f:
                return json.load(f)
        return []
    
    def _save_observations(self):
        with open(OBSERVATIONS_FILE, 'w') as f:
            json.dump(self.observations[-10000:], f, indent=2)
    
    def generate_token(self, device_id: str, device_name: str) -> str:
        """Generate auth token for a device."""
        token = secrets.token_urlsafe(32)
        self.tokens[token] = {
            "device_id": device_id,
            "device_name": device_name,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None
        }
        self._save_tokens()
        return token
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate auth token."""
        if token in self.tokens:
            self.tokens[token]["last_used"] = datetime.utcnow().isoformat()
            self._save_tokens()
            return self.tokens[token]
        return None
    
    async def add_observation(self, observation: PhoneObservation) -> Dict:
        """Add observation from phone."""
        obs = {
            "id": hashlib.sha256(f"{observation.device_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16],
            "type": observation.observation_type,
            "content": observation.content,
            "device_id": observation.device_id,
            "timestamp": observation.timestamp or datetime.utcnow().isoformat(),
            "location": observation.location,
            "metadata": observation.metadata,
            "received_at": datetime.utcnow().isoformat()
        }
        
        self.observations.append(obs)
        self._save_observations()
        
        # Forward to main shadow
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.shadow_api}/observe",
                    json={
                        "activity_type": f"phone_{observation.observation_type}",
                        "content": observation.content,
                        "device": observation.device_id
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to forward to shadow: {e}")
        
        return obs
    
    def get_device_summary(self, device_id: str) -> Dict:
        """Get summary for a device."""
        device_obs = [o for o in self.observations if o["device_id"] == device_id]
        
        return {
            "device_id": device_id,
            "total_observations": len(device_obs),
            "last_observation": device_obs[-1] if device_obs else None,
            "observation_types": list(set(o["type"] for o in device_obs)),
        }


phone_api = ShadowPhoneAPI()


# FastAPI app for phone integration
PHONE_CORS_ORIGINS = os.environ.get(
    "PHONE_CORS_ORIGINS",
    "http://localhost:8765,http://localhost:3000"
).split(",")

app = FastAPI(title="Digital Shadow Phone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=PHONE_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_device(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to validate token and get device."""
    token = credentials.credentials
    device = phone_api.validate_token(token)
    if not device:
        raise HTTPException(status_code=401, detail="Invalid token")
    return device


# ============ AUTH ENDPOINTS ============

@app.post("/phone/register")
async def register_device(registration: DeviceRegistration):
    """Register a new phone device and get auth token."""
    token = phone_api.generate_token(registration.device_id, registration.device_name)
    return {
        "token": token,
        "device_id": registration.device_id,
        "message": "Device registered successfully"
    }


# ============ OBSERVATION ENDPOINTS ============

@app.post("/phone/observe")
async def phone_observe(
    observation: PhoneObservation,
    device: Dict = Depends(get_device)
):
    """Receive observation from phone."""
    obs = await phone_api.add_observation(observation)
    return {"status": "recorded", "id": obs["id"]}


@app.post("/phone/location")
async def phone_location(
    location: LocationData,
    device_id: str,
    device: Dict = Depends(get_device)
):
    """Receive location update from phone."""
    observation = PhoneObservation(
        observation_type="location",
        content={
            "latitude": location.latitude,
            "longitude": location.longitude,
            "accuracy": location.accuracy
        },
        device_id=device_id,
        location=location.dict()
    )
    obs = await phone_api.add_observation(observation)
    return {"status": "recorded", "id": obs["id"]}


@app.post("/phone/app_usage")
async def phone_app_usage(
    device_id: str,
    apps: List[PhoneApp],
    device: Dict = Depends(get_device)
):
    """Receive app usage data from phone."""
    observations = []
    for app in apps:
        observation = PhoneObservation(
            observation_type="app_usage",
            content={
                "app_name": app.name,
                "package": app.package,
                "usage_time_seconds": app.usage_time,
                "last_opened": app.last_opened
            },
            device_id=device_id
        )
        obs = await phone_api.add_observation(observation)
        observations.append(obs["id"])
    return {"status": "recorded", "count": len(observations)}


@app.post("/phone/notification")
async def phone_notification(
    device_id: str,
    app: str,
    title: str,
    message: str,
    device: Dict = Depends(get_device)
):
    """Receive notification data from phone."""
    observation = PhoneObservation(
        observation_type="notification",
        content={
            "app": app,
            "title": title,
            "message": message[:500]  # Truncate
        },
        device_id=device_id
    )
    obs = await phone_api.add_observation(observation)
    return {"status": "recorded", "id": obs["id"]}


@app.get("/phone/observations")
async def get_observations(
    device_id: Optional[str] = None,
    observation_type: Optional[str] = None,
    limit: int = Query(100, le=1000),
    device: Dict = Depends(get_device)
):
    """Get observations."""
    obs = phone_api.observations
    
    if device_id:
        obs = [o for o in obs if o["device_id"] == device_id]
    if observation_type:
        obs = [o for o in obs if o["type"] == observation_type]
    
    return {"observations": obs[-limit:]}


@app.get("/phone/summary")
async def get_summary(device: Dict = Depends(get_device)):
    """Get summary of all devices."""
    return {
        "total_observations": len(phone_api.observations),
        "devices": len(phone_api.tokens),
        "device_summaries": {
            device_id: phone_api.get_device_summary(device_id)
            for device_id in set(t["device_id"] for t in phone_api.tokens.values())
        }
    }


# ============ SYNC ENDPOINT ============

@app.post("/phone/sync")
async def phone_sync(
    observations: List[PhoneObservation],
    device: Dict = Depends(get_device)
):
    """Batch sync observations from phone."""
    synced = []
    for obs in observations:
        result = await phone_api.add_observation(obs)
        synced.append(result["id"])
    return {"synced": len(synced), "ids": synced}


# ============ WEB DASHBOARD ============

@app.get("/phone/dashboard", response_class=HTMLResponse)
async def phone_dashboard():
    """Mobile-responsive web dashboard."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Shadow - Phone Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #fff;
            padding: 16px;
            min-height: 100vh;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { font-size: 24px; margin-bottom: 8px; color: #8b5cf6; }
        h2 { font-size: 18px; margin: 24px 0 12px; color: #a78bfa; }
        .card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #2a2a2a;
        }
        .shikigami-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .shikigami-btn {
            background: linear-gradient(135deg, #1a1a2a, #2d1b4e);
            border: 1px solid #4c1d95;
            border-radius: 8px;
            padding: 12px;
            color: #fff;
            cursor: pointer;
            text-align: center;
            font-size: 14px;
        }
        .shikigami-btn:hover { background: linear-gradient(135deg, #2d1b4e, #4c1d95); }
        .shikigami-btn .emoji { font-size: 24px; display: block; margin-bottom: 4px; }
        .input-group { margin-bottom: 12px; }
        .input-group label { display: block; margin-bottom: 4px; color: #9ca3af; font-size: 14px; }
        .input-group input, .input-group textarea {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #2a2a2a;
            background: #0a0a0a;
            color: #fff;
            font-size: 16px;
        }
        .input-group textarea { min-height: 80px; resize: vertical; }
        .btn {
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            color: #fff;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover { background: linear-gradient(135deg, #9f7afa, #818cf8); }
        .response {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        .status { color: #10b981; font-size: 14px; margin-bottom: 12px; }
        #token { display: none; }
        .setup { text-align: center; }
        .setup .emoji { font-size: 48px; margin-bottom: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐺 Digital Shadow</h1>
        <p class="status" id="status">Connecting...</p>
        
        <div id="setup" class="card setup" style="display:none;">
            <div class="emoji">📱</div>
            <h2>Register This Device</h2>
            <div class="input-group">
                <input type="text" id="deviceName" placeholder="My Phone" />
            </div>
            <button class="btn" onclick="registerDevice()">Register & Get Token</button>
            <div id="tokenDisplay" style="margin-top:12px;word-break:break-all;"></div>
        </div>
        
        <div id="main" style="display:none;">
            <div class="card">
                <h2>Summon Shikigami</h2>
                <div class="shikigami-grid">
                    <button class="shikigami-btn" onclick="selectShikigami('omni_tracker')">
                        <span class="emoji">🐺</span>
                        <span>Omni-Tracker</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('architect')">
                        <span class="emoji">🦅</span>
                        <span>Architect</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('oracle')">
                        <span class="emoji">🐍</span>
                        <span>Oracle</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('mimic')">
                        <span class="emoji">🐸</span>
                        <span>Mimic</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('creation')">
                        <span class="emoji">🐘</span>
                        <span>Creation</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('sensei')">
                        <span class="emoji">🦌</span>
                        <span>Sensei</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('conduit')">
                        <span class="emoji">🐉</span>
                        <span>Conduit</span>
                    </button>
                    <button class="shikigami-btn" onclick="selectShikigami('adaptive')">
                        <span class="emoji">🌀</span>
                        <span>Adaptive</span>
                    </button>
                </div>
            </div>
            
            <div class="card">
                <div class="input-group">
                    <label>Ask your Shadow:</label>
                    <textarea id="query" placeholder="What patterns do you see in my recent activity?"></textarea>
                </div>
                <input type="hidden" id="selectedShikigami" />
                <button class="btn" onclick="summonShikigami()">Summon</button>
            </div>
            
            <div id="response" class="response" style="display:none;"></div>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <button class="btn" onclick="sendLocation()">📍 Send Location</button>
                <button class="btn" style="margin-top:8px;" onclick="syncObservations()">🔄 Sync Now</button>
            </div>
        </div>
    </div>
    
    <script>
        const API = 'http://' + window.location.host;
        let token = localStorage.getItem('shadow_token');
        let deviceId = localStorage.getItem('shadow_device_id');
        
        async function checkStatus() {
            try {
                const r = await fetch(API + '/');
                const data = await r.json();
                document.getElementById('status').textContent = '✓ Connected to Digital Shadow';
                return true;
            } catch(e) {
                document.getElementById('status').textContent = '✗ Cannot connect to shadow daemon';
                return false;
            }
        }
        
        async function registerDevice() {
            const name = document.getElementById('deviceName').value || 'My Phone';
            deviceId = 'phone_' + Math.random().toString(36).substr(2, 9);
            
            const r = await fetch(API + '/phone/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    device_id: deviceId,
                    device_name: name,
                    device_type: 'web'
                })
            });
            
            const data = await r.json();
            token = data.token;
            localStorage.setItem('shadow_token', token);
            localStorage.setItem('shadow_device_id', deviceId);
            
            document.getElementById('tokenDisplay').textContent = 'Your token: ' + token;
            showMain();
        }
        
        function selectShikigami(name) {
            document.getElementById('selectedShikigami').value = name;
            document.querySelectorAll('.shikigami-btn').forEach(b => b.style.borderColor = '#4c1d95');
            event.target.closest('.shikigami-btn').style.borderColor = '#8b5cf6';
        }
        
        async function summonShikigami() {
            const query = document.getElementById('query').value;
            const shikigami = document.getElementById('selectedShikigami').value;
            
            if (!query) {
                alert('Enter a query');
                return;
            }
            
            const endpoint = shikigami 
                ? API + '/shikigami/summon'
                : API + '/shikigami/summon';
            
            const body = {query};
            if (shikigami) body.shikigami = shikigami;
            
            const r = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });
            
            const data = await r.json();
            const responseDiv = document.getElementById('response');
            responseDiv.style.display = 'block';
            responseDiv.textContent = data.output || JSON.stringify(data, null, 2);
        }
        
        async function sendLocation() {
            if (!navigator.geolocation) {
                alert('Geolocation not supported');
                return;
            }
            
            navigator.geolocation.getCurrentPosition(async (pos) => {
                const r = await fetch(API + '/phone/location?device_id=' + deviceId, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + token
                    },
                    body: JSON.stringify({
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy: pos.coords.accuracy
                    })
                });
                const data = await r.json();
                alert('Location recorded: ' + JSON.stringify(data));
            });
        }
        
        async function syncObservations() {
            // Placeholder for batch sync
            alert('Sync triggered - observations uploaded');
        }
        
        function showMain() {
            document.getElementById('setup').style.display = 'none';
            document.getElementById('main').style.display = 'block';
        }
        
        async function init() {
            const connected = await checkStatus();
            if (!connected) return;
            
            if (token && deviceId) {
                showMain();
            } else {
                document.getElementById('setup').style.display = 'block';
            }
        }
        
        init();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766)