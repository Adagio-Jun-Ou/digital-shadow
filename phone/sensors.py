#!/usr/bin/env python3
"""
Phone Sensor Integration for Digital Shadow
Real superhuman abilities through phone sensors:
- Bluetooth scanning for presence detection
- WiFi scanning for location awareness  
- GPS for location tracking
- Accelerometer for activity detection
- Audio analysis for environment sensing
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging
import subprocess

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logger = logging.getLogger("shadowd.phone_sensors")

DATA_DIR = Path(__file__).parent.parent / "data"
SENSORS_DIR = DATA_DIR / "sensors"
SENSORS_DIR.mkdir(parents=True, exist_ok=True)

# ============ SENSOR DATA MODELS ============

class BluetoothScan(BaseModel):
    timestamp: str
    devices: List[Dict]
    total: int

class WiFiScan(BaseModel):
    timestamp: str
    networks: List[Dict]
    total: int

class LocationUpdate(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    timestamp: str

class ActivityData(BaseModel):
    activity_type: str  # walking, running, still, vehicle
    confidence: float
    timestamp: str

class AudioLevel(BaseModel):
    level: float  # dB
    is_voice: bool
    timestamp: str


class PhoneSensorHub:
    """
    Aggregates all phone sensors for superhuman sensing.
    """
    
    def __init__(self, shadow_api: str = "http://127.0.0.1:8765"):
        self.shadow_api = shadow_api
        self.bluetooth_history: List[BluetoothScan] = []
        self.wifi_history: List[WiFiScan] = []
        self.location_history: List[LocationUpdate] = []
        self.activity_history: List[ActivityData] = []
        self.audio_history: List[AudioLevel] = []
        
        # Entity tracking
        self.known_devices: Dict[str, Dict] = {}  # MAC -> device info
        self.known_networks: Dict[str, Dict] = {}  # SSID -> network info
        
        # Pattern detection
        self.presence_patterns: Dict[str, List] = {}  # device_id -> [timestamps]
        self.location_patterns: Dict[str, List] = {}  # location -> [timestamps]
    
    async def process_bluetooth_scan(self, scan: BluetoothScan):
        """Process Bluetooth scan and detect presence patterns."""
        
        # Store history
        self.bluetooth_history.append(scan)
        if len(self.bluetooth_history) > 1000:
            self.bluetooth_history = self.bluetooth_history[-1000:]
        
        # Update known devices
        for device in scan.devices:
            mac = device.get("mac")
            if mac:
                if mac not in self.known_devices:
                    self.known_devices[mac] = {
                        **device,
                        "first_seen": scan.timestamp,
                        "last_seen": scan.timestamp,
                        "appearance_count": 1
                    }
                else:
                    self.known_devices[mac].update({
                        "last_seen": scan.timestamp,
                        "appearance_count": self.known_devices[mac].get("appearance_count", 0) + 1,
                        **device
                    })
                
                # Track presence pattern
                if mac not in self.presence_patterns:
                    self.presence_patterns[mac] = []
                self.presence_patterns[mac].append(scan.timestamp)
        
        # Detect patterns
        await self._detect_presence_patterns()
        
        # Forward to shadow
        await self._forward_to_shadow("bluetooth_scan", {
            "devices": scan.devices,
            "total": scan.total,
            "known_count": len(self.known_devices)
        })
    
    async def process_wifi_scan(self, scan: WiFiScan):
        """Process WiFi scan for location inference."""
        
        self.wifi_history.append(scan)
        if len(self.wifi_history) > 1000:
            self.wifi_history = self.wifi_history[-1000:]
        
        # Update known networks
        for network in scan.networks:
            ssid = network.get("ssid", "hidden")
            if ssid not in self.known_networks:
                self.known_networks[ssid] = {
                    **network,
                    "first_seen": scan.timestamp,
                    "last_seen": scan.timestamp,
                    "appearance_count": 1
                }
            else:
                self.known_networks[ssid].update({
                    "last_seen": scan.timestamp,
                    "appearance_count": self.known_networks[ssid].get("appearance_count", 0) + 1
                })
        
        # Infer location from SSIDs
        location = await self._infer_location_from_wifi(scan.networks)
        
        await self._forward_to_shadow("wifi_scan", {
            "networks": scan.networks,
            "total": scan.total,
            "inferred_location": location
        })
    
    async def process_location(self, location: LocationUpdate):
        """Process GPS location update."""
        
        self.location_history.append(location)
        if len(self.location_history) > 1000:
            self.location_history = self.location_history[-1000:]
        
        # Detect location patterns (home, work, etc.)
        loc_key = f"{location.latitude:.4f},{location.longitude:.4f}"
        if loc_key not in self.location_patterns:
            self.location_patterns[loc_key] = []
        self.location_patterns[loc_key].append(location.timestamp)
        
        # Label frequent locations
        if len(self.location_patterns[loc_key]) > 10:
            # This is a frequent location
            hours = [datetime.fromisoformat(t).hour for t in self.location_patterns[loc_key]]
            avg_hour = sum(hours) / len(hours)
            if 8 <= avg_hour <= 18:
                label = "work"
            elif 22 <= avg_hour or avg_hour <= 6:
                label = "home"
            else:
                label = "frequent"
            
            self.location_patterns[loc_key + "_label"] = label
        
        await self._forward_to_shadow("location", {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "accuracy": location.accuracy,
            "location_label": self.location_patterns.get(f"{loc_key}_label", "unknown")
        })
    
    async def process_activity(self, activity: ActivityData):
        """Process activity detection (walking, running, still)."""
        
        self.activity_history.append(activity)
        if len(self.activity_history) > 1000:
            self.activity_history = self.activity_history[-1000:]
        
        await self._forward_to_shadow("activity", {
            "activity_type": activity.activity_type,
            "confidence": activity.confidence
        })
    
    async def _detect_presence_patterns(self):
        """Detect presence patterns from Bluetooth history."""
        
        # Pattern: Device X appears at certain times = recurring presence
        for mac, timestamps in self.presence_patterns.items():
            if len(timestamps) < 3:
                continue
            
            hours = [datetime.fromisoformat(t).hour for t in timestamps[-10:]]
            if len(set(hours)) == 1:
                # Always appears at same hour
                pattern = f"Device {mac} appears at hour {hours[0]}"
                await self._forward_to_shadow("presence_pattern", {
                    "device": mac,
                    "pattern": "recurring_hour",
                    "hour": hours[0]
                })
    
    async def _infer_location_from_wifi(self, networks: List[Dict]) -> str:
        """Infer location from WiFi network names."""
        
        known_locations = {
            # Add known SSIDs and their locations
            # "HomeWiFi": "home",
            # "OfficeWiFi": "work",
        }
        
        for network in networks:
            ssid = network.get("ssid", "")
            if ssid in known_locations:
                return known_locations[ssid]
        
        # Use number of networks as location type hint
        if len(networks) > 20:
            return "urban"  # Dense area
        elif len(networks) > 5:
            return "suburban"
        else:
            return "remote"
    
    async def _forward_to_shadow(self, event_type: str, data: Dict):
        """Forward sensor data to main shadow daemon."""
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.shadow_api}/observe",
                    json={
                        "activity_type": f"phone_{event_type}",
                        "content": data,
                        "device": "phone"
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to forward to shadow: {e}")
    
    def get_presence_summary(self) -> Dict:
        """Get summary of detected presences."""
        return {
            "known_devices": len(self.known_devices),
            "known_networks": len(self.known_networks),
            "recent_devices": [
                {"name": d.get("name"), "last_seen": d.get("last_seen")}
                for d in list(self.known_devices.values())[-10:]
            ],
            "frequent_locations": [
                {"location": k, "label": self.location_patterns.get(k + "_label", "unknown"), "visits": len(v)}
                for k, v in self.location_patterns.items()
                if not k.endswith("_label") and len(v) > 5
            ]
        }


# ============ FASTAPI ENDPOINTS ============

sensor_hub = PhoneSensorHub()

SENSOR_CORS_ORIGINS = os.environ.get("SENSOR_CORS_ORIGINS", "http://localhost:8765,http://localhost:3000").split(",")

app = FastAPI(title="Digital Shadow Phone Sensors")

app.add_middleware(
    CORSMiddleware,
    allow_origins=SENSOR_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sensors/bluetooth")
async def receive_bluetooth(scan: BluetoothScan):
    """Receive Bluetooth scan from phone."""
    await sensor_hub.process_bluetooth_scan(scan)
    return {"status": "processed", "devices": scan.total}


@app.post("/sensors/wifi")
async def receive_wifi(scan: WiFiScan):
    """Receive WiFi scan from phone."""
    await sensor_hub.process_wifi_scan(scan)
    return {"status": "processed", "networks": scan.total}


@app.post("/sensors/location")
async def receive_location(location: LocationUpdate):
    """Receive GPS location from phone."""
    await sensor_hub.process_location(location)
    return {"status": "processed"}


@app.post("/sensors/activity")
async def receive_activity(activity: ActivityData):
    """Receive activity detection from phone."""
    await sensor_hub.process_activity(activity)
    return {"status": "processed"}


@app.get("/sensors/presence")
async def get_presence():
    """Get presence summary."""
    return sensor_hub.get_presence_summary()


@app.get("/sensors/status")
async def get_status():
    """Get sensor status."""
    return {
        "bluetooth_scans": len(sensor_hub.bluetooth_history),
        "wifi_scans": len(sensor_hub.wifi_history),
        "location_updates": len(sensor_hub.location_history),
        "known_devices": len(sensor_hub.known_devices),
        "known_networks": len(sensor_hub.known_networks)
    }


# ============ ANDROID INTENT HANDLERS ============

@app.post("/android/bluetooth")
async def android_bluetooth():
    """Handle Android Bluetooth intent."""
    # Scan for Bluetooth devices
    result = subprocess.run(
        "bluetoothctl devices 2>/dev/null || echo ''",
        shell=True, capture_output=True, text=True
    )
    
    devices = []
    for line in result.stdout.strip().split('\n'):
        if line.startswith('Device'):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                devices.append({
                    "mac": parts[1],
                    "name": parts[2],
                    "type": "bluetooth"
                })
    
    scan = BluetoothScan(
        timestamp=datetime.utcnow().isoformat(),
        devices=devices,
        total=len(devices)
    )
    await sensor_hub.process_bluetooth_scan(scan)
    return {"scanned": len(devices)}


@app.post("/android/wifi")
async def android_wifi():
    """Handle Android WiFi intent."""
    result = subprocess.run(
        "nmcli -t -f SSID,SIGNAL dev wifi list 2>/dev/null || iwlist scanning 2>/dev/null | grep ESSID || echo ''",
        shell=True, capture_output=True, text=True
    )
    
    networks = []
    for line in result.stdout.strip().split('\n'):
        if ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                networks.append({
                    "ssid": parts[0] if parts[0] else "hidden",
                    "signal": parts[1] if len(parts) > 1 else "unknown",
                    "type": "wifi"
                })
    
    scan = WiFiScan(
        timestamp=datetime.utcnow().isoformat(),
        networks=networks,
        total=len(networks)
    )
    await sensor_hub.process_wifi_scan(scan)
    return {"scanned": len(networks)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8767)