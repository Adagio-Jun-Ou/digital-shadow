#!/usr/bin/env python3
"""
Shadow Phone Integration - Connects phone sensors to Shadow AI

Integrates:
- Phone API (observations)
- Bluetooth scanner (presence)
- WiFi scanner (location)
- Communication observer (patterns)
- Shadow AI unified system
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.unified_system import create_unified_system
from phone.bluetooth_scanner import BluetoothScanner
from phone.wifi_scanner import WiFiScanner  
from phone.communication_observer import CommunicationObserver

logger = __import__('logging').getLogger(__name__)

class ShadowPhoneIntegration:
    """
    Integrates phone sensors with Shadow AI.
    
    Flow:
    1. Phone sends sensor data to API
    2. Integration module processes
    3. Routes to appropriate Shadow AI ability
    4. Stores in unified memory
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "phone")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Shadow AI system
        self.shadow = create_unified_system(self.data_dir)
        
        # Scanners
        self.bluetooth = BluetoothScanner()
        self.wifi = WiFiScanner()
        self.communications = CommunicationObserver()
        
        # Registered devices
        self.devices: Dict[str, Dict] = {}
        
        logger.info("Shadow Phone Integration initialized")
    
    async def handle_location_update(self, device_id: str, location: Dict):
        """
        Handle location update from phone.
        
        Routes to:
        - Omni Sense (environmental awareness)
        - X-Ray (location detection)
        """
        observation = {
            "type": "location",
            "device_id": device_id,
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "accuracy": location.get("accuracy", 10),
            "timestamp": location.get("timestamp", datetime.now().isoformat())
        }
        
        await self.shadow.observe(observation)
        
        logger.info(f"Location update from {device_id}: ({location['latitude']}, {location['longitude']})")
        
        return {"status": "processed", "type": "location"}
    
    async def handle_bluetooth_scan(self, device_id: str, scan: Dict):
        """
        Handle Bluetooth scan from phone.
        
        Routes to:
        - Life Pulse (person identification)
        - X-Ray (see who's nearby)
        """
        devices = scan.get("devices", [])
        
        # Identify present people
        present = self.bluetooth.identify_present_devices(
            [self.bluetooth.BluetoothScanner.BluetoothDevice(**d) if hasattr(self.bluetooth, 'BluetoothDevice') else d 
             for d in devices]
        )
        
        observation = {
            "type": "bluetooth_scan",
            "device_id": device_id,
            "nearby_devices": devices,
            "identified_people": present,
            "timestamp": scan.get("timestamp", datetime.now().isoformat())
        }
        
        await self.shadow.observe(observation)
        
        logger.info(f"Bluetooth scan from {device_id}: {len(devices)} devices, {len(present)} identified")
        
        return {"status": "processed", "devices": len(devices), "identified": len(present)}
    
    async def handle_wifi_scan(self, device_id: str, scan: Dict):
        """
        Handle WiFi scan from phone.
        
        Routes to:
        - X-Ray (location detection)
        - Omni Sense (environmental context)
        """
        networks = scan.get("networks", [])
        
        # Detect location
        wifi_networks = [
            WiFiNetwork(ssid=n.get("ssid"), bssid=n.get("bssid"), 
                       rssi=n.get("rssi"), channel=n.get("channel", 1),
                       security=n.get("security", "unknown"))
            for n in networks
        ]
        
        location = self.wifi.detect_location(wifi_networks)
        
        observation = {
            "type": "wifi_scan",
            "device_id": device_id,
            "nearby_networks": networks,
            "detected_location": location,
            "is_home": self.wifi.is_home(wifi_networks),
            "is_work": self.wifi.is_work(wifi_networks),
            "timestamp": scan.get("timestamp", datetime.now().isoformat())
        }
        
        await self.shadow.observe(observation)
        
        logger.info(f"WiFi scan from {device_id}: {len(networks)} networks, location={location}")
        
        return {"status": "processed", "networks": len(networks), "location": location}
    
    async def handle_call(self, device_id: str, call: Dict):
        """
        Handle call metadata from phone.
        
        Routes to:
        - Social Radar (predicting calls)
        - Pattern Sight (communication patterns)
        """
        from phone.communication_observer import CallMetadata
        
        call_meta = CallMetadata(
            contact_id=call.get("contact_id"),
            contact_name=call.get("contact_name"),
            phone_number=call.get("phone_number"),
            call_type=call.get("call_type"),
            duration_seconds=call.get("duration_seconds", 0),
            timestamp=datetime.fromisoformat(call.get("timestamp", datetime.now().isoformat())),
            device_id=device_id
        )
        
        self.communications.observe_call(call_meta)
        
        observation = {
            "type": "call",
            "device_id": device_id,
            "contact": call.get("contact_name", "unknown"),
            "call_type": call.get("call_type"),
            "duration": call.get("duration_seconds", 0),
            "timestamp": call.get("timestamp")
        }
        
        await self.shadow.observe(observation)
        
        # Check for prediction
        prediction = self.communications.predict_next_call(call.get("contact_id"))
        
        return {
            "status": "processed",
            "type": "call",
            "prediction": prediction
        }
    
    async def handle_message(self, device_id: str, message: Dict):
        """
        Handle message metadata from phone (privacy-safe, no content).
        
        Routes to:
        - Social Radar (message patterns)
        """
        from phone.communication_observer import MessageMetadata
        
        msg_meta = MessageMetadata(
            app=message.get("app"),
            contact_id=message.get("contact_id"),
            message_type=message.get("message_type", "text"),
            timestamp=datetime.fromisoformat(message.get("timestamp", datetime.now().isoformat())),
            device_id=device_id
        )
        
        self.communications.observe_message(msg_meta)
        
        observation = {
            "type": "message",
            "device_id": device_id,
            "app": message.get("app"),
            "contact": message.get("contact_id"),
            "timestamp": message.get("timestamp")
        }
        
        await self.shadow.observe(observation)
        
        return {"status": "processed", "type": "message"}
    
    def register_device(self, device_id: str, device_info: Dict):
        """Register a device."""
        self.devices[device_id] = {
            **device_info,
            "registered": datetime.now().isoformat()
        }
        logger.info(f"Registered device: {device_id}")
    
    def get_device_status(self, device_id: str) -> Optional[Dict]:
        """Get device status."""
        return self.devices.get(device_id)
    
    def get_communication_summary(self, hours: int = 24) -> Dict:
        """Get communication summary."""
        return self.communications.get_communication_summary(hours)
    
    def get_top_contacts(self, limit: int = 10) -> list:
        """Get top contacts."""
        return self.communications.get_top_contacts(limit)
    
    def get_shadow_status(self) -> Dict:
        """Get Shadow AI status."""
        return self.shadow.get_unified_status()


# FastAPI endpoints for integration
def create_integration_app():
    """Create FastAPI app for Shadow Phone Integration."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Shadow Phone Integration", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    
    integration = None
    
    @app.on_event("startup")
    async def startup():
        nonlocal integration
        integration = ShadowPhoneIntegration()
    
    @app.post("/integrate/location")
    async def integrate_location(device_id: str, location: Dict):
        return await integration.handle_location_update(device_id, location)
    
    @app.post("/integrate/bluetooth")
    async def integrate_bluetooth(device_id: str, scan: Dict):
        return await integration.handle_bluetooth_scan(device_id, scan)
    
    @app.post("/integrate/wifi")
    async def integrate_wifi(device_id: str, scan: Dict):
        return await integration.handle_wifi_scan(device_id, scan)
    
    @app.post("/integrate/call")
    async def integrate_call(device_id: str, call: Dict):
        return await integration.handle_call(device_id, call)
    
    @app.post("/integrate/message")
    async def integrate_message(device_id: str, message: Dict):
        return await integration.handle_message(device_id, message)
    
    @app.get("/status")
    async def get_status():
        return integration.get_shadow_status()
    
    @app.get("/communications/summary")
    async def get_comm_summary(hours: int = 24):
        return integration.get_communication_summary(hours)
    
    @app.get("/communications/contacts")
    async def get_top_contacts(limit: int = 10):
        return integration.get_top_contacts(limit)
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_integration_app()
    uvicorn.run(app, host="0.0.0.0", port=8767)
