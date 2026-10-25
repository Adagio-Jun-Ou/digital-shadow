#!/usr/bin/env python3
"""
Digital Shadow Sync - Tailscale-based device mesh synchronization.
"""

import asyncio
import logging
import os
import re
import httpx
from datetime import datetime
from typing import Optional

logger = logging.getLogger("shadowd.sync")

SHADOW_API = os.environ.get("SHADOW_API", "http://127.0.0.1:8765")
SHADOW_API_KEY = os.environ.get("SHADOW_API_KEY", "")
DEVICE_NAME = os.environ.get("DEVICE_NAME", "unknown")
_MAX_NAME_LEN = 64


_HOST_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,253}[a-zA-Z0-9]$')


class ShadowSync:
    """
    Syncs Digital Shadow across devices via Tailscale mesh.
    
    Architecture:
    - Laptop runs as the master node (Shadow Daemon)
    - Other devices connect via Tailscale and send observations
    - All data stays encrypted and local
    """
    
    def __init__(self, master_host: str = None):
        self.master_host = self._validate_host(master_host) if master_host else self._detect_master()
        self.device_id = self._get_device_id()
    
    @staticmethod
    def _validate_host(host: str) -> str:
        if not host or not _HOST_PATTERN.match(host):
            raise ValueError(f"Invalid master_host")
        return host
        
    def _detect_master(self) -> str:
        """Find master shadow node via Tailscale."""
        try:
            result = httpx.get("http://localhost:8765/status", timeout=2.0)
            if result.status_code == 200:
                return "127.0.0.1"
        except Exception:
            pass
        
        try:
            status = httpx.get("http://localhost:4404/api/local/bonus/hello", timeout=2.0)
            if status.status_code == 200:
                return status.json().get("Self", {}).get("HostName")
        except Exception:
            pass
        
        return None
        
    def _get_device_id(self) -> str:
        """Get unique device identifier."""
        device_id_file = os.path.expanduser("~/.config/digital-shadow/device_id")
        
        if os.path.exists(device_id_file):
            with open(device_id_file) as f:
                return f.read().strip()[:64]
        
        config_dir = os.path.dirname(device_id_file)
        os.makedirs(config_dir, exist_ok=True)
        
        import uuid
        device_id = str(uuid.uuid4())
        with open(device_id_file, "w") as f:
            f.write(device_id)
        return device_id
    
    @staticmethod
    def _validate_name(name: str) -> str:
        import re
        if not name or len(name) > _MAX_NAME_LEN:
            return "unknown"
        return re.sub(r'[^a-zA-Z0-9_-]', '', name) or "unknown"
    
    def _headers(self) -> dict:
        """Build common request headers."""
        h = {}
        key = os.environ.get("SHADOW_API_KEY", "")
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h
    
    async def register_device(self):
        """Register this device with the master."""
        if not self.master_host:
            logger.warning("No master host found, running in standalone mode")
            return False
            
        safe_name = self._validate_name(DEVICE_NAME)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"http://{self.master_host}:8765/device/connect",
                    params={
                        "device_id": self.device_id,
                        "name": safe_name
                    },
                    headers=self._headers()
                )
                response.raise_for_status()
                logger.info(f"Registered device: {self.device_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to register device: {type(e).__name__}")
            return False
    
    async def send_observation(self, observation: dict):
        """Send observation to master."""
        if not self.master_host:
            return False
        
        payload = observation.get("content", {})
        if isinstance(payload, str) and len(payload) > 65536:
            logger.warning("Observation payload too large, truncating")
            payload = payload[:65536]
        elif isinstance(payload, (dict, list)):
            try:
                import json as _json
                if len(_json.dumps(payload)) > 65536:
                    logger.warning("Observation payload too large, rejecting")
                    return False
            except (TypeError, ValueError):
                return False
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"http://{self.master_host}:8765/observe",
                    json={
                        "activity_type": observation.get("type"),
                        "content": payload,
                        "device": self._validate_name(DEVICE_NAME)
                    },
                    headers=self._headers()
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Failed to send observation: {type(e).__name__}")
            return False
    
    async def summon(self, query: str) -> str:
        """Forward summon request to master."""
        if not self.master_host:
            return "Error: No master shadow node available"
        
        if len(query) > 10000:
            return "Error: Query too long"
            
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"http://{self.master_host}:8765/summon",
                    json={"query": query},
                    headers=self._headers()
                )
                response.raise_for_status()
                return response.json().get("response")
        except Exception as e:
            logger.error(f"Error summoning shadow: {type(e).__name__}")
            return "Error: Failed to contact shadow daemon"


async def main():
    import argparse
    argparse.ArgumentParser(description="Digital Shadow Sync").parse_args()
    
    sync = ShadowSync()
    if await sync.register_device():
        print(f"Connected to Digital Shadow at {sync.master_host}")
        while True:
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())