#!/usr/bin/env python3
"""
Digital Shadow Sensor Hub - Collect physical world data.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime
import httpx

logger = logging.getLogger("shadowd.sensors")

SHADOW_API = "http://127.0.0.1:8765"


class SensorHub:
    """
    Collects physical world data via various sensors.
    """
    
    def __init__(self):
        self.api_endpoint = SHADOW_API
        self.active_sensors: dict = {}
        
    async def start_location_tracking(self):
        """Track location via WiFi triangulation or GPS if available."""
        while True:
            try:
                location = await self._get_location()
                if location:
                    await self._send_observation("location", location)
            except Exception as e:
                logger.debug(f"Location tracking error: {e}")
            await asyncio.sleep(60)
    
    async def _get_location(self) -> Optional[dict]:
        """
        Get location using available methods.
        Priority: GPS > WiFi triangulation > IP geolocation
        """
        location = await self._try_gps()
        if not location:
            location = await self._try_wifi_location()
        if not location:
            location = await self._try_ip_location()
        return location
    
    async def _try_gps(self) -> Optional[dict]:
        """Try to get GPS location if GPS hardware available."""
        try:
            import gpsd
            gpsd.connect()
            packet = gpsd.get_current()
            return {
                "lat": packet.lat,
                "lon": packet.lon,
                "method": "gps",
                "accuracy": packet.haccuracy if hasattr(packet, 'haccuracy') else None
            }
        except Exception:
            return None
    
    async def _try_wifi_location(self) -> Optional[dict]:
        """Use WiFi triangulation for location."""
        return None
    
    async def _try_ip_location(self) -> Optional[dict]:
        """Fallback to IP-based location."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://ipapi.co/json/")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "lat": data.get("latitude"),
                        "lon": data.get("longitude"),
                        "city": data.get("city"),
                        "country": data.get("country_name"),
                        "method": "ip"
                    }
        except Exception:
            pass
        return None
    
    async def start_motion_tracking(self):
        """Track motion/activity if accelerometer/gyroscope available."""
        pass
    
    async def start_health_tracking(self):
        """Connect to health devices via Bluetooth."""
        pass
    
    async def start_time_tracking(self):
        """Track time-based patterns."""
        while True:
            now = datetime.now()
            await self._send_observation("time_context", {
                "hour": now.hour,
                "weekday": now.strftime("%A"),
                "is_workday": now.weekday() < 5,
            })
            await asyncio.sleep(300)
    
    async def _send_observation(self, obs_type: str, content: dict):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.api_endpoint}/observe",
                    json={
                        "activity_type": obs_type,
                        "content": content,
                        "device": "sensors"
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to send observation: {e}")
    
    async def start_all(self):
        """Start all sensor tracking."""
        tasks = [
            self.start_location_tracking(),
            self.start_motion_tracking(),
            self.start_health_tracking(),
            self.start_time_tracking(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    hub = SensorHub()
    await hub.start_all()


if __name__ == "__main__":
    asyncio.run(main())