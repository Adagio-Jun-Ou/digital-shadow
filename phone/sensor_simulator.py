#!/usr/bin/env python3
"""
Sensor Simulator for Testing Shadow AI Phone Integration

Simulates phone sensors for development and testing.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List
import json

class PhoneSensorSimulator:
    """Simulates phone sensor data."""
    
    def __init__(self, device_id: str = "simulator-001"):
        self.device_id = device_id
        self.running = False
        self.location_center = (37.7749, -122.4194)  # San Francisco
        self.location_variance = 0.001
        self.activity_types = ["walking", "running", "stationary", "driving", "cycling"]
    
    def simulate_location(self) -> Dict:
        """Generate simulated GPS location."""
        lat = self.location_center[0] + random.uniform(-self.location_variance, self.location_variance)
        lon = self.location_center[1] + random.uniform(-self.location_variance, self.location_variance)
        
        return {
            "latitude": lat,
            "longitude": lon,
            "accuracy": random.uniform(5.0, 20.0),
            "altitude": random.uniform(0, 100),
            "speed": random.uniform(0, 30),
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id
        }
    
    def simulate_motion(self) -> Dict:
        """Generate simulated motion/accelerometer data."""
        activity = random.choice(self.activity_types)
        
        return {
            "acceleration_x": random.uniform(-2, 2),
            "acceleration_y": random.uniform(-2, 2),
            "acceleration_z": random.uniform(9, 11),  # Gravity ~9.8
            "activity_type": activity,
            "confidence": random.uniform(0.7, 0.99),
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id
        }
    
    def simulate_battery(self) -> Dict:
        """Generate simulated battery data."""
        return {
            "level": random.randint(20, 100),
            "charging": random.choice([True, False]),
            "battery_health": random.choice(["good", "fair", "degraded"]),
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id
        }
    
    def simulate_app_usage(self) -> Dict:
        """Generate simulated app usage data."""
        apps = [
            ("com.google.apps.maps", "navigation", "Maps"),
            ("com.whatsapp", "communication", "WhatsApp"),
            ("com.instagram", "social", "Instagram"),
            ("com.spotify", "entertainment", "Spotify"),
            ("vscode", "productivity", "VS Code")
        ]
        
        app = random.choice(apps)
        
        return {
            "app_name": app[2],
            "app_category": app[1],
            "duration_seconds": random.randint(30, 1800),
            "foreground": True,
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id
        }
    
    def simulate_bluetooth_scan(self) -> Dict:
        """Generate simulated Bluetooth scan."""
        nearby_devices = [
            {"mac": "AA:BB:CC:11:22:33", "name": "iPhone", "rssi": -60},
            {"mac": "AA:BB:CC:44:55:66", "name": "Watch", "rssi": -70},
            {"mac": "AA:BB:CC:77:88:99", "name": "Headphones", "rssi": -80}
        ]
        
        # Random subset
        devices = random.sample(nearby_devices, k=random.randint(0, len(nearby_devices)))
        
        return {
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            "devices": devices
        }
    
    def simulate_wifi_scan(self) -> Dict:
        """Generate simulated WiFi scan."""
        nearby_networks = [
            {"ssid": "HomeWiFi", "bssid": "AA:BB:CC:11:22:33", "rssi": -50, "security": "WPA2"},
            {"ssid": "HomeWiFi_5G", "bssid": "AA:BB:CC:11:22:34", "rssi": -55, "security": "WPA2"},
            {"ssid": "Neighbor_Net", "bssid": "AA:BB:CC:44:55:66", "rssi": -80, "security": "WPA2"}
        ]
        
        networks = random.sample(nearby_networks, k=random.randint(1, len(nearby_networks)))
        
        return {
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            "networks": networks
        }
    
    async def run_continuous(self, interval: int = 60):
        """
        Run continuous simulation.
        
        Args:
            interval: Seconds between updates
        """
        self.running = True
        print(f"Starting phone simulator ({self.device_id}), interval={interval}s")
        
        while self.running:
            # Simulate all sensors
            location = self.simulate_location()
            motion = self.simulate_motion()
            battery = self.simulate_battery()
            app = self.simulate_app_usage()
            bluetooth = self.simulate_bluetooth_scan()
            wifi = self.simulate_wifi_scan()
            
            data = {
                "location": location,
                "motion": motion,
                "battery": battery,
                "app_usage": app,
                "bluetooth": bluetooth,
                "wifi": wifi
            }
            
            print(json.dumps(data, indent=2))
            print("-" * 50)
            
            await asyncio.sleep(interval)
    
    def stop(self):
        """Stop the simulation."""
        self.running = False


async def main():
    """Run sensor simulator."""
    simulator = PhoneSensorSimulator("test-phone-simulator")
    
    try:
        await simulator.run_continuous(interval=10)
    except KeyboardInterrupt:
        print("\nStopping simulator...")
        simulator.stop()


if __name__ == "__main__":
    asyncio.run(main())