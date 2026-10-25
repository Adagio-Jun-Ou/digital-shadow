#!/usr/bin/env python3
"""
Phone Data Exporter - Export phone data for analysis

Exports:
- Location history
- Bluetooth devices
- WiFi networks
- Communication patterns
- Battery usage
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import csv

class PhoneDataExporter:
    """Export phone data for external analysis."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "phone")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def export_location_history(self, observations: List[Dict], format: str = "json") -> Path:
        """Export location history."""
        locations = [o for o in observations if o.get("type") == "location"]
        
        filename = self.data_dir / f"location_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == "json":
            filename.write_text(json.dumps(locations, indent=2, default=str))
        elif format == "csv":
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "latitude", "longitude", "accuracy"])
                for loc in locations:
                    writer.writerow([
                        loc.get("timestamp"),
                        loc.get("latitude"),
                        loc.get("longitude"),
                        loc.get("accuracy")
                    ])
        
        return filename
    
    def export_bluetooth_devices(self, observations: List[Dict]) -> Path:
        """Export Bluetooth device history."""
        bt_scans = [o for o in observations if o.get("type") == "bluetooth_scan"]
        
        devices = {}
        for scan in bt_scans:
            for device in scan.get("nearby_devices", []):
                mac = device.get("mac")
                if mac and mac not in devices:
                    devices[mac] = {
                        **device,
                        "first_seen": scan.get("timestamp"),
                        "times_seen": 1
                    }
                elif mac:
                    devices[mac]["times_seen"] += 1
        
        filename = self.data_dir / f"bluetooth_devices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(devices, indent=2, default=str))
        
        return filename
    
    def export_wifi_networks(self, observations: List[Dict]) -> Path:
        """Export WiFi network history."""
        wifi_scans = [o for o in observations if o.get("type") == "wifi_scan"]
        
        networks = {}
        for scan in wifi_scans:
            for network in scan.get("nearby_networks", []):
                ssid = network.get("ssid")
                if ssid and ssid not in networks:
                    networks[ssid] = {
                        **network,
                        "first_seen": scan.get("timestamp"),
                        "times_seen": 1
                    }
                elif ssid:
                    networks[ssid]["times_seen"] += 1
        
        filename = self.data_dir / f"wifi_networks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(networks, indent=2, default=str))
        
        return filename
    
    def export_communication_patterns(self, communications: Dict) -> Path:
        """Export communication patterns."""
        filename = self.data_dir / f"communication_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(communications, indent=2, default=str))
        return filename
    
    def export_summary_report(self, summary: Dict) -> Path:
        """Export comprehensive summary report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary
        }
        
        filename = self.data_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(report, indent=2, default=str))
        
        return filename
    
    def cleanup_old_exports(self, days: int = 30):
        """Delete exports older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        deleted = 0
        for file in self.data_dir.glob("*.json"):
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            if file_time < cutoff:
                file.unlink()
                deleted += 1
        
        return deleted

if __name__ == "__main__":
    # Test exporter
    exporter = PhoneDataExporter()
    
    # Test data
    test_observations = [
        {"type": "location", "latitude": 37.7749, "longitude": -122.4194, "timestamp": "2026-04-04T00:00:00"},
        {"type": "bluetooth_scan", "nearby_devices": [{"mac": "AA:BB:CC", "name": "iPhone"}], "timestamp": "2026-04-04T00:01:00"}
    ]
    
    # Export
    loc_file = exporter.export_location_history(test_observations)
    print(f"Exported locations to: {loc_file}")
    
    bt_file = exporter.export_bluetooth_devices(test_observations)
    print(f"Exported Bluetooth to: {bt_file}")
    
    print("Data exporter ready")
