#!/usr/bin/env python3
"""
Example Usage of Shadow AI Phone Integration

Demonstrates:
1. Pairing a device
2. Sending sensor data
3. Integrating with Shadow AI
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from phone.bluetooth_scanner import BluetoothScanner
from phone.wifi_scanner import WiFiScanner
from phone.communication_observer import CommunicationObserver, CallMetadata
from phone.shadow_phone_integration import ShadowPhoneIntegration

async def main():
    print("=" * 60)
    print("Shadow AI Phone Integration - Example Usage")
    print("=" * 60)
    
    # 1. Initialize integration
    print("\n[1] Initializing Shadow Phone Integration...")
    integration = ShadowPhoneIntegration()
    print("   ✓ Integration initialized")
    
    # 2. Register a device
    print("\n[2] Registering device...")
    integration.register_device("pixel-7-pro", {
        "name": "Pixel 7 Pro",
        "type": "android",
        "os_version": "14"
    })
    print("   ✓ Device registered: Pixel 7 Pro")
    
    # 3. Simulate Bluetooth presence detection
    print("\n[3] Testing Bluetooth presence detection...")
    bt_scanner = BluetoothScanner()
    
    # Register known devices
    bt_scanner.register_known_device("AA:BB:CC:DD:EE:FF", "John's iPhone", "John")
    bt_scanner.register_known_device("11:22:33:44:55:66", "Sarah's Watch", "Sarah")
    print("   ✓ Registered 2 known devices")
    
    # Simulate scan
    # (In real usage, this would come from actual Bluetooth scan)
    print("   Scan would detect: John's iPhone, Sarah's Watch")
    
    # 4. Test WiFi location detection
    print("\n[4] Testing WiFi location detection...")
    wifi_scanner = WiFiScanner()
    
    wifi_scanner.register_location("home", ["MyHomeWiFi", "MyHomeWiFi_5G"])
    wifi_scanner.register_location("work", ["CorpWiFi", "GuestWiFi"])
    print("   ✓ Registered 2 locations")
    
    # Simulate scan
    # (In real usage, this would come from actual WiFi scan)
    print("   Would detect: MyHomeWiFi → location: home")
    
    # 5. Test communication patterns
    print("\n[5] Testing communication patterns...")
    comm_observer = CommunicationObserver()
    
    # Add simulated calls
    for i in range(5):
        comm_observer.observe_call(CallMetadata(
            contact_id=f"contact_{i % 3}",
            contact_name=f"Contact {i % 3}",
            phone_number="+1***",
            call_type="incoming" if i % 2 == 0 else "outgoing",
            duration_seconds=60 * (i + 1),
            timestamp=datetime.now(),
            device_id="pixel-7-pro"
        ))
    
    print(f"   ✓ Recorded {len(comm_observer.calls)} calls")
    
    # Get top contacts
    top_contacts = comm_observer.get_top_contacts(5)
    print(f"   ✓ Top contacts: {[c['contact_name'] for c in top_contacts]}")
    
    # Get communication summary
    summary = comm_observer.get_communication_summary(24)
    print(f"   ✓ Summary: {summary['calls']['total']} calls in last 24h")
    
    # 6. Integration status
    print("\n[6] Shadow AI Status:")
    status = integration.get_shadow_status()
    print(f"   Components: {list(status['components'].keys())}")
    
    # 7. Example: Predict next call
    print("\n[7] Testing prediction...")
    # Add regular pattern
    from datetime import timedelta
    base = datetime.now() - timedelta(hours=12)
    for i in range(6):
        comm_observer.observe_call(CallMetadata(
            contact_id="regular_contact",
            contact_name="Regular Contact",
            phone_number="+1***",
            call_type="incoming",
            duration_seconds=300,
            timestamp=base + timedelta(hours=i*2),
            device_id="pixel-7-pro"
        ))
    
    prediction = comm_observer.predict_next_call("regular_contact")
    if prediction:
        print(f"   ✓ Predicted next call: {prediction['predicted_time'][:16]}")
        print(f"   ✓ Confidence: {prediction['confidence']:.0%}")
        print(f"   ✓ Avg interval: {prediction['avg_interval_hours']:.1f} hours")
    
    print("\n" + "=" * 60)
    print("Example complete! Shadow AI Phone Integration is ready.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
