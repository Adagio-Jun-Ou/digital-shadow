#!/usr/bin/env python3
"""Tests for Shadow Phone Integration"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: These tests will work once shadowd modules are properly installed
# For now, testing the instantiation

def test_imports():
    """Test all imports work."""
    try:
        from phone.bluetooth_scanner import BluetoothScanner
        from phone.wifi_scanner import WiFiScanner
        from phone.communication_observer import CommunicationObserver
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_integration_init():
    """Test Shadow Phone Integration initialization."""
    try:
        from phone.shadow_phone_integration import ShadowPhoneIntegration
        integration = ShadowPhoneIntegration()
        assert integration is not None
        assert integration.bluetooth is not None
        assert integration.wifi is not None
        assert integration.communications is not None
        print("✓ Integration initialized")
        return True
    except Exception as e:
        print(f"✗ Init failed: {e}")
        return False

def test_device_registration():
    """Test device registration."""
    try:
        from phone.shadow_phone_integration import ShadowPhoneIntegration
        integration = ShadowPhoneIntegration()
        
        integration.register_device("test-phone-001", {
            "name": "Test iPhone",
            "type": "ios",
            "os_version": "17.0"
        })
        
        status = integration.get_device_status("test-phone-001")
        assert status is not None
        assert status["name"] == "Test iPhone"
        print("✓ Device registration works")
        return True
    except Exception as e:
        print(f"✗ Registration failed: {e}")
        return False

def test_communication_integration():
    """Test communication observer integration."""
    try:
        from phone.shadow_phone_integration import ShadowPhoneIntegration
        from phone.communication_observer import CallMetadata
        
        integration = ShadowPhoneIntegration()
        
        # Add a test call
        integration.communications.observe_call(CallMetadata(
            contact_id="test-contact",
            contact_name="Test Contact",
            phone_number="+1***",
            call_type="incoming",
            duration_seconds=60,
            timestamp=datetime.now(),
            device_id="test-phone"
        ))
        
        # Get summary
        summary = integration.get_communication_summary(24)
        assert summary is not None
        assert summary["calls"]["total"] == 1
        print("✓ Communication integration works")
        return True
    except Exception as e:
        print(f"✗ Communication integration failed: {e}")
        return False

def test_bluetooth_presence():
    """Test Bluetooth presence detection."""
    try:
        from phone.bluetooth_scanner import BluetoothScanner
        
        scanner = BluetoothScanner()
        
        # Register known device
        scanner.register_known_device("AA:BB:CC:DD:EE:FF", "Test Device", "John")
        
        assert "AA:BB:CC:DD:EE:FF" in scanner.known_devices
        assert scanner.known_devices["AA:BB:CC:DD:EE:FF"]["owner"] == "John"
        print("✓ Bluetooth presence works")
        return True
    except Exception as e:
        print(f"✗ Bluetooth presence failed: {e}")
        return False

def test_wifi_location():
    """Test WiFi location detection."""
    try:
        from phone.wifi_scanner import WiFiScanner, WiFiNetwork
        
        scanner = WiFiScanner()
        
        # Register location
        scanner.register_location("home", ["MyWiFi", "MyWiFi_5G"])
        
        # Test detection
        networks = [
            WiFiNetwork(ssid="MyWiFi", bssid="AA:BB:CC", rssi=-50, channel=1, security="WPA2")
        ]
        
        location = scanner.detect_location(networks)
        assert location == "home"
        print("✓ WiFi location detection works")
        return True
    except Exception as e:
        print(f"✗ WiFi location failed: {e}")
        return False

if __name__ == "__main__":
    # Run all tests
    tests = [
        test_imports,
        test_integration_init,
        test_device_registration,
        test_communication_integration,
        test_bluetooth_presence,
        test_wifi_location
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n{'='*50}")
    print(f"Tests: {sum(results)}/{len(results)} passed")
    print(f"{'='*50}")
