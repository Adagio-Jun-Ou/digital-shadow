#!/usr/bin/env python3
"""Tests for Bluetooth and WiFi scanners"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phone.bluetooth_scanner import BluetoothScanner, BluetoothDevice
from phone.wifi_scanner import WiFiScanner, WiFiNetwork
from datetime import datetime

def test_bluetooth_scanner_init():
    scanner = BluetoothScanner()
    assert scanner is not None
    assert scanner.known_devices == {}

def test_bluetooth_register_device():
    scanner = BluetoothScanner()
    scanner.register_known_device("AA:BB:CC:DD:EE:FF", "Test Device", "John")
    assert "AA:BB:CC:DD:EE:FF" in scanner.known_devices
    assert scanner.known_devices["AA:BB:CC:DD:EE:FF"]["name"] == "Test Device"

def test_wifi_scanner_init():
    scanner = WiFiScanner()
    assert scanner is not None
    assert "home" in scanner.locations

def test_wifi_register_location():
    scanner = WiFiScanner()
    scanner.register_location("office", ["OfficeWiFi", "GuestWiFi"])
    assert "office" in scanner.locations
    assert "OfficeWiFi" in scanner.locations["office"]

def test_wifi_detect_location():
    scanner = WiFiScanner()
    scanner.register_location("home", ["TestWiFi_1", "TestWiFi_2"])
    
    networks = [
        WiFiNetwork(ssid="TestWiFi_1", bssid="AA:BB:CC", rssi=-50, channel=1, security="WPA2"),
        WiFiNetwork(ssid="TestWiFi_2", bssid="AA:BB:DD", rssi=-60, channel=6, security="WPA2")
    ]
    
    location = scanner.detect_location(networks)
    assert location == "home"

def test_wifi_is_home():
    scanner = WiFiScanner()
    scanner.register_location("home", ["HomeWiFi"])
    
    networks = [WiFiNetwork(ssid="HomeWiFi", bssid="AA:BB", rssi=-50, channel=1, security="WPA2")]
    
    assert scanner.is_home(networks) == True

def test_wifi_is_work():
    scanner = WiFiScanner()
    scanner.register_location("work", ["WorkWiFi"])
    
    networks = [WiFiNetwork(ssid="WorkWiFi", bssid="CC:DD", rssi=-60, channel=6, security="WPA2")]
    
    assert scanner.is_work(networks) == True

def test_bluetooth_estimate_distance():
    scanner = BluetoothScanner()
    
    # Strong signal should be close
    distance = scanner.estimate_distance(-50)
    assert distance < 10  # Within 10 meters
    
    # Weak signal should be farther
    distance = scanner.estimate_distance(-90)
    assert distance > 10  # Beyond 10 meters

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
