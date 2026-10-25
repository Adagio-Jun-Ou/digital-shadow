# Shadow AI Phone Integration - Quick Start

## Installation

```bash
cd /home/user/digital-shadow
pip install fastapi uvicorn pydantic
```

## Quick Start

### 1. Start the Phone API Server

```bash
cd /home/user/digital-shadow
python -m phone.phone_api_v2
```

Server will start on `http://localhost:8766`

### 2. Pair a Device

```bash
curl -X POST "http://localhost:8766/pair?device_id=my-phone&device_name=My%20iPhone&device_type=ios"
```

Returns:
```json
{
  "success": true,
  "device_id": "my-phone",
  "api_key": "...",
  "message": "Device My iPhone paired successfully"
}
```

### 3. Send Location

```bash
curl -X POST "http://localhost:8766/observe/location" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "accuracy": 10.0,
    "timestamp": "2026-04-04T00:00:00",
    "device_id": "my-phone"
  }'
```

### 4. Send Bluetooth Scan

```bash
curl -X POST "http://localhost:8766/observe/bluetooth" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "my-phone",
    "timestamp": "2026-04-04T00:00:00",
    "devices": [
      {"mac": "AA:BB:CC:DD:EE:FF", "name": "Johns iPhone", "rssi": -60}
    ]
  }'
```

### 5. Send WiFi Scan

```bash
curl -X POST "http://localhost:8766/observe/wifi" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "my-phone",
    "timestamp": "2026-04-04T00:00:00",
    "networks": [
      {"ssid": "HomeWiFi", "bssid": "AA:BB:CC", "rssi": -50, "security": "WPA2"}
    ]
  }'
```

## Python SDK

```python
from phone.shadow_phone_integration import ShadowPhoneIntegration

# Initialize
integration = ShadowPhoneIntegration()

# Register device
integration.register_device("my-phone", {
    "name": "My iPhone",
    "type": "ios"
})

# Handle location
await integration.handle_location_update("my-phone", {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "accuracy": 10.0,
    "timestamp": "2026-04-04T00:00:00"
})

# Handle Bluetooth scan
await integration.handle_bluetooth_scan("my-phone", {
    "devices": [
        {"mac": "AA:BB:CC:DD:EE:FF", "name": "Johns iPhone", "rssi": -60}
    ]
})

# Get communication summary
summary = integration.get_communication_summary(24)
print(summary)
```

## Integrating with 10 Shikigami Abilities

### X-Ray (see through walls)
Uses Bluetooth/WiFi to detect nearby people and locations:
```python
bluetooth_scanner.identify_present_devices(scan_results)
wifi_scanner.detect_location(networks)
```

### Life Pulse (person identification)
Identifies people by their device constellations:
```python
# Register known devices
bluetooth_scanner.register_known_device("AA:BB:CC", "John's iPhone", "John")

# Identify present
present = bluetooth_scanner.identify_present_devices(scan)
# Returns: [{"mac": "AA:BB:CC", "name": "John's iPhone", "owner": "John"}]
```

### Social Radar (predicting calls)
Predicts when someone will call based on patterns:
```python
comm_observer.observe_call(call_metadata)
prediction = comm_observer.predict_next_call("contact_id")
# Returns: {"predicted_time": "2024-04-04T12:00", "confidence": 0.85}
```

## Testing

```bash
# Run all phone module tests
python -m pytest /home/user/digital-shadow/phone/ -v

# Run integration tests
python /home/user/digital-shadow/phone/test_shadow_integration.py

# Run example
python /home/user/digital-shadow/phone/example_usage.py
```

## Directory Structure

```
phone/
├── phone_api_v2.py           (99 lines)  - Main API server
├── bluetooth_scanner.py      (150 lines) - Bluetooth utilities
├── wifi_scanner.py           (148 lines) - WiFi utilities
├── communication_observer.py (285 lines) - Call/message tracking
├── shadow_phone_integration.py (297 lines) - Shadow AI integration
├── sensor_simulator.py       (166 lines) - Test simulator
├── data_exporter.py          (140 lines) - Data export
├── example_usage.py           (115 lines) - Example script
└── tests/
    ├── test_phone_api_v2.py   (5 tests)
    ├── test_scanners.py      (8 tests)
    ├── test_communication.py (6 tests)
    └── test_shadow_integration.py (6 tests)

Total: ~1,400 lines, 25 tests
```

## Next Steps

1. Connect real phone app (Android/iOS)
2. Set up persistent data storage
3. Configure notification forwarding
4. Add encryption for privacy
5. Deploy to production server

## Privacy

- Only metadata stored (no message content)
- No location history beyond configurable retention
- Device IDs are hashed
- User can delete all data

## License

MIT License - See LICENSE file
