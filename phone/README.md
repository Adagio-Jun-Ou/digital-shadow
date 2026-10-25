# Phone Integration Documentation

## Overview

Shadow AI Phone Integration allows mobile devices to send sensor data to Shadow AI for:

- Location tracking (GPS)
- Motion/activity detection (accelerometer)
- Battery monitoring
- App usage patterns
- Bluetooth presence detection
- WiFi location signatures

## Architecture

```
Phone Device (Android/iOS)
          ↓
    HTTP POST (WiFi/Cell)
          ↓
Phone API Server (port 8766)
          ↓
Shadow AI unified_system.observe()
          ↓
Memory/RAG/Analysis
          ↓
10 Shikigami Abilities
```

## API Endpoints

### Device Pairing

```
POST /pair
Parameters:
  - device_id: string
  - device_name: string
  - device_type: string (android/ios)

Returns:
  - success: boolean
  - api_key: string (for authentication)
```

### Location Observation

```
POST /observe/location
Body:
{
  "latitude": float,
  "longitude": float,
  "accuracy": float,
  "altitude": float (optional),
  "speed": float (optional),
  "timestamp": datetime,
  "device_id": string
}
```

### Bluetooth Scan

```
POST /observe/bluetooth
Body:
{
  "device_id": string,
  "timestamp": datetime,
  "devices": [
    {
      "mac": string,
      "name": string,
      "rssi": int
    }
  ]
}
```

### WiFi Scan

```
POST /observe/wifi
Body:
{
  "device_id": string,
  "timestamp": datetime,
  "networks": [
    {
      "ssid": string,
      "bssid": string,
      "rssi": int,
      "security": string
    }
  ]
}
```

## Integration with Abilities

### X-Ray Ability
Uses WiFi scans to detect location
Uses Bluetooth to see who's nearby

### Life Pulse Ability  
Uses Bluetooth to identify people by their devices
Estimates distance from RSSI values

### Omni Sense Ability
Combines all sensor data for environmental awareness

## Privacy

- Only metadata is stored (no message content)
- Location data can be encrypted
- Users can delete their data
- API keys are hashed (not stored in plain text)

## Testing

Run tests:
```bash
cd /home/user/digital-shadow
python -m pytest phone/ -v
```

## Simulator

Run sensor simulator:
```bash
python -m phone.sensor_simulator
```

## Files

- `phone_api_v2.py` - Main API server (99 lines)
- `bluetooth_scanner.py` - Bluetooth utilities (150 lines)
- `wifi_scanner.py` - WiFi utilities (148 lines)
- `sensor_simulator.py` - Testing simulator (166 lines)
- `test_phone_api_v2.py` - API tests (5 tests)
- `test_scanners.py` - Scanner tests (8 tests)

Total: 563 lines, 13 tests