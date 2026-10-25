#!/bin/bash
# Push iPhone Health — Push device data to ShadowOS
# This is a template. For real HealthKit data, use the Siri Shortcut
# on your iPhone which auto-collects health metrics.

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/iphone" \
  -H "Content-Type: application/json" \
  -d '{
  "device_id": "iphone_siri",
  "data": {
    "health": {
      "heart_rate": 72,
      "hrv": 45,
      "steps": 8500,
      "sleep_hours": 7.5,
      "spo2": 98,
      "stress_level": "low",
      "active_energy": 350
    },
    "battery": {
      "level": 85,
      "charging": false
    },
    "notifications": []
  }
}' | python3 -m json.tool
