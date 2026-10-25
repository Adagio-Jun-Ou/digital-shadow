#!/bin/bash
# Push Watch Health — Push device data to ShadowOS
# This is a template. For real HealthKit data, use the Siri Shortcut
# on your iPhone which auto-collects health metrics.

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/watch" \
  -H "Content-Type: application/json" \
  -d '{
  "device_id": "apple_watch_siri",
  "data": {
    "heart_rate": 68,
    "hrv": 50,
    "steps": 3200,
    "sleep_hours": 7.0,
    "spo2": 99,
    "stress_level": "low",
    "active_energy": 180,
    "standing_hours": 6
  }
}' | python3 -m json.tool
