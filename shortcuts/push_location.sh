#!/bin/bash
# Push Location — Push device data to ShadowOS
# This is a template. For real HealthKit data, use the Siri Shortcut
# on your iPhone which auto-collects health metrics.

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/iphone" \
  -H "Content-Type: application/json" \
  -d '{
  "device_id": "iphone_siri",
  "data": {
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "accuracy": 10,
      "name": "Current Location"
    }
  }
}' | python3 -m json.tool
