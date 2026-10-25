#!/usr/bin/env python3
"""
Shadow AI - iPhone Shortcuts Webhook Collector

This script runs as part of the ShadowOS API server and accepts
data pushed from iPhone Shortcuts via POST requests.

## iPhone Shortcuts Setup

Create the following Shortcuts on your iPhone:

### 1. Location Shortcut (runs on significant location change)
1. Open Shortcuts app
2. Create new Shortcut
3. Add action: "Get Current Location"
4. Add action: "Get Contents of URL"
   - URL: https://your-host.your-tailnet.ts.net:8444/os/device/iphone
   - Method: POST
   - Headers: Authorization: Bearer <your-api-key>
   - Body: JSON
5. Body template:
```json
{
  "device_id": "iphone_demo",
  "data": {
    "location": {
      "latitude": "[Current Location Latitude]",
      "longitude": "[Current Location Longitude]",
      "accuracy": "[Current Location Horizontal Accuracy]",
      "name": "[Current Location Name]"
    },
    "battery": {
      "level": "[Battery Level]",
      "charging": "[Is Charging]"
    }
  }
}
```
6. Add to Automation: "When I arrive at [Location]" / "When I leave [Location]"

### 2. Health Data Shortcut (runs every 30 minutes)
1. Create new Shortcut
2. Add action: "Get Health Samples"
   - Type: Heart Rate, Step Count, Sleep Analysis
   - Period: Last 30 minutes
3. Add action: "Get Contents of URL"
   - URL: https://your-host.your-tailnet.ts.net:8444/os/device/watch
   - Method: POST
   - Headers: Authorization: Bearer <your-api-key>
   - Body: JSON
4. Body template:
```json
{
  "device_id": "watch_demo",
  "data": {
    "health": {
      "heart_rate": "[Latest Heart Rate]",
      "step_count": "[Step Count]",
      "sleep_phase": "[Sleep Analysis]",
      "exercise_minutes": "[Exercise Minutes]",
      "spo2": "[Blood Oxygen]"
    }
  }
}
```
6. Add to Automation: "Run every 30 minutes" or "Run on Wake"

### 3. Notification Shortcut (runs on notification)
1. Create new Shortcut
2. Add action: "Get Contents of URL"
   - URL: https://your-host.your-tailnet.ts.net:8444/os/device/iphone
   - Method: POST
   - Body: JSON
3. Body template:
```json
{
  "device_id": "iphone_demo",
  "data": {
    "notifications": [
      {
        "app": "[App Name]",
        "category": "[Notification Category]",
        "timestamp": "[Current Date]"
      }
    ]
  }
}
```
4. Add to Automation: "When I receive a notification"

### 4. Daily Health Summary Shortcut (runs each morning)
Collects overnight heart rate, sleep, HRV, SpO2 from Apple Health.
"""

# The actual webhook handling is done by the FastAPI endpoints defined in api_router.py
# This file serves as documentation for setting up iPhone Shortcuts