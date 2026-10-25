#!/bin/bash
# Register iPhone and Apple Watch devices with ShadowOS
# Run this once after setting up ShadowOS

echo "📱 Registering iPhone..."
curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/register" \
  -H "Content-Type: application/json" \
  -d '{"device_id": "iphone_siri", "device_type": "iphone", "name": "Demo iPhone"}'

echo ""
echo "⌚ Registering Apple Watch..."
curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/register" \
  -H "Content-Type: application/json" \
  -d '{"device_id": "apple_watch_siri", "device_type": "watch", "name": "Demo Apple Watch"}'

echo ""
echo "💻 Registering Laptop..."
curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/device/register" \
  -H "Content-Type: application/json" \
  -d '{"device_id": "laptop_demo", "device_type": "laptop", "name": "Demo Laptop"}'

echo ""
echo "✅ All devices registered!"
