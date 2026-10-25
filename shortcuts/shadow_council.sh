#!/bin/bash
# Shadow Council — Send data to ShadowOS via curl
# Usage: shadow-council.sh "your input"

echo "🎤 Shadow Council"
echo "Enter your transcript:"
read -r INPUT

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/summon/text" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Shadow, should I $INPUT"}' | python3 -m json.tool
