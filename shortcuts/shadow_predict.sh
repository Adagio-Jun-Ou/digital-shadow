#!/bin/bash
# Shadow Predict — Send data to ShadowOS via curl
# Usage: shadow-predict.sh "your input"

echo "🎤 Shadow Predict"
echo "Enter your transcript:"
read -r INPUT

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/summon/text" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Shadow, predict $INPUT"}' | python3 -m json.tool
