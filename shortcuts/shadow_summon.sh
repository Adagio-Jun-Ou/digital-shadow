#!/bin/bash
# Shadow Summon — Send data to ShadowOS via curl
# Usage: shadow-summon.sh "your input"

echo "🎤 Shadow Summon"
echo "Enter your transcript:"
read -r INPUT

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/summon/text" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "$INPUT"}' | python3 -m json.tool
