#!/bin/bash
# Shadow Create Agent — Send data to ShadowOS via curl
# Usage: shadow-create-agent.sh "your input"

echo "🎤 Shadow Create Agent"
echo "Enter your objective:"
read -r INPUT

curl -s -X POST "https://your-host.your-tailnet.ts.net:8766/os/arise" \
  -H "Content-Type: application/json" \
  -d '{"objective": "$INPUT"}' | python3 -m json.tool
