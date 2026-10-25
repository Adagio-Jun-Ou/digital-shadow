#!/bin/bash
# Shadow Status — Siri Shortcut via curl
curl -s "https://your-host.your-tailnet.ts.net:8766/os/status" | python3 -m json.tool
