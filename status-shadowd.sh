#!/bin/bash
# Check Digital Shadow status

if pgrep -f "shadowd.shadowd" > /dev/null; then
    echo "Digital Shadow is running (PID: $(pgrep -f 'shadowd.shadowd'))"
    curl -s http://127.0.0.1:8765/status | jq .
else
    echo "Digital Shadow is not running"
fi