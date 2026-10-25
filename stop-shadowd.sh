#!/bin/bash
# Stop Digital Shadow

if [ -f "$HOME/digital-shadow/data/shadowd.pid" ]; then
    kill $(cat "$HOME/digital-shadow/data/shadowd.pid") 2>/dev/null
    rm "$HOME/digital-shadow/data/shadowd.pid"
    echo "Digital Shadow stopped"
else
    echo "Digital Shadow not running (no PID file)"
    pkill -f "shadowd.shadowd" && echo "Killed any remaining processes"
fi