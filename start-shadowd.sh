#!/bin/bash
# Digital Shadow startup script

cd "$HOME/digital-shadow"
export SHADOW_DATA_DIR="$HOME/digital-shadow/data"
source venv/bin/activate

nohup python -m shadowd.shadowd > "$HOME/digital-shadow/data/logs/shadowd.log" 2>&1 &
echo $! > "$HOME/digital-shadow/data/shadowd.pid"
echo "Digital Shadow started. PID: $(cat $HOME/digital-shadow/data/shadowd.pid)"