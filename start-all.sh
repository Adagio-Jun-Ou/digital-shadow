#!/bin/bash
# Digital Shadow - Complete Startup

cd ~/digital-shadow
source venv/bin/activate

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

SHADOW_HOST="${SHADOW_HOST:-127.0.0.1}"
SHADOW_PORT="${SHADOW_PORT:-8765}"
PHONE_HOST="${PHONE_HOST:-127.0.0.1}"
PHONE_PORT="${PHONE_PORT:-8766}"
SENSOR_HOST="${SENSOR_HOST:-127.0.0.1}"
SENSOR_PORT="${SENSOR_PORT:-8767}"

echo "════════════════════════════════════════════════════════════"
echo "  🐺 DIGITAL SHADOW - TRUE SUPERHUMAN ABILITIES"
echo "════════════════════════════════════════════════════════════"
echo ""

# Kill existing
pkill -f "uvicorn shadowd" 2>/dev/null
pkill -f "uvicorn phone" 2>/dev/null
sleep 2

mkdir -p data/logs

# Start shadow daemon
echo "Starting Shadow Daemon on ${SHADOW_HOST}:${SHADOW_PORT}..."
uvicorn shadowd.shadowd:app --host "${SHADOW_HOST}" --port "${SHADOW_PORT}" > data/logs/shadowd.log 2>&1 &
sleep 3

# Verify
if curl -s "http://${SHADOW_HOST}:${SHADOW_PORT}/" > /dev/null 2>&1; then
    echo "✓ Shadow Daemon running"
else
    echo "✗ Shadow Daemon failed"
    tail -20 data/logs/shadowd.log
    exit 1
fi

# Start phone API
echo "Starting Phone API on ${PHONE_HOST}:${PHONE_PORT}..."
uvicorn phone.phone_api:app --host "${PHONE_HOST}" --port "${PHONE_PORT}" > data/logs/phone_api.log 2>&1 &
sleep 2

# Start sensor API
echo "Starting Sensor API on ${SENSOR_HOST}:${SENSOR_PORT}..."
uvicorn phone.sensors:app --host "${SENSOR_HOST}" --port "${SENSOR_PORT}" > data/logs/sensor_api.log 2>&1 &
sleep 2

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Endpoints:"
echo "    Shadow:  http://${SHADOW_HOST}:${SHADOW_PORT}"
echo "    Phone:   http://${PHONE_HOST}:${PHONE_PORT}"
echo "    Sensors: http://${SENSOR_HOST}:${SENSOR_PORT}"
echo "════════════════════════════════════════════════════════════"