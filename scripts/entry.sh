#!/bin/bash
set -e

echo "Starting Shadow AI..."

# Create data directories if needed
mkdir -p "${SHADOW_DATA_DIR:-/app/data}/memory" \
      "${SHADOW_DATA_DIR:-/app/data}/logs" \
      "${SHADOW_DATA_DIR:-/app/data}/workspace"

# Execute the main command
exec "$@"