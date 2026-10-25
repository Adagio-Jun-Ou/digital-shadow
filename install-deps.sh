#!/bin/bash
# Install optional dependencies for full observer functionality

echo "Installing Digital Shadow dependencies..."

# System dependencies
sudo apt-get update
sudo apt-get install -y \
    xclip xdotool \
    inotify-tools \
    jq \
    gpsd 2>/dev/null || true

# Python dependencies
pip install --user watchfiles || true

echo "Dependencies installed."