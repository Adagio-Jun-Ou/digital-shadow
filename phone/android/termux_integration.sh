#!/bin/bash
# Digital Shadow Phone Integration - Termux/Android Script
# Run this on your Android device via Termux

SHADOW_HOST="${SHADOW_HOST:-http://YOUR_LAPTOP_IP:8765}"
DEVICE_ID="${DEVICE_ID:-phone_$(date +%s)}"
TOKEN_FILE="$HOME/.shadow_token"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🐺 Digital Shadow Phone Integration${NC}"
echo ""

# Register device if first run
if [ ! -f "$TOKEN_FILE" ]; then
    echo -e "${GREEN}Registering device...${NC}"
    DEVICE_NAME="Termux_$(hostname 2>/dev/null || echo 'Android')"
    
    RESPONSE=$(curl -s -X POST "$SHADOW_HOST/phone/register" \
        -H "Content-Type: application/json" \
        -d "{\"device_id\": \"$DEVICE_ID\", \"device_name\": \"$DEVICE_NAME\", \"device_type\": \"android\"}")
    
    TOKEN=$(echo "$RESPONSE" | jq -r '.token // empty')
    
    if [ -n "$TOKEN" ]; then
        echo "$TOKEN" > "$TOKEN_FILE"
        echo "$DEVICE_ID" >> "$TOKEN_FILE"
        echo -e "${GREEN}✓ Device registered${NC}"
        echo "Device ID: $DEVICE_ID"
    else
        echo -e "${RED}✗ Registration failed${NC}"
        exit 1
    fi
else
    TOKEN=$(sed -n '1p' "$TOKEN_FILE")
    DEVICE_ID=$(sed -n '2p' "$TOKEN_FILE")
    echo -e "${GREEN}✓ Using saved token${NC}"
fi

# Function to send location
send_location() {
    echo -e "${GREEN}Getting location...${NC}"
    
    # Termux location command
    LOCATION=$(termux-location 2>/dev/null)
    
    if [ -n "$LOCATION" ]; then
        LAT=$(echo "$LOCATION" | jq -r '.latitude // empty')
        LON=$(echo "$LOCATION" | jq -r '.longitude // empty')
        ACC=$(echo "$LOCATION" | jq -r '.accuracy // empty')
        
        if [ -n "$LAT" ] && [ -n "$LON" ]; then
            curl -s -X POST "$SHADOW_HOST/phone/location?device_id=$DEVICE_ID" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $TOKEN" \
                -d "{\"latitude\": $LAT, \"longitude\": $LON, \"accuracy\": $ACC}" > /dev/null
            echo -e "${GREEN}✓ Location sent: $LAT, $LON${NC}"
        fi
    else
        echo -e "${RED}✗ Could not get location${NC}"
    fi
}

# Function to send notification data
send_notification() {
    APP="$1"
    TITLE="$2"
    MESSAGE="$3"
    
    curl -s -X POST "$SHADOW_HOST/phone/notification?device_id=$DEVICE_ID" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"app\": \"$APP\", \"title\": \"$TITLE\", \"message\": \"$MESSAGE\"}" > /dev/null
    
    echo -e "${GREEN}✓ Notification logged${NC}"
}

# Function to send app usage
send_app_usage() {
    echo -e "${GREEN}Getting app usage...${NC}"
    
    # This requires notification listener permission in Termux
    # termux-notification-list could be used here
    
    echo -e "${PURPLE}App usage monitoring requires Termux:API${NC}"
}

# Function to sync to shadow
sync_to_shadow() {
    echo -e "${GREEN}Syncing to shadow...${NC}"
    
    # Create snapshot of current state
    SNAPSHOT=$(cat <<EOF
{
    "observation_type": "sync",
    "content": {
        "timestamp": "$(date -Iseconds)",
        "device_type": "termux_android"
    },
    "device_id": "$DEVICE_ID"
}
EOF
)
    
    curl -s -X POST "$SHADOW_HOST/phone/sync" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "[$SNAPSHOT]" > /dev/null
    
    echo -e "${GREEN}✓ Synced${NC}"
}

# Summon shikigami
summon() {
    QUERY="$1"
    SHIKIGAMI="$2"
    
    if [ -n "$SHIKIGAMI" ]; then
        ENDPOINT="$SHADOW_HOST/shikigami/summon"
        BODY="{\"query\": \"$QUERY\", \"shikigami\": \"$SHIKIGAMI\"}"
    else
        ENDPOINT="$SHADOW_HOST/shikigami/summon"
        BODY="{\"query\": \"$QUERY\"}"
    fi
    
    echo -e "${PURPLE}Summoning...${NC}"
    
    RESPONSE=$(curl -s -X POST "$ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "$BODY")
    
    echo ""
    echo "$RESPONSE" | jq -r '.output // .error // .'
}

# Interactive mode
interactive() {
    echo ""
    echo -e "${PURPLE}Commands:${NC}"
    echo "  location   - Send current location"
    echo "  sync       - Sync with shadow"
    echo "  summon     - Summon a shikigami"
    echo "  status     - Check connection"
    echo "  quit       - Exit"
    echo ""
    
    while true; do
        echo -ne "${GREEN}shadow>${NC} "
        read -r CMD
        
        case "$CMD" in
            location) send_location ;;
            sync) sync_to_shadow ;;
            summon)
                echo -ne "Query: "
                read -r QUERY
                echo -ne "Shikigami (empty for auto): "
                read -r SHIKI
                summon "$QUERY" "$SHIKI"
                ;;
            status)
                curl -s "$SHADOW_HOST/" | jq .
                ;;
            quit|exit) exit 0 ;;
            *) echo "Unknown command: $CMD" ;;
        esac
    done
}

# Main
case "${1:-interactive}" in
    location) send_location ;;
    sync) sync_to_shadow ;;
    summon) summon "$2" "$3" ;;
    *) interactive ;;
esac