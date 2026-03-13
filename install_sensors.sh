#!/bin/bash
# Configuration
VERSION="1.0.81"
BASE_URL="https://sensors-updates.srswti.com/darwin/arm64"
MIN_KERNEL_VERSION=26 # Tahoe (Darwin 26)
# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${BLUE}=== BodegaOS Sensors Installer ===${NC}"
echo -e "${BLUE}Checking system requirements...${NC}"
# 1. Check OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" != "Darwin" ]]; then
    echo -e "${RED}Error: This script is only for macOS (Darwin). Detected: $OS${NC}"
    exit 1
fi
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${RED}Error: BodegaOS Sensors requires Apple Silicon (arm64). Detected: $ARCH${NC}"
    exit 1
fi
# 2. Check macOS Version (Tahoe is macOS 16.x)
MACOS_VERSION=$(sw_vers -productVersion)
MAJOR_VERSION=$(echo "$MACOS_VERSION" | cut -d. -f1)
echo -e "  • macOS Version: $MACOS_VERSION"
if (( MAJOR_VERSION < 16 )); then
    echo -e "${RED}Error: Requires macOS Tahoe (Version 16.x) or newer. You are running macOS $MACOS_VERSION.${NC}"
    exit 1
fi
# 3. Check RAM
RAM_BYTES=$(sysctl -n hw.memsize)
RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
echo -e "  • System RAM: ${RAM_GB} GB"
# 4. Determine Edition
if (( RAM_GB > 32 )); then
    EDITION="Pro"
    FILENAME="BodegaOS Sensors Pro-${VERSION}-arm64.dmg"
    URL_FILENAME="BodegaOS%20Sensors%20Pro-${VERSION}-arm64.dmg"
    DOWNLOAD_URL="${BASE_URL}/pro/${URL_FILENAME}"
    echo -e "${GREEN}✓ High-performance system detected (>32GB RAM). Selecting 'Pro' edition.${NC}"
else
    EDITION="Standard"
    FILENAME="BodegaOS Sensors-${VERSION}-arm64.dmg"
    URL_FILENAME="BodegaOS%20Sensors-${VERSION}-arm64.dmg"
    DOWNLOAD_URL="${BASE_URL}/${URL_FILENAME}"
    echo -e "${YELLOW}✓ Standard system detected (<=32GB RAM). Selecting 'Standard' edition.${NC}"
fi
# 5. Download BodegaOS Sensors
DOWNLOAD_DIR="$HOME/Downloads"
SENSORS_PATH="${DOWNLOAD_DIR}/${FILENAME}"
echo -e "\n${BLUE}Downloading BodegaOS Sensors to ${DOWNLOAD_DIR}...${NC}"
echo -e "URL: $DOWNLOAD_URL\n"
curl -L -# -o "$SENSORS_PATH" "$DOWNLOAD_URL"
SENSORS_STATUS=$?
# 6. Download BodegaOS Client
CLIENT_VERSION="1.0.178"
CLIENT_FILENAME="BodegaOS-${CLIENT_VERSION}-arm64.dmg"
CLIENT_URL="https://assets.srswti.com/darwin/arm64/${CLIENT_FILENAME}"
CLIENT_PATH="${DOWNLOAD_DIR}/${CLIENT_FILENAME}"
echo -e "\n${BLUE}Downloading BodegaOS Client...${NC}"
echo -e "URL: $CLIENT_URL\n"
curl -L -# -o "$CLIENT_PATH" "$CLIENT_URL"
CLIENT_STATUS=$?

if [[ $SENSORS_STATUS -eq 0 && $CLIENT_STATUS -eq 0 ]]; then
    echo -e "\n${GREEN}✓ Both downloads complete! Files saved to: ${DOWNLOAD_DIR}${NC}"
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}        INSTALLATION INSTRUCTIONS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "\n${YELLOW}Step 1 — Install BodegaOS Sensors (required):${NC}"
    echo -e "  • Double-click:  ${SENSORS_PATH}"
    echo -e "  • Drag & drop ${GREEN}BodegaOS Sensors${NC} into the Applications folder"
    echo -e "\n${YELLOW}Step 2 — Install BodegaOS (optional):${NC}"
    echo -e "  • Double-click:  ${CLIENT_PATH}"
    echo -e "  • Drag & drop ${GREEN}BodegaOS${NC} into the Applications folder"
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ You're all set and ready to go!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "\n${BLUE}💡 Good to know:${NC}"
    echo -e "  • ${GREEN}BodegaOS Sensors${NC} is the backend inference server — it runs"
    echo -e "    independently and does not require BodegaOS to be installed."
    echo -e "  • ${GREEN}BodegaOS${NC} is an ecosystem of apps built for Apple Silicon —"
    echo -e "    from an Apple Silicon-accelerated browser, chat, and speech engine,"
    echo -e "    to much more. Install it only if you want to explore beyond the"
    echo -e "    inference engine."
    echo -e "\n${BLUE}Getting Started with BodegaOS:${NC}"
    echo -e "  1. Open BodegaOS and log in with Google."
    echo -e "  2. Go to Chat → Bodega Hub → Advanced."
    echo -e "  3. Click Docs to learn how to use the Inference Engine"
    echo -e "     or add Bodega as a provider."
else
    echo -e "\n${RED}✗ One or more downloads failed.${NC}"
    exit 1
fi