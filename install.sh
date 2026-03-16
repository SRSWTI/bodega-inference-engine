#!/bin/bash
# One-liner bootstrap: curl -fsSL https://raw.githubusercontent.com/SRSWTI/bodega-inference-engine/main/bodega_engine_tests/install.sh | bash

set -e

REPO_URL="https://github.com/SRSWTI/bodega-inference-engine.git"
CLONE_DIR="bodega-inference-engine"

echo "Cloning bodega-inference-engine..."
git clone "$REPO_URL" "$CLONE_DIR"
cd "$CLONE_DIR"

echo "Making scripts executable..."
chmod +x setup.sh
[ -f install_sensors.sh ] && chmod +x install_sensors.sh

echo "Running setup..."
./setup.sh
