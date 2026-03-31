#!/bin/bash
# Phase 2: Setup Python paths for FlatQuantBundled module
# Sources the setup_env.sh from phase1 to configure PYTHONPATH

# Get directory of this script and phase1 directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE1_DIR="$SCRIPT_DIR/../phase1"

# Source phase1's setup script
source "$PHASE1_DIR/setup_env.sh"

echo "✓ Phase 2 environment ready"
