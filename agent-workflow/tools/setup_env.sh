#!/bin/bash
# Setup Python paths for FlatQuantBundled module
# Run this before executing unified pipeline scripts
# This script is located at: agent-workflow/tools/setup_env.sh
# We need to go up 3 levels to reach the repo root

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/FlatQuantBundled:$PYTHONPATH"

# Verify flatquant is available
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1

echo "✓ PYTHONPATH configured"
echo "✓ Ready to run unified pipeline"
