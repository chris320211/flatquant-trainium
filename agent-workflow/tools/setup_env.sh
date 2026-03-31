#!/bin/bash
# Setup Python paths for FlatQuantBundled module
# Run this before executing unified pipeline scripts

# Get script directory using parameter expansion (works everywhere)
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"

# Go to script directory and calculate repo root
cd "$SCRIPT_DIR" || exit 1
cd ../.. || exit 1
REPO_ROOT="$(pwd)"

export PYTHONPATH="${REPO_ROOT}/FlatQuantBundled:$PYTHONPATH"

# Verify flatquant is available
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1

echo "✓ PYTHONPATH configured"
echo "✓ Ready to run unified pipeline"
