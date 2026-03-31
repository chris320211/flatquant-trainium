#!/bin/bash
# Phase 1: Setup Python paths for FlatQuantBundled module
# Run this before executing any Phase 1 scripts

# This script is located at: agent-workflow/tools/phase1/setup_env.sh
# We need to go up 3 levels to reach the repo root
# Use parameter expansion to avoid dirname issues
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
# Get absolute path without changing directories
if [[ "$SCRIPT_DIR" = /* ]]; then
  REPO_ROOT="$SCRIPT_DIR/../../.."
else
  REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)/../../.."
fi

export PYTHONPATH="${REPO_ROOT}/FlatQuantBundled:${SCRIPT_DIR}:$PYTHONPATH"

# CRITICAL: Do NOT verify deploy import here
# FlatQuantBundled/deploy/transformers/ shadows the real transformers module
# The Python scripts will import things in the correct order:
# 1. transformers (real module) - imported FIRST
# 2. flatquant - from FlatQuantBundled
# 3. deploy - from FlatQuantBundled, but AFTER transformers is loaded

# Verify only flatquant
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1

echo "✓ PYTHONPATH configured"
echo "✓ Ready to run calibration scripts"
