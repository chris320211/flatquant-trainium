#!/bin/bash
# Phase 1: Setup Python paths for FlatQuantBundled module
# Run this before executing any Phase 1 scripts

# This script is located at: agent-workflow/tools/phase1/setup_env.sh
# We need to go up 3 levels to reach the repo root
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)/../../.."

export PYTHONPATH="${REPO_ROOT}/FlatQuantBundled:$PYTHONPATH"

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
