#!/bin/bash
# Phase 1: Setup Python paths for FlatQuantBundled module
# Run this before executing any Phase 1 scripts

export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled:$PYTHONPATH

# NOTE: Do NOT add FlatQuantBundled/deploy to PYTHONPATH
# It contains a deploy/transformers/ dir that shadows the real transformers module
# Instead, import deploy from FlatQuantBundled directly

# Verify modules are importable
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1
python3 -c "from flatquant import deploy; print('✓ deploy imported')" || exit 1

echo "✓ All paths configured"
