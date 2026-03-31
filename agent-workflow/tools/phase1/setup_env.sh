#!/bin/bash
# Phase 1: Setup Python paths for FlatQuantBundled module
# Run this before executing any Phase 1 scripts

export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled/deploy:$PYTHONPATH

# Verify modules are importable
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1
python3 -c "import deploy; print('✓ deploy imported')" || exit 1

echo "✓ All paths configured"
