#!/bin/bash
# Setup Python paths for FlatQuantBundled module
# This script uses dynamic paths that work on any machine

# This script is at: agent-workflow/outputs/meta-llama__Llama-2-7b-hf/setup_env.sh
# FlatQuantBundled is at: FlatQuantBundled/ (3 levels up)
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)/../../.."

export PYTHONPATH="${REPO_ROOT}/FlatQuantBundled:$PYTHONPATH"

# Verify modules are importable
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1

echo "✓ PYTHONPATH configured: $PYTHONPATH"
echo "✓ Ready to run Phase 1 and Phase 2 scripts"
