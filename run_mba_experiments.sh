#!/bin/bash
# Run MBA (Membership Inference Attack) Experiments
# 
# Usage:
#   ./run_mba_experiments.sh [dataset] [options]
#
# Examples:
#   ./run_mba_experiments.sh nq
#   ./run_mba_experiments.sh pubmedqa --M 15 --gamma 0.6
#   ./run_mba_experiments.sh triviaqa --num-members 100

set -e

# Auto-activate venv if present
if [ -f "env/bin/activate" ]; then
    echo "Activating virtual environment: env/"
    source env/bin/activate
fi

echo "=========================================="
echo "MBA Membership Inference Attack"
echo "=========================================="

python scripts/run_mba_experiments.py "$@"
