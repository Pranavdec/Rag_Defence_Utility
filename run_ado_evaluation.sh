#!/bin/bash
# Run ADO Attack & Utility Evaluation
# Evaluates attack success rate and utility metrics with ADO enabled/disabled
#
# Usage:
#   ./run_ado_evaluation.sh [dataset] [--compare]
#
# Examples:
#   ./run_ado_evaluation.sh nq                    # Single run with current config
#   ./run_ado_evaluation.sh nq --compare          # Compare ADO ON vs OFF
#   ./run_ado_evaluation.sh pubmedqa --compare    # Compare on PubMedQA

set -e

# Auto-activate venv if present
if [ -f "env/bin/activate" ]; then
    echo "Activating virtual environment: env/"
    source env/bin/activate
fi

# Check if Ollama is running (needed for ADO)
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Warning: Ollama is not running. ADO requires Ollama."
    echo "   Starting Ollama service..."
    ollama serve &
    sleep 3
    echo "   Pulling llama3 model (if needed)..."
    ollama pull llama3
    sleep 2
    echo "✓ Ollama started"
else
    echo "✓ Ollama is already running"
fi

DATASET=${1:-nq}
shift || true

echo "=========================================="
echo "ADO Attack & Utility Evaluation"
echo "=========================================="
echo "Dataset: $DATASET"
echo ""

python scripts/evaluate_ado_attacks.py "$DATASET" "$@"
