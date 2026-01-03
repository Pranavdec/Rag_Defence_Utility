#!/bin/bash
set -e

# Datasets to process
DATASETS=("nq" "pubmedqa" "triviaqa")

echo "============================================================"
echo " 🚀 RAG Defence Utility - Run & Evaluate"
echo "============================================================"

# Auto-activate venv if present
if [ -f "env/bin/activate" ]; then
    echo "Using virtual environment: env/"
    source env/bin/activate
fi

for DATASET in "${DATASETS[@]}"; do
    echo -e "\n------------------------------------------------------------"
    echo "Processing: $DATASET"
    echo "------------------------------------------------------------"
    
    # 1. Run Inference
    echo "▶️  Running Inference..."
    python main.py run "$DATASET"
    
    # 2. Find the latest results file for this dataset
    # We use ls -t to sort by time (newest first) and head -n1 to take the top one
    LATEST_RESULT=$(ls -t data/results/run_${DATASET}_*.json 2>/dev/null | head -n1)
    
    if [ -z "$LATEST_RESULT" ]; then
        echo "❌ No result file found for $DATASET. Skipping evaluation."
        continue
    fi
    
    echo "📄 Generated: $LATEST_RESULT"
    
    # 3. Evaluate
    echo "📊 Evaluating..."
    python main.py evaluate "$LATEST_RESULT"
done

echo -e "\n============================================================"
echo "✅ All experiments completed."
echo "Check data/metrics/ for CSV reports."
