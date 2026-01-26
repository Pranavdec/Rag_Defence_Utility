#!/bin/bash
# Run PoisonedRAG / CorruptRAG Attacks
# Prerequisites: Run python scripts/ingest_data.py [dataset] first
# Usage: ./run_poisoning_attack.sh [dataset] [attack_type] [poison_rate] [num_targets]
# Example: ./run_poisoning_attack.sh nq poisonedrag 0.01 5


# Auto-activate venv if present
if [ -f "env/bin/activate" ]; then
    echo "Activating virtual environment: env/"
    source env/bin/activate
fi

DATASET=${1:-nq}
ATTACK=${2:-poisonedrag}
RATE=${3:-0.01}
NUM_TARGETS=${4:-5}

echo "================================================"
echo "Corpus Poisoning Attack"
echo "================================================"
echo "Dataset: $DATASET"
echo "Attack Type: $ATTACK"
echo "Poison Rate: $RATE"
echo "Num Targets: $NUM_TARGETS"
echo ""
echo "================================================"

# Create output dir
mkdir -p data/results/attack

# Ingest Data
echo ""
echo "------------------------------------------------"
echo "INGESTING DATA"
echo "------------------------------------------------"
python scripts/ingest_data.py $DATASET

# Run Baseline (No Defense)
echo ""
echo "------------------------------------------------"
echo "PHASE 1: Baseline Attack (No Defense)"
echo "------------------------------------------------"
python scripts/run_poisoning_attack.py $DATASET --attack_type $ATTACK --poison_rate $RATE --defense none --num_targets $NUM_TARGETS

# Run with Defense
echo ""
echo "------------------------------------------------"
echo "PHASE 2: Defended (Using Configured Defenses)"
echo "------------------------------------------------"
python scripts/run_poisoning_attack.py $DATASET --attack_type $ATTACK --poison_rate $RATE --defense trustrag --num_targets $NUM_TARGETS

echo ""
echo "================================================"
echo "Done! Results saved to data/results/attack/"
echo "================================================"
