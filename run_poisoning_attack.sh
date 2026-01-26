#!/bin/bash
# Run PoisonedRAG / CorruptRAG Attacks
# Usage: ./run_poisoning_attack.sh [dataset] [attack_type] [poison_rate]

DATASET=${1:-nq}
ATTACK=${2:-poisonedrag}
RATE=${3:-0.0001} # Default low rate for millions of documents, or 0.01 for smaller

echo "Running Corpus Poisoning Attack..."
echo "Dataset: $DATASET"
echo "Attack: $ATTACK"
echo "Rate: $RATE"

# Create output dir
mkdir -p data/results/attack

# Run Baseline (No Defense)
echo "------------------------------------------------"
echo "PHASE 1: Baseline Attack (No Defense)"
echo "------------------------------------------------"
python scripts/run_poisoning_attack.py $DATASET --attack_type $ATTACK --poison_rate $RATE --defense none

# Run with Defense
echo "------------------------------------------------"
echo "PHASE 2: Defended (TrustRAG/Configured)"
echo "------------------------------------------------"
python scripts/run_poisoning_attack.py $DATASET --attack_type $ATTACK --poison_rate $RATE --defense trustrag

echo "Done."
