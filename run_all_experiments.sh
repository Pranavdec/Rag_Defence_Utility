#!/bin/bash
# Script to run all defense combinations on all datasets and evaluate results
# Usage: ./run_all_experiments.sh [--limit N]

set -e  # Exit on error

# Parse command line arguments
LIMIT_ARG=""
if [ "$1" == "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    echo "Using limit: $2 samples per dataset"
fi

# Datasets to test
DATASETS=("pubmedqa" "triviaqa")

# Defense combinations to test
# Each entry: "NAME:dp_enabled:trustrag_enabled:av_enabled"
DEFENSE_COMBOS=(
    "no_defense:False:False:False"
    "dp_only:True:False:False"
    "trustrag_only:False:True:False"
    "av_only:False:False:True"
    "dp_trustrag:True:True:False"
    "dp_av:True:False:True"
    "trustrag_av:False:True:True"
    "all_defenses:True:True:True"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RAG Defense Evaluation - Full Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Datasets: ${DATASETS[@]}"
echo "Defense combinations: ${#DEFENSE_COMBOS[@]}"
echo ""

# Backup original config
CONFIG_FILE="config/config.yaml"
BACKUP_FILE="config/config.yaml.backup"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo -e "${GREEN}✓${NC} Backed up config to $BACKUP_FILE"

# Array to store all result files for evaluation
RESULT_FILES=()

# Function to restore config on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Restoring original config...${NC}"
    mv "$BACKUP_FILE" "$CONFIG_FILE"
    echo -e "${GREEN}✓${NC} Config restored"
}
trap cleanup EXIT

# Function to update defense settings in config.yaml
update_defense_config() {
    local dp_enabled=$1
    local trustrag_enabled=$2
    local av_enabled=$3
    
    # Use Python to update YAML properly
    python3 << EOF
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Update defense settings
for defense in config.get('defenses', []):
    if defense['name'] == 'differential_privacy':
        defense['enabled'] = $dp_enabled
    elif defense['name'] == 'trustrag':
        defense['enabled'] = $trustrag_enabled
    elif defense['name'] == 'attention_filtering':
        defense['enabled'] = $av_enabled

with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Updated config: DP=$dp_enabled, TrustRAG=$trustrag_enabled, AV=$av_enabled")
EOF
}

# Main experiment loop
total_runs=$((${#DATASETS[@]} * ${#DEFENSE_COMBOS[@]}))
current_run=0

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Experiments (Total: $total_runs)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for combo in "${DEFENSE_COMBOS[@]}"; do
    IFS=':' read -r combo_name dp_enabled trustrag_enabled av_enabled <<< "$combo"
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Defense Configuration: $combo_name${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "  DP: $dp_enabled | TrustRAG: $trustrag_enabled | AV: $av_enabled"
    echo ""
    
    # Update config for this defense combination
    update_defense_config "$dp_enabled" "$trustrag_enabled" "$av_enabled"
    
    for dataset in "${DATASETS[@]}"; do
        current_run=$((current_run + 1))
        echo ""
        echo -e "${GREEN}[Run $current_run/$total_runs]${NC} Dataset: ${BLUE}$dataset${NC} | Defense: ${YELLOW}$combo_name${NC}"
        echo "----------------------------------------"
        
        # Run the experiment
        if python main.py run "$dataset" $LIMIT_ARG; then
            echo -e "${GREEN}✓${NC} Run completed successfully"
            
            # Find the most recent result file for this dataset
            latest_result=$(ls -t data/results/run_${dataset}_*.json | head -1)
            if [ -f "$latest_result" ]; then
                RESULT_FILES+=("$latest_result")
                echo -e "${GREEN}✓${NC} Result saved: $latest_result"
            else
                echo -e "${RED}✗${NC} Warning: Could not find result file"
            fi
        else
            echo -e "${RED}✗${NC} Run failed for $dataset with $combo_name"
            echo -e "${RED}   Continuing with next experiment...${NC}"
        fi
        
        # Small delay between runs
        sleep 2
    done
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Runs Complete! Starting Evaluation...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total result files to evaluate: ${#RESULT_FILES[@]}"
echo ""

# Evaluate all result files
eval_count=0
eval_success=0
eval_failed=0

for result_file in "${RESULT_FILES[@]}"; do
    eval_count=$((eval_count + 1))
    echo ""
    echo -e "${GREEN}[Eval $eval_count/${#RESULT_FILES[@]}]${NC} Evaluating: ${BLUE}$(basename $result_file)${NC}"
    echo "----------------------------------------"
    
    if python main.py evaluate "$result_file"; then
        eval_success=$((eval_success + 1))
        echo -e "${GREEN}✓${NC} Evaluation completed"
    else
        eval_failed=$((eval_failed + 1))
        echo -e "${RED}✗${NC} Evaluation failed"
    fi
    
    # Small delay between evaluations
    sleep 1
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EXPERIMENT SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Runs: $total_runs"
echo "Result Files: ${#RESULT_FILES[@]}"
echo ""
echo "Evaluations:"
echo -e "  ${GREEN}Success: $eval_success${NC}"
echo -e "  ${RED}Failed: $eval_failed${NC}"
echo ""
echo "Results directory: data/results/"
echo "Metrics directory: data/metrics/"
echo ""

# Generate summary report
echo -e "${YELLOW}Generating summary report...${NC}"
python3 << 'PYTHON_SCRIPT'
import json
import os
from datetime import datetime
from pathlib import Path

metrics_dir = Path("data/metrics")
results_dir = Path("data/results")

# Find all recent metric files (from this run)
metric_files = sorted(metrics_dir.glob("eval_*_metrics.csv"), key=os.path.getmtime, reverse=True)

print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total metric files: {len(metric_files)}")
print(f"\nLatest metrics available in: {metrics_dir}")
print(f"Latest results available in: {results_dir}")
print("\nUse these metrics to compare defense effectiveness!")
print("="*60)
PYTHON_SCRIPT

echo -e "${GREEN}✓${NC} All experiments and evaluations complete!"
echo ""
