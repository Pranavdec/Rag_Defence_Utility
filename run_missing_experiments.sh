#!/bin/bash
# Script to run only the missing defense configurations
# Based on analysis of run_20260105_experiment

set -e

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollama is not running. Starting Ollama service..."
    ollama serve &
    sleep 3
    echo "✓ Ollama started"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONFIG_FILE="config/config.yaml"
BACKUP_FILE="config/config.yaml.backup"

# Backup config
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo -e "${GREEN}✓${NC} Backed up config"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Restoring config...${NC}"
    mv "$BACKUP_FILE" "$CONFIG_FILE"
    echo -e "${GREEN}✓${NC} Config restored"
}
trap cleanup EXIT

# Function to update defense settings
update_defense_config() {
    local dp_enabled=$1
    local trustrag_enabled=$2
    local av_enabled=$3
    
    python3 << EOF
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

for defense in config.get('defenses', []):
    if defense['name'] == 'differential_privacy':
        defense['enabled'] = $dp_enabled
    elif defense['name'] == 'trustrag':
        defense['enabled'] = $trustrag_enabled
    elif defense['name'] == 'attention_filtering':
        defense['enabled'] = $av_enabled

with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Config: DP=$dp_enabled, TrustRAG=$trustrag_enabled, AV=$av_enabled")
EOF
}

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Missing Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Missing configurations:"
echo "  NQ:        trustrag_av, all_defenses"
echo "  PubMedQA:  dp_av, trustrag_av, all_defenses"
echo "  TriviaQA:  dp_av, trustrag_av, all_defenses"
echo ""
echo "Total: 8 missing experiments"
echo ""

RESULT_FILES=()

# Run missing experiments
run_count=0

# # NQ - trustrag_av
# run_count=$((run_count + 1))
# echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}nq${NC} | Config: ${YELLOW}trustrag_av${NC}"
# update_defense_config "False" "True" "True"
# python main.py run nq --limit 50
# latest_result=$(ls -t data/results/run_nq_*.json | head -1)
# RESULT_FILES+=("$latest_result")
# echo -e "${GREEN}✓${NC} Result: $latest_result"
# echo ""

# # NQ - all_defenses
# run_count=$((run_count + 1))
# echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}nq${NC} | Config: ${YELLOW}all_defenses${NC}"
# update_defense_config "True" "True" "True"
# pythoLT_FILES+=("$latest_result")
# echon main.py run nq --limit 50
# latest_result=$(ls -t data/results/run_nq_*.json | head -1)
# RESU -e "${GREEN}✓${NC} Result: $latest_result"
# echo ""

# # PubMedQA - dp_av
# run_count=$((run_count + 1))
# echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}pubmedqa${NC} | Config: ${YELLOW}dp_av${NC}"
# update_defense_config "True" "False" "True"
# python main.py run pubmedqa --limit 50
# latest_result=$(ls -t data/results/run_pubmedqa_*.json | head -1)
# RESULT_FILES+=("$latest_result")
# echo -e "${GREEN}✓${NC} Result: $latest_result"
# echo ""

# PubMedQA - trustrag_av
run_count=$((run_count + 1))
echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}pubmedqa${NC} | Config: ${YELLOW}trustrag_av${NC}"
update_defense_config "False" "True" "True"
python main.py run pubmedqa --limit 50
latest_result=$(ls -t data/results/run_pubmedqa_*.json | head -1)
RESULT_FILES+=("$latest_result")
echo -e "${GREEN}✓${NC} Result: $latest_result"
echo ""

# PubMedQA - all_defenses
run_count=$((run_count + 1))
echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}pubmedqa${NC} | Config: ${YELLOW}all_defenses${NC}"
update_defense_config "True" "True" "True"
python main.py run pubmedqa --limit 50
latest_result=$(ls -t data/results/run_pubmedqa_*.json | head -1)
RESULT_FILES+=("$latest_result")
echo -e "${GREEN}✓${NC} Result: $latest_result"
echo ""

# TriviaQA - dp_av
run_count=$((run_count + 1))
echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}triviaqa${NC} | Config: ${YELLOW}dp_av${NC}"
update_defense_config "True" "False" "True"
python main.py run triviaqa --limit 50
latest_result=$(ls -t data/results/run_triviaqa_*.json | head -1)
RESULT_FILES+=("$latest_result")
echo -e "${GREEN}✓${NC} Result: $latest_result"
echo ""

# TriviaQA - trustrag_av
run_count=$((run_count + 1))
echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}triviaqa${NC} | Config: ${YELLOW}trustrag_av${NC}"
update_defense_config "False" "True" "True"
python main.py run triviaqa --limit 50
latest_result=$(ls -t data/results/run_triviaqa_*.json | head -1)
RESULT_FILES+=("$latest_result")
echo -e "${GREEN}✓${NC} Result: $latest_result"
echo ""

# TriviaQA - all_defenses
run_count=$((run_count + 1))
echo -e "${GREEN}[Run $run_count/8]${NC} Dataset: ${BLUE}triviaqa${NC} | Config: ${YELLOW}all_defenses${NC}"
update_defense_config "True" "True" "True"
python main.py run triviaqa --limit 50
latest_result=$(ls -t data/results/run_triviaqa_*.json | head -1)
RESULT_FILES+=("$latest_result")
echo -e "${GREEN}✓${NC} Result: $latest_result"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Runs Complete! Starting Evaluation...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Evaluate all new results
eval_count=0
for result_file in "${RESULT_FILES[@]}"; do
    eval_count=$((eval_count + 1))
    echo -e "${GREEN}[Eval $eval_count/8]${NC} Evaluating: ${BLUE}$(basename $result_file)${NC}"
    python main.py evaluate "$result_file"
    echo ""
done

echo -e "${GREEN}✓${NC} All missing experiments and evaluations complete!"
echo ""
echo "New files saved to:"
echo "  - data/results/"
echo "  - data/metrics/"
