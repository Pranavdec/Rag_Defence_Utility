#!/bin/bash
# Quick test script - runs a small subset of experiments
# Usage: ./run_quick_test.sh [samples_per_dataset]

set -e

LIMIT=${1:-5}  # Default to 5 samples if not specified

echo "========================================="
echo "Quick Defense Test (${LIMIT} samples each)"
echo "========================================="
echo ""

# Minimal test: one dataset, three defense configs
DATASET="nq"
CONFIGS=(
    "no_defense:False:False:False"
    "trustrag_only:False:True:False"
    "all_defenses:True:True:True"
)

CONFIG_FILE="config/config.yaml"
BACKUP_FILE="config/config.yaml.backup_quick"
cp "$CONFIG_FILE" "$BACKUP_FILE"

cleanup() {
    mv "$BACKUP_FILE" "$CONFIG_FILE"
    echo "Config restored"
}
trap cleanup EXIT

update_defense_config() {
    python3 << EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
for defense in config.get('defenses', []):
    if defense['name'] == 'differential_privacy':
        defense['enabled'] = $1
    elif defense['name'] == 'trustrag':
        defense['enabled'] = $2
    elif defense['name'] == 'attention_filtering':
        defense['enabled'] = $3
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
EOF
}

RESULT_FILES=()

for combo in "${CONFIGS[@]}"; do
    IFS=':' read -r name dp tr av <<< "$combo"
    echo ""
    echo "Testing: $name (DP:$dp, TR:$tr, AV:$av)"
    echo "-----------------------------------------"
    
    update_defense_config "$dp" "$tr" "$av"
    
    if python main.py run "$DATASET" --limit "$LIMIT"; then
        latest=$(ls -t data/results/run_${DATASET}_*.json | head -1)
        RESULT_FILES+=("$latest")
        echo "✓ Run complete: $latest"
    else
        echo "✗ Run failed"
    fi
done

echo ""
echo "========================================="
echo "Evaluating results..."
echo "========================================="

for result in "${RESULT_FILES[@]}"; do
    echo ""
    echo "Evaluating: $(basename $result)"
    python main.py evaluate "$result" || echo "Evaluation failed"
done

echo ""
echo "✓ Quick test complete!"
echo "Check data/metrics/ for results"
