#!/bin/bash

# Comprehensive test runner for all dataset and configuration combinations
# This script runs each configuration one after another with separate log files

source /media/crk/datastore_1/Rag_Defence_Utility/env/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

CONFIG_FILE="config/config.yaml"
BACKUP_CONFIG="config/config_backup.yaml"

# Backup original config
cp ${CONFIG_FILE} ${BACKUP_CONFIG}

# Function to update YAML config
update_config() {
    local dataset=$1
    local test_size=$2
    local deepeval=$3
    local mba_enabled=$4
    local poisoned_enabled=$5
    local dp_enabled=$6
    local trustrag_enabled=$7
    local av_enabled=$8
    local ado_enabled=$9
    
    # Create new config
    cat > ${CONFIG_FILE} << EOF
system:
  embedding_model: all-MiniLM-L6-v2
  llm:
    provider: huggingface 
    model_path: meta-llama/Llama-3.1-8B-Instruct
    device: auto
    temperature: 0.0
    model_name: llama3
  judge_llm: llama3

paths:
  chroma_db: data/chroma_db
  results: data/results
  cache: data/raw
data:
  dataset: ${dataset}
  ingestion_size: 700
  ingestion_seed: 42
  test_size: ${test_size}
  test_seed: 123
evaluation:
  skip_deepeval: ${deepeval}
  deepeval_max_concurrent: 5
retrieval:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50
defenses:
- name: differential_privacy
  enabled: ${dp_enabled}
  method: dp_approx
  epsilon: 3.0
  delta: 0.01
  candidate_multiplier: 3
- name: trustrag
  enabled: ${trustrag_enabled}
  similarity_threshold: 0.88
  rouge_threshold: 0.25
  candidate_multiplier: 3
- name: attention_filtering
  enabled: ${av_enabled}
  model_path: meta-llama/Llama-3.1-8B-Instruct
  top_tokens: 100
  max_corruptions: 3
  threshold: 50
  device: cuda
  candidate_multiplier: 3
attack:
  mba:
    enabled: ${mba_enabled}
    M: 5
    gamma: 0.35
    num_members: 30
    num_non_members: 20
    device: auto
    proxy_model: gpt2-xl
    enable_spelling_correction: false
    max_document_words: 200
    seed: 1235213
  poisoned_rag:
    enabled: ${poisoned_enabled}
    poisoning_rate: 10
    num_targets: 50
    seed: 42
    target_start_index: 0
    diversity_level: true
ado:
  enabled: ${ado_enabled}
  user_id: test_user_001
  sentinel_model: mistral
  strategist_model: mistral
  strategist_mode: llm
  trust_score_decay: 0.05
EOF
}

# Function to run test
run_test() {
    local log_name=$1
    shift
    local description="$@"
    
    echo "========================================" | tee -a ${LOG_DIR}/summary.log
    echo "Running: ${description}" | tee -a ${LOG_DIR}/summary.log
    echo "Log file: ${LOG_DIR}/${log_name}.log" | tee -a ${LOG_DIR}/summary.log
    echo "Started at: $(date)" | tee -a ${LOG_DIR}/summary.log
    echo "========================================" | tee -a ${LOG_DIR}/summary.log
    
    python scripts/comprehensive_eval.py > ${LOG_DIR}/${log_name}.log 2>&1
    
    echo "Completed at: $(date)" | tee -a ${LOG_DIR}/summary.log
    echo "" | tee -a ${LOG_DIR}/summary.log
}

echo "Starting comprehensive test suite at $(date)" | tee ${LOG_DIR}/summary.log
echo "Log directory: ${LOG_DIR}" | tee -a ${LOG_DIR}/summary.log
echo "" | tee -a ${LOG_DIR}/summary.log

# ==========================================
# 4. All datasets with both attacks + ADO
# ==========================================

# # NQ - Both attacks + ADO
update_config "nq" 50 false true true false false false true
run_test "nq_12_both_attacks_ado" "NQ: Both attacks and ADO enabled"

# # PubMedQA - Both attacks + ADO
update_config "pubmedqa" 50 false true true false false false true
run_test "pubmedqa_11_both_attacks_ado" "PubMedQA: Both attacks and ADO enabled"

# TriviaQA - Both attacks + ADO
update_config "triviaqa" 50 false true true false false false true
run_test "triviaqa_14_both_attacks_ado" "TriviaQA: Both attacks and ADO enabled"

# ==========================================
# Restore original config and finish
# ==========================================

echo "========================================" | tee -a ${LOG_DIR}/summary.log
echo "All tests completed at $(date)" | tee -a ${LOG_DIR}/summary.log
echo "Restoring original configuration" | tee -a ${LOG_DIR}/summary.log
echo "========================================" | tee -a ${LOG_DIR}/summary.log

cp ${BACKUP_CONFIG} ${CONFIG_FILE}
rm ${BACKUP_CONFIG}

echo ""
echo "Test suite complete!"
echo "All logs are saved in: ${LOG_DIR}/"
echo "Summary log: ${LOG_DIR}/summary.log"
