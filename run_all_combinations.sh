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
  sentinel_model: llama3
  strategist_model: llama3
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
# 1. Dataset: NQ
# ==========================================

# NQ - All disabled
update_config "nq" 50 false false false false false false false
run_test "nq_01_all_disabled" "NQ: All defenses, ADO, and attacks disabled"

# NQ - Just poisoning
update_config "nq" 50 false false true false false false false
run_test "nq_02_poisoning_only" "NQ: Just poisoning enabled"

# NQ - Poisoning + TrustRAG
update_config "nq" 50 false false true false true false false
run_test "nq_03_poisoning_trustrag" "NQ: Poisoning and TrustRAG enabled"

# NQ - Poisoning + Attention Filtering
update_config "nq" 50 false false true false false true false
run_test "nq_04_poisoning_av" "NQ: Poisoning and Attention Filtering enabled"

# NQ - All attacks disabled, defense combinations
# Just DP
update_config "nq" 50 false false false true false false false
run_test "nq_05_dp_only" "NQ: Just Differential Privacy enabled"

# Just TrustRAG
update_config "nq" 50 false false false false true false false
run_test "nq_06_trustrag_only" "NQ: Just TrustRAG enabled"

# Just AV
update_config "nq" 50 false false false false false true false
run_test "nq_07_av_only" "NQ: Just Attention Filtering enabled"

# DP + TrustRAG
update_config "nq" 50 false false false true true false false
run_test "nq_08_dp_trustrag" "NQ: DP and TrustRAG enabled"

# DP + AV
update_config "nq" 50 false false false true false true false
run_test "nq_09_dp_av" "NQ: DP and Attention Filtering enabled"

# TrustRAG + AV
update_config "nq" 50 false false false false true true false
run_test "nq_10_trustrag_av" "NQ: TrustRAG and Attention Filtering enabled"

# All defenses
update_config "nq" 50 false false false true true true false
run_test "nq_11_all_defenses" "NQ: All defenses enabled"

# ==========================================
# 2. Dataset: PubMedQA
# ==========================================

# PubMedQA - All disabled
update_config "pubmedqa" 50 false false false false false false false
run_test "pubmedqa_01_all_disabled" "PubMedQA: All defenses, ADO, and attacks disabled"

# PubMedQA - Just MBA
update_config "pubmedqa" 50 false true false false false false false
run_test "pubmedqa_02_mba_only" "PubMedQA: Just MBA enabled"

# PubMedQA - MBA + DP
update_config "pubmedqa" 50 false true false true false false false
run_test "pubmedqa_03_mba_dp" "PubMedQA: MBA and DP enabled"

# PubMedQA - All attacks disabled, defense combinations
# Just DP
update_config "pubmedqa" 50 false false false true false false false
run_test "pubmedqa_04_dp_only" "PubMedQA: Just Differential Privacy enabled"

# Just TrustRAG
update_config "pubmedqa" 50 false false false false true false false
run_test "pubmedqa_05_trustrag_only" "PubMedQA: Just TrustRAG enabled"

# Just AV
update_config "pubmedqa" 50 false false false false false true false
run_test "pubmedqa_06_av_only" "PubMedQA: Just Attention Filtering enabled"

# DP + TrustRAG
update_config "pubmedqa" 50 false false false true true false false
run_test "pubmedqa_07_dp_trustrag" "PubMedQA: DP and TrustRAG enabled"

# DP + AV
update_config "pubmedqa" 50 false false false true false true false
run_test "pubmedqa_08_dp_av" "PubMedQA: DP and Attention Filtering enabled"

# TrustRAG + AV
update_config "pubmedqa" 50 false false false false true true false
run_test "pubmedqa_09_trustrag_av" "PubMedQA: TrustRAG and Attention Filtering enabled"

# All defenses
update_config "pubmedqa" 50 false false false true true true false
run_test "pubmedqa_10_all_defenses" "PubMedQA: All defenses enabled"

# ==========================================
# 3. Dataset: TriviaQA
# ==========================================

# TriviaQA - All disabled
update_config "triviaqa" 50 false false false false false false false
run_test "triviaqa_01_all_disabled" "TriviaQA: All defenses, ADO, and attacks disabled"

# TriviaQA - Just MBA
update_config "triviaqa" 50 false true false false false false false
run_test "triviaqa_02_mba_only" "TriviaQA: Just MBA enabled"

# TriviaQA - MBA + DP
update_config "triviaqa" 50 false true false true false false false
run_test "triviaqa_03_mba_dp" "TriviaQA: MBA and DP enabled"

# TriviaQA - All attacks disabled, defense combinations
# Just DP
update_config "triviaqa" 50 false false false true false false false
run_test "triviaqa_04_dp_only" "TriviaQA: Just Differential Privacy enabled"

# Just TrustRAG
update_config "triviaqa" 50 false false false false true false false
run_test "triviaqa_05_trustrag_only" "TriviaQA: Just TrustRAG enabled"

# Just AV
update_config "triviaqa" 50 false false false false false true false
run_test "triviaqa_06_av_only" "TriviaQA: Just Attention Filtering enabled"

# DP + TrustRAG
update_config "triviaqa" 50 false false false true true false false
run_test "triviaqa_07_dp_trustrag" "TriviaQA: DP and TrustRAG enabled"

# DP + AV
update_config "triviaqa" 50 false false false true false true false
run_test "triviaqa_08_dp_av" "TriviaQA: DP and Attention Filtering enabled"

# TrustRAG + AV
update_config "triviaqa" 50 false false false false true true false
run_test "triviaqa_09_trustrag_av" "TriviaQA: TrustRAG and Attention Filtering enabled"

# All defenses
update_config "triviaqa" 50 false false false true true true false
run_test "triviaqa_10_all_defenses" "TriviaQA: All defenses enabled"

# TriviaQA - Just poisoning
update_config "triviaqa" 50 false false true false false false false
run_test "triviaqa_11_poisoning_only" "TriviaQA: Just poisoning enabled"

# TriviaQA - Poisoning + TrustRAG
update_config "triviaqa" 50 false false true false true false false
run_test "triviaqa_12_poisoning_trustrag" "TriviaQA: Poisoning and TrustRAG enabled"

# TriviaQA - Poisoning + AV
update_config "triviaqa" 50 false false true false false true false
run_test "triviaqa_13_poisoning_av" "TriviaQA: Poisoning and Attention Filtering enabled"

# ==========================================
# 4. All datasets with both attacks + ADO
# ==========================================

# NQ - Both attacks + ADO
update_config "nq" 50 false true true false false false true
run_test "nq_12_both_attacks_ado" "NQ: Both attacks and ADO enabled"

# PubMedQA - Both attacks + ADO
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
