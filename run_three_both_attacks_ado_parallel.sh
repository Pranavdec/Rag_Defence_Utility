#!/bin/bash

# Run only the 3 "both attacks + ADO" experiments with identical settings,
# while isolating runtime state so two datasets can run in parallel safely.

set -uo pipefail

source /media/crk/datastore_1/Rag_Defence_Utility/env/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${TIMESTAMP}_both_attacks_ado_parallel"
RUNTIME_CONFIG_DIR="config/runtime_configs_${TIMESTAMP}"
STATE_ROOT="data/runtime_state_${TIMESTAMP}"

mkdir -p "${LOG_DIR}" "${RUNTIME_CONFIG_DIR}" "${STATE_ROOT}"

echo "========================================" | tee "${LOG_DIR}/summary.log"
echo "Starting 3-run parallel suite at $(date)" | tee -a "${LOG_DIR}/summary.log"
echo "Log directory: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "Runtime config directory: ${RUNTIME_CONFIG_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "State root: ${STATE_ROOT}" | tee -a "${LOG_DIR}/summary.log"
echo "========================================" | tee -a "${LOG_DIR}/summary.log"
echo "" | tee -a "${LOG_DIR}/summary.log"

create_config() {
    local dataset="$1"
    local cfg_path="$2"
    local chroma_dir="$3"
    local results_dir="$4"
    local user_data_dir="$5"

    mkdir -p "${results_dir}" "${user_data_dir}" "${chroma_dir}"

    cat > "${cfg_path}" << EOF
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
  chroma_db: ${chroma_dir}
  results: ${results_dir}
  cache: data/raw
  user_data: ${user_data_dir}

data:
  dataset: ${dataset}
  ingestion_size: 700
  ingestion_seed: 42
  test_size: 50
  test_seed: 123

evaluation:
  skip_deepeval: false
  deepeval_max_concurrent: 5

retrieval:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50

defenses:
- name: differential_privacy
  enabled: false
  method: dp_approx
  epsilon: 3.0
  delta: 0.01
  candidate_multiplier: 3
- name: trustrag
  enabled: false
  similarity_threshold: 0.88
  rouge_threshold: 0.25
  candidate_multiplier: 3
- name: attention_filtering
  enabled: false
  model_path: meta-llama/Llama-3.1-8B-Instruct
  top_tokens: 100
  max_corruptions: 3
  threshold: 50
  device: cuda
  candidate_multiplier: 3

attack:
  mba:
    enabled: true
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
    enabled: true
    poisoning_rate: 10
    num_targets: 50
    seed: 42
    target_start_index: 0
    diversity_level: true

ado:
  enabled: true
  user_id: test_user_001
  sentinel_model: llama3
  strategist_model: llama3
  strategist_mode: llm
  trust_score_decay: 0.05
EOF
}

run_dataset() {
    local dataset="$1"
    local run_name="$2"

    local cfg_path="${RUNTIME_CONFIG_DIR}/${run_name}.yaml"
    local chroma_dir="${STATE_ROOT}/${dataset}/chroma_db"
    local results_dir="${STATE_ROOT}/${dataset}/results"
    local user_data_dir="${STATE_ROOT}/${dataset}/users"
    local log_path="${LOG_DIR}/${run_name}.log"

    create_config "${dataset}" "${cfg_path}" "${chroma_dir}" "${results_dir}" "${user_data_dir}"

    echo "----------------------------------------" | tee -a "${LOG_DIR}/summary.log"
    echo "Running ${run_name}" | tee -a "${LOG_DIR}/summary.log"
    echo "Config: ${cfg_path}" | tee -a "${LOG_DIR}/summary.log"
    echo "Log: ${log_path}" | tee -a "${LOG_DIR}/summary.log"
    echo "Started at: $(date)" | tee -a "${LOG_DIR}/summary.log"
    echo "----------------------------------------" | tee -a "${LOG_DIR}/summary.log"

    python scripts/comprehensive_eval.py --config "${cfg_path}" > "${log_path}" 2>&1
    local status=$?

    if [[ ${status} -eq 0 ]]; then
        echo "${run_name} completed successfully at $(date)" | tee -a "${LOG_DIR}/summary.log"
    else
        echo "${run_name} failed with exit code ${status} at $(date)" | tee -a "${LOG_DIR}/summary.log"
    fi
    echo "" | tee -a "${LOG_DIR}/summary.log"

    return ${status}
}

# 2-way parallel batch: NQ + PubMedQA
run_dataset "nq" "nq_12_both_attacks_ado" &
PID_NQ=$!

run_dataset "pubmedqa" "pubmedqa_11_both_attacks_ado" &
PID_PUB=$!

wait ${PID_NQ}
STATUS_NQ=$?
wait ${PID_PUB}
STATUS_PUB=$?

# Final run sequentially to avoid overloading model memory
run_dataset "triviaqa" "triviaqa_14_both_attacks_ado"
STATUS_TRI=$?

echo "========================================" | tee -a "${LOG_DIR}/summary.log"
echo "All requested runs finished at $(date)" | tee -a "${LOG_DIR}/summary.log"
echo "Statuses: nq=${STATUS_NQ}, pubmedqa=${STATUS_PUB}, triviaqa=${STATUS_TRI}" | tee -a "${LOG_DIR}/summary.log"
echo "Logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "Configs: ${RUNTIME_CONFIG_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "State root: ${STATE_ROOT}" | tee -a "${LOG_DIR}/summary.log"
echo "========================================" | tee -a "${LOG_DIR}/summary.log"

if [[ ${STATUS_NQ} -ne 0 || ${STATUS_PUB} -ne 0 || ${STATUS_TRI} -ne 0 ]]; then
    exit 1
fi

exit 0
