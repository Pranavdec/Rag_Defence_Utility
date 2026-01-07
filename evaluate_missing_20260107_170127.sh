#!/bin/bash
# Auto-generated script to evaluate missing runs
# Generated on 2026-01-07 17:01:27
# Found 12 runs without evaluations

set -e

# Check if Ollama is running, if not start it
if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollama is not running. Starting Ollama service..."
    ollama serve &
    sleep 3  # Wait for service to start
    echo "Starting llama3 model..."
    ollama run llama3 &
    sleep 2  # Wait for model to load
    echo "✓ Ollama started successfully"
else
    echo "✓ Ollama is already running"
fi

echo '========================================='
echo 'Evaluating missing runs...'
echo '========================================='
echo ''
echo '[1/12] Evaluating: run_nq_20260106_233301.json'
python main.py evaluate 'data/results/run_nq_20260106_233301.json' || echo '✗ Failed: run_nq_20260106_233301.json'
echo ''
echo '[2/12] Evaluating: run_triviaqa_20260105_223750.json'
python main.py evaluate 'data/results/run_triviaqa_20260105_223750.json' || echo '✗ Failed: run_triviaqa_20260105_223750.json'
echo ''
echo '[3/12] Evaluating: run_triviaqa_20260105_222931.json'
python main.py evaluate 'data/results/run_triviaqa_20260105_222931.json' || echo '✗ Failed: run_triviaqa_20260105_222931.json'
echo ''
echo '[4/12] Evaluating: run_triviaqa_20260105_222026.json'
python main.py evaluate 'data/results/run_triviaqa_20260105_222026.json' || echo '✗ Failed: run_triviaqa_20260105_222026.json'
echo ''
echo '[5/12] Evaluating: run_nq_20260105_223131.json'
python main.py evaluate 'data/results/run_nq_20260105_223131.json' || echo '✗ Failed: run_nq_20260105_223131.json'
echo ''
echo '[6/12] Evaluating: run_nq_20260106_012916.json'
python main.py evaluate 'data/results/run_nq_20260106_012916.json' || echo '✗ Failed: run_nq_20260106_012916.json'
echo ''
echo '[7/12] Evaluating: run_pubmedqa_20260107_011540.json'
python main.py evaluate 'data/results/run_pubmedqa_20260107_011540.json' || echo '✗ Failed: run_pubmedqa_20260107_011540.json'
echo ''
echo '[8/12] Evaluating: run_pubmedqa_20260106_072704.json'
python main.py evaluate 'data/results/run_pubmedqa_20260106_072704.json' || echo '✗ Failed: run_pubmedqa_20260106_072704.json'
echo ''
echo '[9/12] Evaluating: run_triviaqa_20260106_072130.json'
python main.py evaluate 'data/results/run_triviaqa_20260106_072130.json' || echo '✗ Failed: run_triviaqa_20260106_072130.json'
echo ''
echo '[10/12] Evaluating: run_nq_20260105_221408.json'
python main.py evaluate 'data/results/run_nq_20260105_221408.json' || echo '✗ Failed: run_nq_20260105_221408.json'
echo ''
echo '[11/12] Evaluating: run_triviaqa_20260106_072832.json'
python main.py evaluate 'data/results/run_triviaqa_20260106_072832.json' || echo '✗ Failed: run_triviaqa_20260106_072832.json'
echo ''
echo '[12/12] Evaluating: run_nq_20260106_221344.json'
python main.py evaluate 'data/results/run_nq_20260106_221344.json' || echo '✗ Failed: run_nq_20260106_221344.json'
echo ''
echo '========================================='
echo 'All missing evaluations completed!'
echo 'Check data/metrics/ for results'
echo '========================================='