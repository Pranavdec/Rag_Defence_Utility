#!/bin/bash
# =============================================================================
# COMPREHENSIVE EVALUATION - QUICK REFERENCE
# =============================================================================
#
# PREREQUISITES:
#   1. Activate virtual environment:
#      source env/bin/activate
#
#   2. Start Ollama (for ADO modes):
#      ollama serve &
#      ollama pull llama3  # one-time download
#
#   3. Ingest data (if not done):
#      python scripts/ingest_data.py nq
#      python scripts/ingest_data.py pubmedqa
#      python scripts/ingest_data.py triviaqa
#
# =============================================================================
# MODES
# =============================================================================
#
# quick  - Fast validation (~2-5 min)
#          10 benign + 5 MBA + 5 poison
#          Output: Utility (DeepEval) + Attack metrics
#          Add --deepeval flag to compute utility metrics
#
# full   - Complete evaluation (~30+ min)
#          All 3 datasets × 6 defense combos
#          Output: Utility (DeepEval) + Attack metrics + Summary table
#          Add --deepeval flag for utility metrics
#
# mixed  - Realistic attack series (~10-15 min)
#          Customizable counts
#          Output: Utility (DeepEval) + Attack metrics
#          Add --deepeval flag for utility metrics
#
# utility - Answer quality only (~5-10 min)
#           No attacks, tests answer quality using DeepEval metrics
#           Output: DeepEval metrics (relevancy, faithfulness, etc.)
#           REQUIRES --deepeval flag
#
# attack  - Attack success only (~10 min)
#           Tests poisoning and/or MBA
#           Output: Attack metrics only
#
# =============================================================================
# DEFENSE COMBINATIONS (--defenses)
# =============================================================================
#
# STATIC DEFENSES (no Ollama needed):
#   none          - No defenses
#   dp            - Differential Privacy only
#   trustrag      - TrustRAG only
#   av            - Attention/Verification filtering only
#   dp_trustrag   - DP + TrustRAG
#   dp_av         - DP + AV
#   trustrag_av   - TrustRAG + AV
#   all_static    - DP + TrustRAG + AV
#
# ADO DEFENSES (requires Ollama):
#   ado_only      - Only ADO (dynamic, enables defenses based on threat)
#   ado_dp        - ADO + DP always on
#   ado_trustrag  - ADO + TrustRAG always on
#   ado_all       - ADO + all defenses always on (maximum protection)
#
# =============================================================================
# EXAMPLES
# =============================================================================

echo "Choose an example to run:"
echo ""
echo "1. Quick test (no ADO, no Ollama needed):"
echo "   ./run_comprehensive_eval.sh quick --defenses none --deepeval"
echo ""
echo "2. Quick test with ADO (needs Ollama):"
echo "   ./run_comprehensive_eval.sh quick --defenses ado_only --deepeval"
echo ""
echo "3. Compare no defense vs ADO:"
echo "   ./run_comprehensive_eval.sh mixed --defenses none,ado_all --num-benign 20 --num-poison 10 --num-mba 10 --deepeval"
echo ""
echo "4. Utility sweep on multiple datasets (REQUIRES --deepeval):"
echo "   ./run_comprehensive_eval.sh utility --datasets nq,pubmedqa --defenses none,dp,trustrag,dp_trustrag --deepeval"
echo ""
echo "5. Attack-only evaluation:"
echo "   ./run_comprehensive_eval.sh attack --attack-type both --defenses none,ado_only --num-poison 10 --num-mba 10"
echo ""
echo "6. Full evaluation (all datasets, key defenses):"
echo "   ./run_comprehensive_eval.sh full --num-benign 30 --num-poison 15 --num-mba 10"
echo ""
echo "7. Test specific defense combo:"
echo "   ./run_comprehensive_eval.sh mixed --dataset pubmedqa --defenses trustrag_av --num-benign 15 --num-poison 8"
echo ""

# =============================================================================
# COMMAND LINE OPTIONS
# =============================================================================
#
# --dataset       Single dataset: nq, pubmedqa, triviaqa (default: nq)
# --datasets      Multiple datasets: nq,pubmedqa,triviaqa
# --defenses      Defense combos: none,ado_all,dp_trustrag (comma-separated)
# --num-benign    Benign queries count (default: 20)
# --num-poison    Poisoning targets count (default: 10)
# --num-mba       MBA samples count (default: 10)
# --attack-type   For attack mode: poisoning, mba, both
# --output-dir    Results directory (default: data/results/comprehensive_eval)
#
# =============================================================================
# OUTPUT INTERPRETATION
# =============================================================================
#
# UTILITY METRICS (DeepEval - requires --deepeval flag):
#   Answer Relevancy = How relevant the answer is to the question (0-1)
#   Faithfulness = How well answer aligns with retrieved context (0-1)
#   Contextual Relevancy = How relevant retrieved docs are (0-1)
#   Contextual Recall = Coverage of ground truth in retrieval (0-1)
#   Goal: HIGH scores (0.7+ is good, 0.8+ is excellent)
#
# ATTACK METRICS:
#   Poisoning ASR = % of poisoned queries where attack succeeded
#   MBA ASR = % of membership inference attacks that succeeded
#   Goal: LOW (defenses should reduce ASR)
#
# ADO METRICS:
#   Trust Score = User trust level (0-1, starts at 0.5)
#   Risk Levels = How ADO classified queries (LOW/ELEVATED/CRITICAL)
#   Defenses Triggered = How often each defense was activated by ADO
#
# =============================================================================
# TYPICAL WORKFLOW FOR REPORTING
# =============================================================================
#
# Step 1: Start Ollama
#   ollama serve &
#
# Step 2: Quick validation (with DeepEval utility metrics)
#   ./run_comprehensive_eval.sh quick --defenses none --deepeval
#
# Step 3: Run comparison for paper/report
#   ./run_comprehensive_eval.sh mixed --defenses none,dp_trustrag,all_static,ado_only,ado_all \
#       --num-benign 50 --num-poison 20 --num-mba 20 --deepeval
#
# Step 4: Check results
#   ls data/results/comprehensive_eval/
#   cat data/results/comprehensive_eval/full_eval_*.json
#
# =============================================================================
