#!/bin/bash
# Comprehensive RAG Defense & Attack Evaluation
#
# Usage:
#   ./run_comprehensive_eval.sh [mode] [options]
#
# Modes:
#   quick   - Fast test on single dataset (5 benign, 5 attack)
#   full    - Complete evaluation on all datasets with all defense combos
#   utility - Test answer quality across defense configurations
#   attack  - Test attack success rates
#   mixed   - Run mixed series (benign + MBA + poisoning)
#
# Examples:
#   ./run_comprehensive_eval.sh quick
#   ./run_comprehensive_eval.sh full
#   ./run_comprehensive_eval.sh mixed --dataset nq --num-benign 20 --num-poison 10
#   ./run_comprehensive_eval.sh utility --datasets nq,pubmedqa --defenses none,ado_all
#   ./run_comprehensive_eval.sh attack --attack-type poisoning --defenses none,ado_only

set -e

# Auto-activate venv
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
fi

# Check Ollama for ADO
check_ollama() {
    if ! pgrep -x "ollama" > /dev/null; then
        echo "⚠️  Ollama not running. ADO features need Ollama."
        echo "   Run: ollama serve & ollama pull llama3"
        read -p "   Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Ollama is running"
    fi
}

# Print banner
echo "=========================================="
echo "Comprehensive RAG Defense Evaluation"
echo "=========================================="
echo ""

MODE=${1:-mixed}
shift || true

case $MODE in
    quick)
        echo "Mode: QUICK TEST"
        echo "Testing with minimal queries for fast validation"
        check_ollama
        python scripts/comprehensive_eval.py --mode quick --dataset nq "$@"
        ;;
    full)
        echo "Mode: FULL EVALUATION"
        echo "Testing all datasets × all defense combinations"
        echo "This will take a while..."
        check_ollama
        python scripts/comprehensive_eval.py --mode full "$@"
        ;;
    utility)
        echo "Mode: UTILITY EVALUATION"
        echo "Testing answer quality across configurations"
        check_ollama
        python scripts/comprehensive_eval.py --mode utility "$@"
        ;;
    attack)
        echo "Mode: ATTACK EVALUATION"
        echo "Testing attack success rates"
        check_ollama
        python scripts/comprehensive_eval.py --mode attack "$@"
        ;;
    mixed)
        echo "Mode: MIXED SERIES"
        echo "Running realistic attack simulation"
        check_ollama
        python scripts/comprehensive_eval.py --mode mixed "$@"
        ;;
    help|--help|-h)
        echo "Usage: ./run_comprehensive_eval.sh [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  quick   - Fast test (~2 min)"
        echo "  full    - Complete evaluation (~30+ min)"
        echo "  utility - Utility sweep"
        echo "  attack  - Attack evaluation"
        echo "  mixed   - Mixed attack series"
        echo ""
        echo "Options:"
        echo "  --dataset       Single dataset (nq, pubmedqa, triviaqa)"
        echo "  --datasets      Comma-separated datasets"
        echo "  --defenses      Comma-separated defense combos"
        echo "  --num-benign    Number of benign queries"
        echo "  --num-poison    Number of poisoning targets"
        echo "  --num-mba       Number of MBA samples"
        echo "  --attack-type   poisoning, mba, or both"
        echo ""
        echo "Defense Combos:"
        echo "  none, dp, trustrag, av, all_static"
        echo "  ado_only, ado_dp, ado_trustrag, ado_all"
        echo ""
        echo "Examples:"
        echo "  ./run_comprehensive_eval.sh quick"
        echo "  ./run_comprehensive_eval.sh mixed --defenses none,ado_all"
        echo "  ./run_comprehensive_eval.sh full"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Run: ./run_comprehensive_eval.sh help"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: data/results/comprehensive_eval/"
echo "=========================================="
