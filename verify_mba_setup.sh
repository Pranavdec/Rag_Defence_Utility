#!/bin/bash
# Quick verification script to check MBA implementation

echo "MBA Implementation Verification"
echo "================================"
echo ""

echo "Checking created files..."
echo ""

files=(
    "src/attacks/__init__.py"
    "src/attacks/mba.py"
    "scripts/run_mba_experiments.py"
    "run_mba_experiments.sh"
    "tests/test_mba_attack.py"
    "MBA_README.md"
    "MBA_IMPLEMENTATION.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
    fi
done

echo ""
echo "Checking dependencies in requirements.txt..."
echo ""

required_deps=("transformers" "torch" "scikit-learn" "scipy")

for dep in "${required_deps[@]}"; do
    if grep -q "$dep" requirements.txt; then
        echo "✓ $dep"
    else
        echo "✗ $dep (missing)"
    fi
done

echo ""
echo "Checking Python syntax..."
python -m py_compile src/attacks/mba.py 2>/dev/null && echo "✓ mba.py syntax OK" || echo "✗ mba.py syntax error"
python -m py_compile scripts/run_mba_experiments.py 2>/dev/null && echo "✓ run_mba_experiments.py syntax OK" || echo "✗ run_mba_experiments.py syntax error"
python -m py_compile tests/test_mba_attack.py 2>/dev/null && echo "✓ test_mba_attack.py syntax OK" || echo "✗ test_mba_attack.py syntax error"

echo ""
echo "Checking script permissions..."
if [ -x "run_mba_experiments.sh" ]; then
    echo "✓ run_mba_experiments.sh is executable"
else
    echo "✗ run_mba_experiments.sh is not executable"
    echo "  Run: chmod +x run_mba_experiments.sh"
fi

echo ""
echo "================================"
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Ingest dataset: python scripts/ingest_data.py nq"
echo "  3. Run MBA attack: ./run_mba_experiments.sh nq"
echo "================================"
