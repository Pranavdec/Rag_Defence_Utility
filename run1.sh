#!/bin/bash

# Activate virtual environment
source env/bin/activate

# Run comprehensive evaluation
python scripts/comprehensive_eval.py 2>&1 | tee run1.log

# Keep track of exit code
exit ${PIPESTATUS[0]}
