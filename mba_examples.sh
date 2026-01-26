#!/bin/bash
# Quick Start Guide for MBA Experiments
#
# This script demonstrates common MBA experiment scenarios

echo "MBA Attack - Quick Start Examples"
echo "=================================="
echo ""

# Example 1: Basic attack on NQ dataset
echo "Example 1: Basic MBA attack on NQ dataset"
echo "Command: ./run_mba_experiments.sh nq"
echo ""

# Example 2: Attack with custom parameters
echo "Example 2: Custom M and gamma values"
echo "Command: ./run_mba_experiments.sh pubmedqa --M 15 --gamma 0.6"
echo ""

# Example 3: Larger sample size
echo "Example 3: Test with 100 samples each"
echo "Command: ./run_mba_experiments.sh nq --num-members 100 --num-non-members 100"
echo ""

# Example 4: Force CPU (if GPU memory is insufficient)
echo "Example 4: Force CPU execution"
echo "Command: ./run_mba_experiments.sh nq --device cpu"
echo ""

echo "=================================="
echo "Note: Ensure dataset is ingested first:"
echo "  python scripts/ingest_data.py <dataset>"
echo ""
echo "Configure defenses in config/config.yaml before running"
echo "=================================="
