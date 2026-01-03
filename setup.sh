#!/bin/bash
set -e  # Exit on error

echo "============================================================"
echo " 🛠️  RAG Defence Utility - Setup Script"
echo "============================================================"

# 1. Environment Setup
echo "[1/4] Setting up environment (venv)..."

# Check if "env" directory exists
if [ -d "env" ]; then
    echo "Virtual environment 'env' already exists."
else
    echo "Creating virtual environment in 'env'..."
    python3 -m venv env
fi

# Activate environment
echo "Activating environment..."
source env/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Upgrade pip just in case
pip install --upgrade pip

echo "Environment active: $(which python)"

# 2. Install Dependencies
echo -e "\n[2/4] Installing dependencies..."
pip install -r requirements.txt
pip install -U langchain-huggingface  # Ensure this is present for local embeddings

# 3. Download Data
echo -e "\n[3/4] Downloading raw datasets..."
python scripts/download_datasets.py

# 4. Ingest Data
echo -e "\n[4/4] Ingesting data (Smart Indexing)..."
# Ingest all configured datasets
python scripts/ingest_data.py

echo -e "\n✅ Setup Complete!"
echo "To run experiments: ./run_eval.sh"
