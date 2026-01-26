# MBA Attack Implementation Summary

## Files Created

### Core Implementation
1. **src/attacks/__init__.py** - Module initialization
2. **src/attacks/mba.py** (658 lines) - Main MBA framework implementation
   - `MBAFramework` class with mask generation and attack execution
   - GPT-2 XL proxy model for word difficulty scoring
   - Spelling correction using `oliverguhr/spelling-correction-english-base`
   - Strategic mask placement algorithm
   - RAG response parsing and membership classification

### Experiment Runner
3. **scripts/run_mba_experiments.py** (327 lines) - Experiment orchestration
   - Loads real datasets using existing data loaders
   - Replicates ingestion seed to identify members vs non-members
   - Integrates with existing RAG pipeline (VectorStore, DefenseManager, Generator)
   - Computes attack metrics (accuracy, TPR, FPR, precision)
   - Saves detailed results to JSON

### Execution Scripts
4. **run_mba_experiments.sh** - Main execution script
5. **mba_examples.sh** - Quick reference examples

### Testing
6. **tests/test_mba_attack.py** (284 lines) - Comprehensive unit tests
   - Tests mask generation logic
   - Tests attack flow with mocked RAG
   - Tests accuracy calculation
   - Tests error handling
   - Tests helper functions

### Documentation
7. **MBA_README.md** - Complete usage guide and documentation

## Files Modified

1. **requirements.txt** - Added dependencies:
   - `transformers>=4.30.0` (for GPT-2 and spelling models)
   - `torch>=2.0.0` (for model execution)
   - `scikit-learn>=1.3.0` (for metrics)
   - `scipy>=1.11.0` (for statistical analysis)

## Key Features

### Real Dataset Integration
- Uses actual vector store (not mocked)
- Replicates ingestion sampling using `config.data.ingestion_seed`
- Correctly identifies member vs non-member documents
- Uses dataset specified in `config.yaml`

### Defense Testing
- Integrates with existing `DefenseManager`
- Tests attack success across different defense configurations
- Maintains compatibility with existing RAG pipeline

### Flexible Configuration
- Adjustable M (number of masks): default 10, optimal 5-15
- Adjustable γ (membership threshold): default 0.5
- Configurable sample sizes for members/non-members
- Device selection (auto/cuda/cpu)

### Comprehensive Metrics
- Attack accuracy (overall correctness)
- True Positive Rate (sensitivity)
- False Positive Rate
- Precision
- Confusion matrix
- Per-group mask prediction accuracy

## Usage Quick Start

```bash
# 1. Ensure dataset is ingested
python scripts/ingest_data.py nq

# 2. (Optional) Configure defenses in config/config.yaml

# 3. Run MBA attack
./run_mba_experiments.sh nq

# 4. View results
cat data/results/mba_nq_*.json
```

## Testing the Implementation

```bash
# Run unit tests
python -m pytest tests/test_mba_attack.py -v

# Or with unittest
python -m unittest tests/test_mba_attack.py
```

## Implementation Highlights

### Strategic Mask Selection
- Divides document into M sections
- Selects highest-ranked word per section
- Avoids adjacent masks (minimum 20 char spacing)
- Uses GPT-2 XL to score word difficulty

### Robust Parsing
- Multiple regex patterns for mask answer extraction
- Handles various response formats from different LLMs
- Fallback strategies for non-standard responses
- Substring matching for fuzzy answer comparison

### Memory Efficient
- Models moved to appropriate device (CPU/CUDA)
- Batch processing disabled in proxy scoring (to fit in memory)
- Graceful fallback if spelling model fails to load

## Architecture Integration

```
Config (config.yaml)
    ↓
Data Loaders (NQ/PubMed/Trivia)
    ↓
Member/Non-Member Split (using ingestion_seed)
    ↓
RAG System (VectorStore + DefenseManager + Generator)
    ↓
MBA Framework (mask generation → query → classify)
    ↓
Results (JSON with metrics)
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Ingest data**: `python scripts/ingest_data.py <dataset>`
3. **Run baseline**: `./run_mba_experiments.sh <dataset>` (no defenses)
4. **Enable defenses**: Edit `config/config.yaml`
5. **Run protected**: `./run_mba_experiments.sh <dataset>`
6. **Compare results**: Analyze attack success rate reduction

## Expected Behavior

- **No Defense**: Attack accuracy ~85-95% (strong signal)
- **Weak Defense**: Attack accuracy ~70-80% (moderate protection)
- **Strong Defense**: Attack accuracy ~45-55% (approaching random)

Success is measured by how close the attack accuracy is to 50% (random guessing).
