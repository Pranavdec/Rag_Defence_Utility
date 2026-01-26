# MBA (Membership Inference Attack) Framework

This implementation adds a Mask-Based Attack (MBA) framework to test the membership inference vulnerability of RAG systems against different defense configurations.

## Overview

The MBA framework operates in two phases:

1. **Mask Generation**: Strategically masks M words in a target document using:
   - Fragmented word extraction (words split by tokenizer)
   - Optional misspelling correction (disabled by default to save memory)
   - Proxy language model scoring (GPT-2 XL by default) to identify hard-to-predict words
   - Strategic mask placement across document sections

2. **Membership Inference**: Queries the RAG system with masked documents and classifies membership based on prediction accuracy against threshold γ.

## Configuration

MBA settings are configured in `config/config.yaml`:

```yaml
attack:
  mba:
    M: 10  # Number of masks per document (optimal: 5-15)
    gamma: 0.5  # Membership threshold (accuracy > gamma = member)
    num_members: 50  # Number of member documents to test
    num_non_members: 50  # Number of non-member documents to test
    device: auto  # cuda, cpu, or auto
    proxy_model: gpt2-xl  # Proxy model for word difficulty scoring
    enable_spelling_correction: false  # Disable spelling correction (saves memory)
```

### Configuration Options

- **M**: Number of masks (5-15 optimal, default: 10)
- **gamma**: Membership threshold (default: 0.5)
- **num_members**: Test sample size for members (default: 50)
- **num_non_members**: Test sample size for non-members (default: 50)
- **device**: `cuda`, `cpu`, or `auto` (default: auto)
- **proxy_model**: Model for difficulty scoring (default: `gpt2-xl`, can use `gpt2`, `gpt2-medium`, `gpt2-large`)
- **enable_spelling_correction**: Enable spelling model (default: false, saves ~2GB memory)

## Files Added

- **`src/attacks/mba.py`**: Core MBA framework implementation
- **`scripts/run_mba_experiments.py`**: Experiment runner that uses real datasets
- **`run_mba_experiments.sh`**: Bash script for easy execution
- **`tests/test_mba_attack.py`**: Unit tests with mocked components

## Installation

Install the new dependencies:

```bash
pip install transformers torch
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

**Note**: Spelling correction dependencies are optional and disabled by default.

## Usage

### Basic Usage (using config.yaml settings)

Run MBA experiments on a dataset (must be ingested first):

```bash
# Run on NQ dataset with settings from config.yaml
./run_mba_experiments.sh nq

# Run on PubMedQA
./run_mba_experiments.sh pubmedqa

# Run on TriviaQA
./run_mba_experiments.sh triviaqa
```

### Override Config with Command Line Arguments

You can override config.yaml settings using command-line arguments:

```bash
# Override M and gamma
./run_mba_experiments.sh nq --M 15 --gamma 0.6

# Override sample sizes
./run_mba_experiments.sh pubmedqa --num-members 100 --num-non-members 100

# Use smaller proxy model (faster, less accurate)
./run_mba_experiments.sh nq --proxy-model gpt2-medium

# Force CPU execution
./run_mba_experiments.sh nq --device cpu
```

### Command Line Options (all optional, defaults from config.yaml)

- `dataset`: Dataset to attack (`nq`, `pubmedqa`, or `triviaqa`)
- `--M`: Number of masks per document
- `--gamma`: Accuracy threshold for membership classification
- `--num-members`: Number of member documents to test
- `--num-non-members`: Number of non-member documents to test
- `--device`: Device for models: `cuda`, `cpu`, or `auto`
- `--proxy-model`: Proxy model name (e.g., `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`)

### Prerequisites

Before running MBA experiments:

1. **Ingest a dataset** to populate the vector store:
   ```bash
   python scripts/ingest_data.py nq
   ```

2. **Configure defenses** in `config/config.yaml` to test different protection mechanisms

## How It Works

### Member vs Non-Member Split

The script uses the **exact same random seed** (`config.data.ingestion_seed`) used during data ingestion to:

1. Identify which documents are currently in the vector store (**Members**)
2. Sample a disjoint set of documents not in the vector store (**Non-Members**)

This ensures a valid membership inference evaluation without mocking the vector store.

### Attack Flow

For each test document:

1. **Mask Generation**:
   - Extract candidate words (excluding stop words)
   - Score words using GPT-2 XL proxy model
   - Select M highest-scoring words distributed across document sections
   - Replace selected words with `[Mask_1]`, `[Mask_2]`, etc.

2. **RAG Query**:
   - Submit masked document to RAG system
   - Request predictions for each mask

3. **Membership Decision**:
   - Compare predicted words with ground truth
   - Calculate accuracy = correct_predictions / M
   - Classify as member if accuracy > γ

### Metrics Reported

- **Attack Accuracy**: Overall correct membership classifications
- **True Positive Rate (TPR)**: Members correctly identified
- **False Positive Rate (FPR)**: Non-members incorrectly identified as members
- **Precision**: Accuracy of positive predictions
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Average Mask Prediction Accuracy**: Per-group accuracy for filled masks

## Results

Results are saved to `data/results/mba_{dataset}_{timestamp}.json` with:

```json
{
  "config": {
    "dataset": "nq",
    "M": 10,
    "gamma": 0.5,
    ...
  },
  "member_results": [...],
  "non_member_results": [...],
  "summary": {
    "attack_accuracy": 0.85,
    "true_positive_rate": 0.88,
    "false_positive_rate": 0.18,
    ...
  }
}
```

## Testing Different Defenses

To evaluate how defenses protect against MBA attacks:

1. **Edit `config/config.yaml`** to enable/configure defenses:
   ```yaml
   defenses:
     - name: differential_privacy
       enabled: true
       epsilon: 3.0
   ```

2. **Configure MBA settings** (if needed):
   ```yaml
   attack:
     mba:
       M: 10
       gamma: 0.5
       num_members: 50
       num_non_members: 50
   ```

3. **Run MBA experiment**:
   ```bash
   ./run_mba_experiments.sh nq
   ```

4. **Compare results** across different defense configurations

## Key Hyperparameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| **M** | 5-15 | Number of masks (optimal range per paper) |
| **γ** | 0.5 | Prediction accuracy threshold |
| **Proxy Model** | gpt2-xl | For difficulty scoring (gpt2-medium for speed) |
| **Device** | auto | Use CUDA if available |
| **Spelling** | false | Disabled by default (saves ~2GB memory) |

## Notes

- **Memory Requirements**: 
  - GPT-2 XL: ~5GB RAM/VRAM
  - GPT-2 Large: ~3GB RAM/VRAM
  - GPT-2 Medium: ~1.5GB RAM/VRAM
  - GPT-2: ~500MB RAM/VRAM
- **Spelling Correction**: Disabled by default, saves memory with minimal accuracy loss
- **Dataset Size**: Ensure `ingestion_size < total_dataset_size` to have non-members
- **Defense Impact**: Strong defenses should reduce attack accuracy close to random (0.5)

## Example Output

```
========================================
MBA MEMBERSHIP INFERENCE ATTACK
========================================
Dataset: nq
Masks per document (M): 10
Members to test: 50
Non-members to test: 50
========================================

Attack Success Metrics:
  Attack Accuracy: 0.850 (85/100)
  True Positive Rate: 0.880
  False Positive Rate: 0.180
  Precision: 0.830

Confusion Matrix:
  True Positives:   44 (member correctly identified)
  False Negatives:   6 (member missed)
  False Positives:   9 (non-member incorrectly identified)
  True Negatives:   41 (non-member correctly identified)
```

## References

Based on the Mask-Based Attack (MBA) framework for membership inference in RAG systems. The attack leverages strategic word masking and RAG system responses to infer document membership in the vector store.
