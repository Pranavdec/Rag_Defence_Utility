# Running Poisoning Attacks

This guide explains how to run the Corpus Poisoning attacks (PoisonedRAG/CorruptRAG).

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Quick Start
The easiest way to run the full verification loop (Baseline vs Defended) is using the shell script:

```bash
# Syntax: ./run_poisoning_attack.sh [dataset] [attack_type] [poison_rate]
./run_poisoning_attack.sh nq poisonedrag 0.01
```

This will:
1.  Run the attack **without defense** -> Expect High ASR (Attack Success Rate).
2.  Run the attack **with defenses** (TrustRAG) -> Expect Low ASR.

## Manual Execution (Python)

You can run the python script directly for more granular control.

### Arguments
*   `dataset`: `nq`, `pubmedqa`, or `triviaqa`.
*   `--attack_type`: `poisonedrag` (default) or `corruptrag`.
*   `--poison_rate`: Fraction of the corpus to poison (e.g., `0.01` for 1%).
*   `--defense`: `none` for baseline, `trustrag` to load defenses from `config/config.yaml`.
*   `--output_dir`: Directory to save JSON results.

### Examples

**1. Run PoisonedRAG Baseline (Vulnerability Test)**
```bash
python scripts/run_poisoning_attack.py nq \
    --attack_type poisonedrag \
    --poison_rate 0.01 \
    --defense none
```

**2. Test TrustRAG Defense Effectiveness**
```bash
python scripts/run_poisoning_attack.py nq \
    --attack_type poisonedrag \
    --poison_rate 0.01 \
    --defense trustrag
```

**3. Run CorruptRAG (Single Doc Injection)**
```bash
python scripts/run_poisoning_attack.py pubmedqa \
    --attack_type corruptrag \
    --poison_rate 0.005 \
    --defense none
```

## Understanding Results

The script outputs the **Attack Success Rate (ASR)**.

```text
==========================================
RESULTS: nq | poisonedrag
Poison Rate: 0.01
Defense: none
Attack Success Rate (ASR): 96.00%
==========================================
```

*   **ASR**: Percentage of target questions where the attack successfully forced the specific incorrect answer.
*   **Target Answer**: The attack sets the target answer to `Attack Successful: <pair_id>`. The evaluator checks if this string appears in the generated output.

## Output Files
Results are saved to `data/results/attack/`:
*   `*_targets.json`: The specific questions and target answers used.
*   `results_*.json`: Full inference results including generated answers, context, and metrics.
