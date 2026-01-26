# Corpus Poisoning Attack Implementation

## Overview
This module implements **PoisonedRAG** and **CorruptRAG** attacks against the RAG pipeline. These attacks work by injecting malicious documents into the knowledge base (corpus) to manipulate the LLM's answers for specific target questions.

*   **PoisonedRAG**: Injects multiple documents (e.g., 5 per target) optimized to be retrieved and containing a trigger for a specific incorrect answer.
*   **CorruptRAG**: A variant focusing on high-confidence single-document injection (often with "admin" or "system override" style prompts).

## File Structure

### Attack Core (`src/attacks/`)
*   **`__init__.py`**: Exports attack classes.
*   **`base_attack.py`**: Abstract base class defining the `generate_poisoned_corpus` interface.
*   **`poisonedrag_attack.py`**: Main implementation of PoisonedRAG.
    *   Generates optimized attack documents.
    *   Uses `LocalEmbedder` to verify retrieval potential (placeholder for gradient-based optimization).
*   **`corruptrag_attack.py`**: Implementation of CorruptRAG.
    *   Uses stronger prompt injection templates (e.g., "CONFIDENTIAL SYSTEM UPDATE").
*   **`utils.py`**: Helper functions for evaluation and target sampling.

### Execution Scripts (`scripts/`)
*   **`run_poisoning_attack.py`**: **Standalone orchestration script**.
    *   Does NOT modify `main.py` or existing evaluators.
    *   Handles:
        1.  **Target Sampling**: Picks random QA pairs from the dataset.
        2.  **Attack Generation**: Creates poisoned documents for targets.
        3.  **Ingestion**: Uses `ModularRAG` pipeline to ingest both clean and poisoned data.
        4.  **Inference**: Runs retrieval+generation for the target questions.
        5.  **Evaluation**: Computes Attack Success Rate (ASR) locally.

### Shell Wrapper
*   **`run_poisoning_attack.sh`**: Helper script to run baseline (no defense) and defended experiments sequentially.

## Key Features

### 1. Independent Execution
The attack logic is decoupled from the core application `main.py`. This ensures:
*   No accidental pollution of the main evaluation pipeline.
*   Ability to run destructive tests (corpus poisoning) without affecting the persistent vector store state of other experiments (assuming ephemeral or separate collection usage, though currently it appends to the active collection - **Note**: In a production test, you should ideally separate the ChromaDB collection).

### 2. Attack Success Rate (ASR) Metric
The script calculates ASR specifically for the injected targets:
*   **ASR**: The percentage of target questions where the generated answer contains the attacker's desired output (e.g., "Attack Successful: [ID]").

### 3. Modular Attack Types
Easily extensible class structure allowing for new attack strategies by inheriting from `BaseAttack`.

## Architecture Flow

```mermaid
graph TD
    A[Dataset (NQ/Trivia)] -->|Sample Targets| B[run_poisoning_attack.py]
    B -->|Generate| C[Attack Module (PoisonedRAG)]
    C -->|Poisoned Docs| D[Vector Store]
    A -->|Clean Docs| D
    B -->|Query Targets| E[RAG Pipeline]
    D -->|Retrieval| E
    E -->|Answer| F[ASR Evaluator]
```

## Integration with TrustRAG
The script supports testing against defenses:
*   `--defense none`: Runs with an empty defense manager.
*   `--defense trustrag`: Initializes the full `ModularRAG` pipeline which loads defenses from `config/config.yaml`.
