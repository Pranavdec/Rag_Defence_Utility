# Implementation Plan: Config-Driven Comprehensive Evaluation Script

## Goal
Create a unified, configuration-driven script (`scripts/comprehensive_eval.py`) that handles data ingestion, vector DB management, and evaluation (benign + attacks) based on `config.yaml` settings. It will support both ADO-enabled (mixed traffic) and ADO-disabled (sequential) modes.

## User Review Required
> [!IMPORTANT]
> This script will **DELETE** the existing Vector DB at the start of every run to ensure a clean state as requested. Ensure importance data is backed up.

## Proposed Changes

### Script: `scripts/comprehensive_eval.py`
Refactor or rewrite the existing script to remove CLI arguments and rely solely on `config.yaml`.

#### Workflow Description
1.  **Initialization**
    *   Load `config/config.yaml`.
    *   Create a timestamped results folder: `data/results/<timestamp>_eval/`.
    *   **Action**: Save a copy of `config.yaml` into this folder.
    *   **Action**: Clear Vector DB (remove `data/chroma_db` directory).

2.  **Ingestion**
    *   Initialize `ModularRAG` pipeline.
    *   Call `rag.ingest()` (automatically reads dataset, size, seed from config).

3.  **Execution Logic**
    *   **Condition**: Check `config['ado']['enabled']`.

    #### Path A: ADO Disabled (Sequential Evaluation)
    1.  **Benign Evaluation**:
        *   Load benign QA pairs (count = `config['data']['test_size']`).
        *   Run queries.
        *   Calculate DeepEval metrics (Answer Relevancy, Faithfulness, etc.).
    2.  **Attack Evaluation** (if enabled in config):
        *   **Poisoning**:
            *   If `config['attack']['poisoned_rag']['enabled']`:
                *   Inject poison (using `rag.vector_store`).
                *   Run target queries.
                *   Calculate Success Rate.
                *   Clean poison after (clean it).
        *   **MBA**:
            *   If `config['attack']['mba']['enabled']`:
                *   Run MBA attack inference.
                *   Calculate Success Rate.
    3.  **Reporting**: Save merged results to the results folder.

    #### Path B: ADO Enabled (Mixed Traffic Evaluation)
    -- clear any files which have previous session stored data -- 
    1.  **Preparation**:
        *   **Benign**: Load `config['data']['test_size']` benign queries.
        *   **Attacks**:
            *   If `config['attack']['poisoned_rag']['enabled']`: Generate partial poison payloads/queries.
            *   If `config['attack']['mba']['enabled']`: Generate MBA probing queries.
    2.  **Execution**:
        *   Combine all queries (Benign + Poison + MBA).
        *   **Shuffle** the combined list to simulate realistic mixed traffic.
        *   Run queries sequentially through the pipeline (ADO will react to the stream).
    3.  **Reporting**:
        *   Calculate ASR (Attack Success Rate) for attacks.
        *   Calculate DeepEval metrics for benign queries.
        *   Save comprehensive log to the results folder.

## Verification Plan

### Automated Verification
*   **Run ADO Disabled**: `python scripts/comprehensive_eval.py` (with `ado.enabled=false` in config). Verify separate benign and attack metrics are generated.
*   **Run ADO Enabled**: `python scripts/comprehensive_eval.py` (with `ado.enabled=true` in config). Verify mixed stream execution and ADO metric logs.

### Manual Verification
*   Check `data/results/` for the new folder.
*   Verify `config.yaml` copy exists inside.
*   Verify Vector DB was reset (logs should show ingestion starting from scratch).
