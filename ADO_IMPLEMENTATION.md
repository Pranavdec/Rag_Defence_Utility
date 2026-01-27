# Adaptive Defense Orchestration (ADO) - Implementation Details

## Overview
This document details the implementation of the ADO architecture, which transforms the static defense pipeline into a dynamic, stateful system capable of detecting and responding to sophisticated attacks like Membership Inference Attacks (MIA), Jailbreaking, and Data Poisoning.

## Core Components

### 1. Persistence Layer (`src/core/persistence.py`)
**Class:** `UserTrustManager`
- **Purpose**: Manages long-term user reputation.
- **Mechanism**:
    - JSON-based file storage in `data/users/{user_id}.json`.
    - Tracks `Global_Trust_Score` (0.0 - 1.0) per user.
    - Updates score based on Sentinel feedback.

### 2. Sensing Layer (`src/core/sensing.py`)
**Class:** `MetricsCollector`
- **Purpose**: Gathers objective "tight metrics" for analysis.
- **Metrics Implemented**:
    - **Pre-Retrieval**:
        - `M_LEX`: Lexical overlap (Jaccard) to detect repetitive probing.
        - `M_CMP`: Complexity score (special char ratio) to detect obfuscation.
        - `M_INT`: Intent velocity (time delta) to detect automated attacks.
    - **Retrieval**:
        - `M_DRP`: Score drop-off (Top-1 vs Top-5) to detect boundary probing.
        - `M_DIS`: Vector dispersion (variance) to detect conflicting contexts.

### 3. Intelligence Layer (`src/core/ado.py`)
**Class:** `Sentinel`
- **Role**: Intelligence Analyst.
- **Function**: Fuses inputs (Trust Score, Metrics, Query) into a `RiskProfile`.
- **Model**: Uses **Ollama** (default `llama3`) or internal HF model to reason about threats.
- **Output**: JSON containing `overall_threat_level`, `reasoning_trace`, and `specific_threats`.

**Class:** `Strategist`
- **Role**: Defense Commander.
- **Function**: Maps `RiskProfile` to a `DefensePlan`.
- **Logic**:
    - **Critical Risk / High MIA**: Enable `differential_privacy` (High Noise).
    - **High Poisoning**: Enable `trustrag` (Strict Filtering).
    - **High Jailbreak**: Enable `attention_filtering`.

### 4. Integration (`src/core/pipeline.py`)
**Class:** `ModularRAG`
- **Modifications**:
    - **Initialization**: Loads ADO components if `ado.enabled` is true.
    - **`run_single` Workflow**:
        1. **Identity**: Resolves `user_id`.
        2. **Sense & Reason**: Calls Sentinel/Strategist *before* retrieval.
        3. **Act**: Calls `defense_manager.set_dynamic_config()` to apply the Strategist's plan.
        4. **Execute**: Runs the standard retrieval/generation loop with active defenses.
        5. **Update**: Persists new Trust Score to disk.

## Configuration (`config/config.yaml`)
New `ado` section:
```yaml
ado:
  enabled: true
  user_id: "test_user_001"    # Default identity for testing
  sentinel_model: "llama3"    # Model to use for analysis
  strategist_model: "llama3"
  trust_score_decay: 0.05
```

## Defense Manager Updates (`src/defenses/manager.py`)
- Added `set_dynamic_config(defense_plan)`: Allows the runtime re-initialization of defense strategies based on the Strategist's output without restarting the application.
