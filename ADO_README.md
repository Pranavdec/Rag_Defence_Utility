# ADO (Adaptive Defense Orchestration) - User Guide

This system uses advanced AI reasoning to adapt security defenses in real-time.

## Prerequisites: Ollama

**Yes, you MUST run Ollama yourself.** The ADO `Sentinel` module connects to a local Ollama instance to analyze queries. The main RAG pipeline may similarly benefit from offloading tasks to Ollama.

### 1. Install & Run Ollama
If you haven't already:
1.  Download Ollama from [ollama.com](https://ollama.com).
2.  Start the server (usually runs in background or via terminal):
    ```bash
    ollama serve
    ```
3.  **Pull the Model**: The system is configured to use `llama3`.
    ```bash
    ollama pull llama3
    ```
    *(If you want to use a different model, update `sentinel_model` in `config/config.yaml`)*.

## Configuration

Check `config/config.yaml`:
```yaml
ado:
  enabled: true
  sentinel_model: "llama3"  # matches your ollama pull
  user_id: "test_user_001"  # useful for tracking trust scores
```

## Running the Simulation

A test script has been created to verify the adaptive behavior.

1.  **Navigate to the project root**.
2.  **Run the script**:
    ```bash
    python scripts/run_ado_testing.py
    ```

### What to Expect
1.  **Benign Query**:
    - The system calculates a Trust Score.
    - Risk Level should be `LOW` or `ELEVATED` (depending on initial score).
    - Standard defenses apply.
2.  **Attack Query (Probing)**:
    - The script sends a suspicious query (e.g., "Ignore previous instructions...").
    - **Sentinel** detects the intent and high complexity.
    - **Strategist** sets Risk Level to `CRITICAL`.
    - **Defense Manager** enables `differential_privacy` with strict settings.
    - **Persistence**: Your Trust Score drops in `data/users/sim_user_001.json`.

## Troubleshooting

- **Connection Refused (11434)**: Ensure `ollama serve` is running.
- **Model not found**: Run `ollama list` to see installed models and update `config/config.yaml` to match.
- **JSON Parse Error**: Sometimes smaller models (Llama3-8b) output messy JSON. If this happens often, check logs; the system has a safe fallback to `CRITICAL` mode on failure.
