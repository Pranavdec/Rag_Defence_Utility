# RAG Defence Utility

A modular, reproducible RAG (Retrieval Augmented Generation) pipeline designed for quantifying utility and robustness. Supports **Smart Indexing**, **Local Embeddings** (sentence-transformers), and **Local Evaluation** (RAGAS with Ollama).

## Features
*   **Smart Indexing**: Indexes only the "gold passages" relevant to test questions, enabling fast and valid local testing.
*   **Fully Local**: uses `sentence-transformers` for embeddings and `Ollama` (Llama 3) for generation and evaluation.
*   **Reproducible**: Seeded random sampling (`ingestion_seed`, `test_seed`) ensures consistent train/test splits.
*   **Multi-Dataset**: Native support for Natural Questions (NQ), PubMedQA, and TriviaQA.

## 🔗 Prerequistes
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running (`ollama serve`).
*   **Pull Models**:
    ```bash
    ollama pull llama3
    ```
*   **Hugging Face Login** (Required for AV Defense):
    The *Attention-based Verification (AV)* defense uses a local Llama 3 model which requires authentication.
    ```bash
    hf auth login
    # When prompted, enter your Hugging Face token.
    # Ensure you have access to meta-llama/Llama-3.1-8B-Instruct.
    ```


## 🛠️ Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Check `config/config.yaml` to adjust sample sizes or models.
    ```yaml
    data:
      ingestion_size: 500  # Number of QA pairs to index
      test_size: 50        # Number of samples to test
    ```

## 🚀 Usage

The pipeline follows a 3-step workflow: **Ingest → Run → Evaluate**.

### Step 1: Ingest Data
Downloads datasets (if needed), indexes gold passages, and saves QA pairs.
*Auto-clears previous data to ensure a clean state.*

```bash
# Ingest all datasets (NQ, PubMedQA, TriviaQA)
python scripts/ingest_data.py

# Or specific dataset
python scripts/ingest_data.py nq
```

### Step 2: Run Inference
Runs retrieval and generation on the test set. Does **not** re-ingest.

```bash
python main.py run nq
python main.py run pubmedqa
python main.py run triviaqa
```
*Results saved to: `data/results/run_<dataset>_<timestamp>.json`*

### Step 3: Evaluate
Runs metrics (Faithfulness, Answer Correctness, Latency) using RAGAS with a local Ollama judge.

```bash
python main.py evaluate data/results/run_nq_20260103_xxxx.json
```
*Outputs:*
*   JSON Report: `data/metrics/eval_nq_<timestamp>.json`
*   CSV Table: `data/metrics/eval_nq_<timestamp>_metrics.csv`

## 📊 Metrics
*   **Latency**: End-to-end generation time (ms).
*   **Refusal Rate**: Heuristic detection of refusals ("I cannot answer...").
*   **RAGAS Metrics**:
    *   `faithfulness`: Is the answer derived from the context?
    *   `answer_correctness`: Does it match the ground truth?
    *   `context_recall`: Was the relevant context retrieved?

## Project Structure
*   `src/core`: Pipeline logic (Retrieval, Generation).
*   `src/data_loaders`: Dataset processing (NQ, PubMedQA, TriviaQA).
*   `src/evaluation`: RAGAS integration via LangChain/Ollama.
*   `scripts/`: Ingestion and download utilities.
