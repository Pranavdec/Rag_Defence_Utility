# RAG Defence Utility

A modular, reproducible RAG (Retrieval Augmented Generation) pipeline designed for quantifying utility and robustness. Supports **Smart Indexing**, **Local Embeddings** (sentence-transformers), and **Local Evaluation** (RAGAS with Ollama).

## Features
*   **Smart Indexing**: Indexes only the "gold passages" relevant to test questions, enabling fast and valid local testing.
*   **Fully Local**: uses `sentence-transformers` for embeddings and `Ollama` (Llama 3) for generation and evaluation.
*   **Reproducible**: Seeded random sampling (`ingestion_seed`, `test_seed`) ensures consistent train/test splits.
*   **Multi-Dataset**: Native support for Natural Questions (NQ), PubMedQA, TriviaQA, and FinanceBench.

## 🔗 Prerequistes
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running (`ollama serve`).
*   **Pull Models**:
    ```bash
    ollama pull llama3
    ```
*   **Hugging Face Login** (when using the Hugging Face generation backend or gated models):
    ```bash
    hf auth login
    # When prompted, enter your Hugging Face token if your chosen model requires it.
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

The default entrypoint is config-driven and performs ingestion plus evaluation:

```bash
python scripts/comprehensive_eval.py --config config/config.yaml
```

To switch datasets, set `data.dataset` in `config/config.yaml` to one of:
`nq`, `pubmedqa`, `triviaqa`, `financebench`.

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
