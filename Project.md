# Project Specification: Phase 1 Utility Baseline

**Project Name:** Quantifying Alignment Tax in Defensive RAG
**Phase:** 1 - Baseline Infrastructure (Core, Data, Eval)
**Goal:** Establish a modular RAG pipeline capable of ingesting your specific diverse datasets (General, Medical, Complex Reasoning) and measuring baseline performance (Accuracy, Recall, Latency, Refusal Rate) to serve as the control group (0 Defenses) for future experiments.

---

## 1. System Architecture

The system follows a modular "Pipeline" architecture. The **Core** orchestrates the flow, while **Data Loaders** normalize the distinct structures of NQ, PubMedQA, and TriviaQA into a unified format.

### High-Level Data Flow

1. **Ingestion (Smart Indexing):** Raw Dataset -> `Data Loader` -> Sampled QA Pairs -> Index **Gold Passages Only** -> ChromaDB.
2. **Inference (Online):** Test Question -> `Core Pipeline` -> Retrieval (Top-K) + Generation (Local LLM) -> Prediction.
3. **Evaluation (Post-Hoc):** Logs -> `Eval Module` (RAGAS with Local Judge) -> Metrics (JSON/CSV).

---

## 2. Directory Structure

```
rag-defense-utility/
├── config/
│   └── config.yaml              # Global settings (embedding model, ingestion_size, test_size)
│
├── data/                        # Local storage
│   ├── raw/                     # Cached HuggingFace datasets
│   ├── qa_pairs/                # Usage: Saved QA pairs for valid testing
│   ├── chroma_db/               # Persisted VectorDB on disk
│   ├── results/                 # JSON outputs of experiments
│   └── metrics/                 # Evaluation reports (JSON + CSV)
│
├── scripts/
│   ├── download_datasets.py     # Download raw data to cache
│   └── ingest_data.py           # Ingest sampled gold passages
│
├── src/
│   ├── core/
│   │   ├── pipeline.py          # Main RAG orchestration
│   │   ├── retrieval.py         # ChromaDB wrapper (Local Embeddings)
│   │   └── generation.py        # Ollama LLM wrapper
│   │
│   ├── data_loaders/            # Dataset-specific logic
│   │   ├── base_loader.py
│   │   ├── nq_loader.py
│   │   ├── pubmed_loader.py
│   │   └── trivia_loader.py
│   │
│   └── evaluation/
│       └── evaluator.py         # RAGAS runner (using Ollama judge)
│
└── main.py                      # CLI runner (run, evaluate)
```

---

## 3. Module Specifications

### A. Data Loaders (Smart Indexing)

**Strategy**: Instead of indexing the entire corpus (millions of docs), we use **Smart Indexing**:
1. Load all available QA pairs.
2. Randomly sample `ingestion_size` (e.g., 500) pairs using `ingestion_seed`.
3. Index **only** the gold passages associated with these questions.
4. Save the QA pairs to `data/qa_pairs/` for testing.
5. In testing, sample `test_size` (e.g., 50) from these known pairs using `test_seed`.

**Goal**: Ensures tests are valid (answer known to be in index) and run fast locally.

### B. Core RAG (`src/core`)

**`retrieval.py` Specification:**
*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local).
*   **Batching**: Supports high-throughput batch embedding (64+ docs at once).
*   **Storage**: ChromaDB local persistence.

**`generation.py` Specification:**
*   **Provider**: Ollama.
*   **Model**: Configurable (default: `llama3`).
*   **Params**: `temperature=0.0` for reproducibility.

### C. Evaluation (`src/evaluation`)

**`evaluator.py` Specification:**
*   **Framework**: RAGAS (Retrieval Augmented Generation Assessment).
*   **Judge LLM**: Local Ollama model (configured via `judge_llm`).
*   **Embeddings**: Same local model as ingestion (`all-MiniLM-L6-v2`).
*   **Metrics**:
    *   `faithfulness` (RAGAS)
    *   `answer_correctness` (RAGAS)
    *   `context_recall` (RAGAS)
    *   `refusal_rate` (Custom heuristic)
    *   `latency` (Avg/P99)
*   **Output**: Saves distinct CSV file for easy plotting.

---

## 4. Configuration (`config/config.yaml`)

```yaml
system:
  embedding_model: "all-MiniLM-L6-v2"  # Fast local embedding
  llm:
    model_name: "llama3"
    temperature: 0.0
  judge_llm: "llama3"                 # LLM for RAGAS evaluation

data:
  ingestion_size: 500   # Number of QA pairs to ingest
  ingestion_seed: 42
  test_size: 50         # Number of samples to test
  test_seed: 123

retrieval:
  chunk_size: 256
  chunk_overlap: 20
  top_k: 3
```