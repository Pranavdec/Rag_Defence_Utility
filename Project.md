# Project Specification: RAG Defense & Attack Evaluation Framework

**Project Name:** Quantifying Alignment Tax in Defensive RAG  
**Current Phase:** Full Implementation (Defenses, Attacks, ADO, Comprehensive Evaluation)  
**Goal:** A modular RAG pipeline for evaluating defense mechanisms against adversarial attacks (PoisonedRAG, CorruptRAG, MBA) while measuring utility trade-offs across diverse datasets.

---

## 1. System Architecture

The system follows a modular "Pipeline" architecture with **Attacks**, **Defenses**, and **ADO (Adaptive Defense Orchestration)** layers.

### High-Level Data Flow

1. **Ingestion:** Raw Dataset → `Data Loader` → Sampled QA Pairs → Index Gold Passages → ChromaDB
2. **Attack Generation:** Target Questions → `Attack Module` → Poisoned/Adversarial Documents → Injected into VectorDB
3. **Defense Layer:** Query → `Defense Manager` (DP-RAG, TrustRAG, AV) → Filtered Retrieval
4. **ADO Layer:** Query → `Sentinel` (Threat Analysis) → `Strategist` (Defense Plan) → Dynamic Defense Activation
5. **Inference:** Filtered Context → `LLM Generation` → Answer
6. **Evaluation:** Results → `DeepEval` + Custom Metrics → JSON/CSV Reports

---

## 2. Directory Structure

```
rag-defense-utility/
├── config/
│   └── config.yaml              # Global settings (defenses, attacks, ADO)
│
├── data/
│   ├── raw/                     # Cached HuggingFace datasets
│   ├── qa_pairs/                # Saved QA pairs for testing
│   ├── chroma_db/               # Persisted VectorDB
│   ├── results/                 # JSON outputs
│   ├── metrics/                 # Evaluation reports (JSON + CSV)
│   └── users/                   # ADO user trust persistence
│
├── scripts/
│   ├── comprehensive_eval.py    # Unified evaluation script
│   ├── run_mba_experiments.py   # MBA attack runner
│   ├── run_poisoning_attack.py  # Poisoning attack runner
│   ├── run_ado_testing.py       # ADO system testing
│   ├── ingest_data.py           # Data ingestion
│   └── download_datasets.py     # Dataset download
│
├── src/
│   ├── core/
│   │   ├── pipeline.py          # Main RAG orchestration
│   │   ├── retrieval.py         # ChromaDB wrapper
│   │   └── generation.py        # HuggingFace/Ollama LLM wrapper
│   │
│   ├── data_loaders/
│   │   ├── base_loader.py
│   │   ├── nq_loader.py
│   │   ├── pubmed_loader.py
│   │   └── trivia_loader.py
│   │
│   ├── attacks/
│   │   ├── base_attack.py       # Abstract attack interface
│   │   ├── poisonedrag_attack.py # PoisonedRAG implementation
│   │   ├── corruptrag_attack.py  # CorruptRAG implementation
│   │   ├── mba.py               # Membership Inference Attack
│   │   └── utils.py             # Attack utilities
│   │
│   ├── defenses/
│   │   ├── base.py              # Abstract defense interface
│   │   ├── manager.py           # Defense orchestration
│   │   ├── dp_rag.py            # Differential Privacy RAG
│   │   ├── trustrag.py          # TrustRAG (similarity + rouge filtering)
│   │   └── av_defense.py        # Attention-based Verification
│   │
│   └── evaluation/
│       └── evaluator.py         # DeepEval + custom metrics
│
├── tests/
│   ├── test_mba_attack.py
│   └── test_av_defense_mock.py
│
└── main.py                      # CLI runner
```

---

## 3. Implemented Components

### A. Data Loaders (Smart Indexing)

**Datasets Supported:**
- **Natural Questions (NQ):** General knowledge QA
- **PubMedQA:** Medical domain QA
- **TriviaQA:** Complex reasoning QA

**Smart Indexing Strategy:**
1. Sample `ingestion_size` QA pairs using `ingestion_seed`
2. Index only gold passages associated with sampled questions
3. Save QA pairs to `data/qa_pairs/` for reproducible testing

### B. Defense Mechanisms (`src/defenses/`)

| Defense | Description | Key Parameters |
|---------|-------------|----------------|
| **DP-RAG** | Differential Privacy via approximate DP mechanism | `epsilon=3.0`, `delta=0.01` |
| **TrustRAG** | Filters documents by similarity + ROUGE thresholds | `similarity_threshold=0.88`, `rouge_threshold=0.25` |
| **AV Defense** | Attention-based verification using Llama 3.1 | `top_tokens=100`, `max_corruptions=3` |

**Defense Combinations Available:**
- `none`, `dp`, `trustrag`, `av`
- `dp_trustrag`, `dp_av`, `trustrag_av`, `all_static`
- `ado_only`, `ado_dp`, `ado_trustrag`, `ado_all`

### C. Attack Mechanisms (`src/attacks/`)

| Attack | Type | Description |
|--------|------|-------------|
| **PoisonedRAG** | Corpus Poisoning | Injects multiple adversarial docs per target question |
| **CorruptRAG** | Corpus Poisoning | Single-doc injection with prompt injection templates |
| **MBA** | Membership Inference | Mask-based attack to infer document membership |

**MBA Parameters:**
- `M`: Number of masks per document (default: 7)
- `gamma`: Membership threshold (default: 0.5)
- `proxy_model`: GPT-2 XL for word difficulty scoring

### D. ADO (Adaptive Defense Orchestration)

**Components:**
- **UserTrustManager:** Persistent trust scores per user
- **MetricsCollector:** Pre/post retrieval behavioral metrics
- **Sentinel:** LLM-based threat analyzer (Ollama llama3)
- **Strategist:** LLM-based defense planner

**Flow:** Query → Sentinel (Risk Analysis) → Strategist (Defense Plan) → Dynamic Defense Activation

### E. Evaluation (`src/evaluation/`)

**DeepEval Metrics:**
- `answer_relevancy`: Answer addresses the question
- `faithfulness`: Answer derived from context
- `contextual_relevancy`: Retrieved context relevant to query
- `contextual_recall`: Relevant context was retrieved

**Attack Metrics:**
- `ASR (Attack Success Rate)`: % of attacks that succeeded
- `Poisoning ASR`: % of poisoned queries returning attacker's answer
- `MBA ASR`: % of membership inferences correct

**Performance Metrics:**
- `latency_ms`: End-to-end query latency
- `refusal_rate`: Heuristic refusal detection

---

## 4. Configuration (`config/config.yaml`)

```yaml
system:
  embedding_model: all-MiniLM-L6-v2
  llm:
    provider: huggingface
    model_path: meta-llama/Llama-3.1-8B-Instruct
    device: auto
    temperature: 0.0
  judge_llm: llama3

data:
  ingestion_size: 700
  ingestion_seed: 42
  test_size: 1
  test_seed: 123

retrieval:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50

defenses:
  - name: differential_privacy
    enabled: false
    epsilon: 3.0
    delta: 0.01
  - name: trustrag
    enabled: false
    similarity_threshold: 0.88
    rouge_threshold: 0.25
  - name: attention_filtering
    enabled: false
    model_path: meta-llama/Llama-3.1-8B-Instruct

attack:
  mba:
    M: 7
    gamma: 0.5
    num_members: 50
    num_non_members: 50
    proxy_model: gpt2-xl

ado:
  enabled: true
  sentinel_model: llama3
  strategist_model: llama3
  trust_score_decay: 0.05
```

---

## 5. Usage

### Comprehensive Evaluation (Recommended)

```bash
# Full evaluation: all datasets, all defenses, mixed attacks
./run_comprehensive_eval.sh mixed --datasets nq pubmedqa triviaqa \
    --defenses none dp trustrag ado_only \
    --num-benign 20 --num-poison 10 --num-mba 10 --deepeval

# Quick test on single dataset
./run_comprehensive_eval.sh mixed --datasets pubmedqa \
    --defenses ado_only --num-benign 5 --num-mba 5 --num-poison 5

# Utility-only (no attacks)
./run_comprehensive_eval.sh utility --datasets nq --defenses dp trustrag
```

### Individual Attack Testing

```bash
# MBA Attack
./run_mba_experiments.sh nq --M 10 --gamma 0.5

# Poisoning Attack
python scripts/run_poisoning_attack.py --dataset nq --defense trustrag
```

### Data Ingestion

```bash
# Ingest all datasets
python scripts/ingest_data.py

# Specific dataset
python scripts/ingest_data.py pubmedqa
```

---

## 6. Output Files

| Path | Description |
|------|-------------|
| `data/metrics/eval_*_metrics.csv` | Per-query evaluation metrics |
| `data/metrics/eval_*.json` | Full evaluation report with aggregates |
| `data/results/run_*.json` | Raw inference results |
| `data/users/*.json` | ADO user trust history |

---

## 7. Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for AV defense and MBA)
- Ollama running locally (`ollama serve && ollama pull llama3`)
- HuggingFace authentication for Llama models (`huggingface-cli login`)