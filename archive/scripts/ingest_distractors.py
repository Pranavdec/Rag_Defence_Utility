#!/usr/bin/env python3
"""
One-off script to download HuggingFace datasets and embed exactly 50k distractor
chunks per dataset into separate ChromaDB base directories.

This allows for offline execution and fast vector DB cloning during evaluation.
"""
import os
import shutil
import logging
from datasets import load_dataset
from typing import List, Dict

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, PROJECT_ROOT)

from src.core.retrieval import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
BASE_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db_base")
TARGET_CHUNKS = 50000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Dataset configs mapping to the target collections
DATASETS_CONFIG = {
    "nq": {
        "hf_path": "microsoft/ms_marco",
        "subset": "v1.1",
        "split": "train",
        "text_field": ["passages", "passage_text"], # Nested in 'passages' dict
        "collection_name": "nq-corpus",
        "out_dir": "nq"
    },
    "triviaqa": {
        "hf_path": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_field": "text",
        "collection_name": "trivia-corpus",
        "out_dir": "triviaqa"
    },
    "pubmedqa": {
        "hf_path": "uiyunkim-hub/pubmed-abstract",
        "subset": None,
        "split": "train",
        "text_field": "abstract",
        "collection_name": "pubmed-corpus",
        "out_dir": "pubmedqa"
    },
    "financebench": {
        "hf_path": "TeraflopAI/SEC-EDGAR",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "collection_name": "financebench-corpus",
        "out_dir": "financebench"
    }
}

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple text chunking by whitespace with overlap approximation."""
    words = text.split()
    if not words:
        return []
    
    # Approximate words per token (1.3 tokens per word)
    words_per_chunk = int(chunk_size / 1.3)
    words_overlap = int(overlap / 1.3)
    step = max(1, words_per_chunk - words_overlap)
    
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + words_per_chunk]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_text(item: dict, field_path) -> List[str]:
    if isinstance(field_path, list):
        if len(field_path) == 2 and field_path[0] == "passages":
            return item[field_path[0]][field_path[1]]
        # Extend as needed
    return [item[field_path]]

def process_dataset(name: str, config: dict):
    logger.info(f"Processing dataset '{name}'...")
    
    db_path = os.path.join(BASE_DB_DIR, config["out_dir"])
    
    # Check if a fully populated DB already exists, if so skip, otherwise clear it to start fresh
    if os.path.exists(db_path):
        try:
            # Let's check if the count is correct
            store = VectorStore(
                collection_name=config["collection_name"],
                persist_directory=db_path,
                embedding_model=EMBEDDING_MODEL
            )
            count = store.collection.count()
            if count >= TARGET_CHUNKS:
                logger.info(f"Base DB for {name} already correctly populated at {db_path} with {count} docs. Skipping.")
                return
            else:
                logger.warning(f"Base DB for {name} exists but has {count} docs (expected {TARGET_CHUNKS}). Re-creating...")
        except Exception as e:
            logger.warning(f"Could not verify existing DB for {name}: {e}. Re-creating...")
        
        shutil.rmtree(db_path)
    
    # Load dataset with streaming to avoid massive downloads
    logger.info(f"Loading '{config['hf_path']}' (streaming mode)...")
    if config["subset"]:
        dataset = load_dataset(config["hf_path"], config["subset"], split=config["split"], streaming=True)
    else:
        dataset = load_dataset(config["hf_path"], split=config["split"], streaming=True)
    
    logger.info(f"Dataset streaming started. Targeting {TARGET_CHUNKS} chunks...")
    
    vector_store = VectorStore(
        collection_name=config["collection_name"],
        persist_directory=db_path,
        embedding_model=EMBEDDING_MODEL
    )
    
    collected_chunks = []
    collected_ids = []
    collected_metas = []
    
    doc_idx = 0
    chunk_count = 0
    
    for item in dataset:
        texts = extract_text(item, config["text_field"])
        for t in texts:
            if not t or not t.strip():
                continue
                
            chunks = chunk_text(t)
            for c_idx, chunk in enumerate(chunks):
                collected_chunks.append(chunk)
                collected_ids.append(f"distractor_{name}_{chunk_count}")
                collected_metas.append({"source": "distractor", "dataset": name, "doc_idx": doc_idx})
                chunk_count += 1
                
                if chunk_count >= TARGET_CHUNKS:
                    break
            
            if chunk_count >= TARGET_CHUNKS:
                break
        
        doc_idx += 1
        if chunk_count >= TARGET_CHUNKS:
            break
            
    logger.info(f"Extracted {len(collected_chunks)} chunks for '{name}'. Adding to Vector DB...")
    
    vector_store.add_documents(
        documents=collected_chunks,
        metadatas=collected_metas,
        ids=collected_ids,
        batch_size=5000
    )
    
    count = vector_store.collection.count()
    logger.info(f"Finished {name}. DB contains {count} documents at {db_path}.")

def main():
    os.makedirs(BASE_DB_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Process all datasets
    for name, config in DATASETS_CONFIG.items():
        process_dataset(name, config)

if __name__ == "__main__":
    main()
