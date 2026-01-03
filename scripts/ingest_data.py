"""
Ingest data into ChromaDB using smart indexing (gold passages only).
Usage: python scripts/ingest_data.py [dataset] [--limit N]
"""
import argparse
import os
import sys
import json
import yaml
import logging
import shutil

# Suppress noisy httpx logs (only show errors)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders.nq_loader import NQLoader
from src.data_loaders.pubmed_loader import PubMedLoader
from src.data_loaders.trivia_loader import TriviaLoader
from src.core.retrieval import VectorStore


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def get_loader(name: str):
    loaders = {
        "nq": NQLoader,
        "pubmedqa": PubMedLoader,
        "triviaqa": TriviaLoader,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name]()


def clear_all_data(config: dict):
    """Clear all previous ingestion data for reproducibility."""
    chroma_path = config["paths"]["chroma_db"]
    qa_path = "data/qa_pairs"
    
    print("Clearing previous ingestion data...")
    
    # Clear ChromaDB
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"  ✓ Removed {chroma_path}")
    
    # Clear QA pairs
    if os.path.exists(qa_path):
        shutil.rmtree(qa_path)
        print(f"  ✓ Removed {qa_path}")
    
    # Recreate directories
    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs(qa_path, exist_ok=True)
    print("  ✓ Fresh directories created")


def ingest_dataset(dataset_name: str, limit: int, seed: int, config: dict):
    """Ingest a single dataset with random sampling."""
    import random
    
    print(f"\n{'='*50}")
    print(f"INGESTING: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    # Load ALL QA pairs first, then sample
    loader = get_loader(dataset_name)
    all_qa_pairs = loader.load_qa_pairs(limit=None)  # Load all
    print(f"Total available: {len(all_qa_pairs)} QA pairs")
    
    # Random sample
    random.seed(seed)
    if len(all_qa_pairs) > limit:
        qa_pairs = random.sample(all_qa_pairs, limit)
    else:
        qa_pairs = all_qa_pairs
    print(f"Randomly selected {len(qa_pairs)} QA pairs (seed={seed})")
    
    if not qa_pairs:
        print("No QA pairs to ingest!")
        return
    
    # Initialize vector store (fresh - already cleared in main())
    vs = VectorStore(
        collection_name=loader.name,
        persist_directory=config["paths"]["chroma_db"],
        embedding_model=config["system"]["embedding_model"]
    )
    
    # Chunk and index gold passages
    chunk_size = config["retrieval"]["chunk_size"]
    chunk_overlap = config["retrieval"]["chunk_overlap"]
    
    all_chunks = []
    all_metas = []
    all_ids = []
    
    for qa in qa_pairs:
        for p_idx, passage in enumerate(qa.gold_passages):
            # Simple chunking
            chunks = chunk_text(passage, chunk_size, chunk_overlap)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metas.append({
                    "pair_id": qa.pair_id,
                    "question": qa.question[:200],  # Store truncated question for reference
                    "passage_idx": p_idx,
                    "chunk_idx": c_idx
                })
                all_ids.append(f"{qa.pair_id}_p{p_idx}_c{c_idx}")
    
    print(f"Created {len(all_chunks)} chunks from gold passages")
    
    # Add to vector store
    vs.add_documents(all_chunks, all_metas, all_ids)
    
    # Save QA pairs for later retrieval testing
    qa_data = [
        {
            "pair_id": qa.pair_id,
            "question": qa.question,
            "answer": qa.answer,
            "metadata": qa.metadata
        }
        for qa in qa_pairs
    ]
    
    os.makedirs("data/qa_pairs", exist_ok=True)
    with open(f"data/qa_pairs/{dataset_name}.json", "w") as f:
        json.dump(qa_data, f, indent=2)
    
    print(f"✓ Ingested {dataset_name}: {len(all_chunks)} chunks from {len(qa_pairs)} QA pairs")


def chunk_text(text: str, size: int, overlap: int) -> list:
    """Simple text chunking."""
    if len(text) <= size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest datasets into ChromaDB")
    parser.add_argument("dataset", nargs="?", choices=["nq", "pubmedqa", "triviaqa"],
                        help="Dataset to ingest (default: all)")
    parser.add_argument("--limit", type=int, help="Override config ingestion_size")
    args = parser.parse_args()
    
    config = load_config()
    limit = args.limit or config["data"]["ingestion_size"]
    seed = config["data"]["ingestion_seed"]
    
    datasets = [args.dataset] if args.dataset else ["nq", "pubmedqa", "triviaqa"]
    
    print("=" * 60)
    print("RAG Defence Utility - Smart Ingestion")
    print("=" * 60)
    print(f"Ingestion limit: {limit} QA pairs per dataset")
    print(f"Random seed: {seed}")
    
    # Clear previous data for reproducibility
    clear_all_data(config)
    
    for ds in datasets:
        try:
            ingest_dataset(ds, limit, seed, config)
        except Exception as e:
            print(f"✗ Failed to ingest {ds}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Ingestion complete!")


if __name__ == "__main__":
    main()
