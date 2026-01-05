"""
Download only the required datasets for smart indexing.
- NQ: ir_datasets (beir/nq) - has qrels + docs + queries
- PubMedQA: pqa_labeled only (question + context in same row)
- TriviaQA: rc validation only (question + entity_pages in same row)
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# CRITICAL: Set IR_DATASETS_HOME before importing ir_datasets
# ir_datasets reads this environment variable once on import and caches it
os.environ['IR_DATASETS_HOME'] = os.path.abspath("data/raw/ir_datasets")
os.makedirs(os.environ['IR_DATASETS_HOME'], exist_ok=True)

def download_nq():
    """Download NQ via ir_datasets."""
    print("Downloading NQ (dpr-w100/natural-questions/dev) via ir_datasets...")
    import ir_datasets
    
    # This triggers the download
    dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
    
    # Verify we can iterate
    print(f"  - Docs: {dataset.docs_count()}")
    print(f"  - Queries: {dataset.queries_count()}")
    print(f"  - Qrels available: {dataset.has_qrels()}")
    print("✓ NQ downloaded")

def download_pubmedqa():
    """Download PubMedQA pqa_labeled."""
    print("Downloading PubMedQA (pqa_labeled)...")
    from datasets import load_dataset
    
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", 
                      cache_dir="data/raw")
    print(f"  - Rows: {len(ds)}")
    print("✓ PubMedQA downloaded")

def download_triviaqa():
    """Download TriviaQA rc validation."""
    print("Downloading TriviaQA (rc validation)...")
    from datasets import load_dataset
    
    ds = load_dataset("trivia_qa", "rc", split="validation",
                      cache_dir="data/raw")
    print(f"  - Rows: {len(ds)}")
    print("✓ TriviaQA downloaded")

def main():
    print("=" * 60)
    print("RAG Defence Utility - Dataset Downloader (Smart Indexing)")
    print("=" * 60)
    print()
    
    os.makedirs("data/raw", exist_ok=True)
    
    results = {}
    
    # NQ
    try:
        download_nq()
        results["nq"] = True
    except Exception as e:
        print(f"✗ NQ failed: {e}")
        results["nq"] = False
    
    print()
    
    # PubMedQA
    try:
        download_pubmedqa()
        results["pubmedqa"] = True
    except Exception as e:
        print(f"✗ PubMedQA failed: {e}")
        results["pubmedqa"] = False
    
    print()
    
    # TriviaQA
    try:
        download_triviaqa()
        results["triviaqa"] = True
    except Exception as e:
        print(f"✗ TriviaQA failed: {e}")
        results["triviaqa"] = False
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

if __name__ == "__main__":
    main()
