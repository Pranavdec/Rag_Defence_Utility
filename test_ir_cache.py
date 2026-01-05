#!/usr/bin/env python3
"""
Test script to verify ir_datasets caching is working correctly.
This will load the NQ dataset multiple times and check if it's using the cache.
"""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loaders.nq_loader import NQLoader

def test_caching():
    print("=" * 60)
    print("Testing IR Datasets Caching")
    print("=" * 60)
    print()
    
    # Check if cache directory exists and has data
    cache_path = os.path.abspath("data/raw/ir_datasets")
    print(f"Cache directory: {cache_path}")
    print(f"Cache exists: {os.path.exists(cache_path)}")
    
    if os.path.exists(cache_path):
        # Count files in cache
        total_files = sum([len(files) for _, _, files in os.walk(cache_path)])
        print(f"Files in cache: {total_files}")
    print()
    
    # Test 1: First load
    print("Test 1: Loading NQ dataset (limit=2)...")
    start = time.time()
    loader1 = NQLoader()
    qa_pairs1 = loader1.load_qa_pairs(limit=2)
    elapsed1 = time.time() - start
    print(f"  ✓ Loaded {len(qa_pairs1)} QA pairs in {elapsed1:.2f}s")
    print()
    
    # Test 2: Second load (should use cache)
    print("Test 2: Loading NQ dataset again (limit=2)...")
    start = time.time()
    loader2 = NQLoader()
    qa_pairs2 = loader2.load_qa_pairs(limit=2)
    elapsed2 = time.time() - start
    print(f"  ✓ Loaded {len(qa_pairs2)} QA pairs in {elapsed2:.2f}s")
    print()
    
    # Compare times
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"First load:  {elapsed1:.2f}s")
    print(f"Second load: {elapsed2:.2f}s")
    
    if elapsed2 < elapsed1:
        speedup = elapsed1 / elapsed2
        print(f"✓ Second load was {speedup:.1f}x faster (cache working!)")
    else:
        print("⚠ Second load was not faster (cache may not be working)")
    
    # Verify cache directory was created correctly
    expected_cache = os.environ.get('IR_DATASETS_HOME')
    print()
    print(f"IR_DATASETS_HOME env var: {expected_cache}")
    print(f"Cache path matches: {expected_cache == cache_path}")
    
    return qa_pairs1, qa_pairs2

if __name__ == "__main__":
    test_caching()
