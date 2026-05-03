#!/usr/bin/env python3
"""
Extract core dataset numbers + distractor doc counts.
Skips Wikipedia distractor (6M rows) - user will provide that separately.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loaders.nq_loader import NQLoader
from src.data_loaders.trivia_loader import TriviaLoader
from src.data_loaders.pubmed_loader import PubMedLoader
from datasets import load_dataset
import json

results = {}

# ===== NQ =====
print("\n=== NQ (Natural Questions) ===")
try:
    import ir_datasets
    dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
    
    # Count raw queries
    queries_count = 0
    for _ in dataset.queries_iter():
        queries_count += 1
    print(f"✓ Raw queries loaded: {queries_count}")
    
    # Use loader to get filtered count
    loader = NQLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    print(f"✓ Usable QA pairs after filtering: {usable_count}")
    
    # Count gold passages
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    print(f"✓ Gold passages total: {gold_count}")
    
    results['nq'] = {
        'raw_source_count': queries_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_chunk_count': 50000
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['nq'] = {'error': str(e)}

# ===== TriviaQA =====
print("\n=== TriviaQA ===")
try:
    # Count raw
    ds_raw = load_dataset("trivia_qa", "rc", split="validation", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds_raw)
    print(f"✓ Raw rows: {raw_count}")
    
    # Use loader for filtered
    loader = TriviaLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    print(f"✓ Usable QA pairs after filtering: {usable_count}")
    
    # Count gold passages
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    print(f"✓ Gold passages total: {gold_count}")
    
    results['triviaqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_doc_count': 'SKIPPED (Wikipedia 6M rows)',
        'base_distractor_chunk_count': 50000,
        'eligible_nonmember_count': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['triviaqa'] = {'error': str(e)}

# ===== PubMedQA =====
print("\n=== PubMedQA ===")
try:
    # Count raw
    ds_raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds_raw)
    print(f"✓ Raw rows: {raw_count}")
    
    # Use loader for filtered
    loader = PubMedLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    print(f"✓ Usable QA pairs after filtering: {usable_count}")
    
    # Count gold passages
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    print(f"✓ Gold passages total: {gold_count}")
    
    results['pubmedqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_chunk_count': 50000,
        'eligible_nonmember_count': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['pubmedqa'] = {'error': str(e)}

# ===== FinDER =====
print("\n=== FinDER (Linq-AI-Research/FinDER) ===")
try:
    # Count raw
    ds_raw = load_dataset("Linq-AI-Research/FinDER", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds_raw)
    print(f"✓ Raw rows: {raw_count}")
    
    # Inspect first few rows to find field mapping
    print("\nInspecting rows to determine field mappings...")
    if len(ds_raw) > 0:
        first_row = ds_raw[0]
        print(f"Fields in first row: {list(first_row.keys())}")
        
        # Auto-detect field mappings
        field_mapping = {}
        for key in first_row.keys():
            val = first_row[key]
            key_lower = key.lower()
            if 'question' in key_lower and 'question' not in field_mapping:
                field_mapping['question'] = key
                print(f"  → question: {key}")
            elif 'answer' in key_lower and 'answer' not in field_mapping:
                field_mapping['answer'] = key
                print(f"  → answer: {key}")
            elif 'evidence' in key_lower and 'evidence' not in field_mapping:
                field_mapping['evidence'] = key
                print(f"  → evidence: {key}")
            elif 'context' in key_lower and 'evidence' not in field_mapping:
                field_mapping['evidence'] = key
                print(f"  → evidence (as context): {key}")
    
    # Count usable after filtering
    usable_count = 0
    gold_count = 0
    
    q_field = field_mapping.get('question')
    a_field = field_mapping.get('answer')
    e_field = field_mapping.get('evidence')
    
    if q_field and a_field and e_field:
        for row in ds_raw:
            q = str(row.get(q_field, '')).strip()
            a = str(row.get(a_field, '')).strip()
            e = row.get(e_field)
            
            if q and a and e:
                usable_count += 1
                if isinstance(e, list):
                    gold_count += len(e)
                else:
                    gold_count += 1
    
    print(f"✓ Usable rows after filtering: {usable_count}")
    print(f"✓ Gold evidence units total: {gold_count}")
    
    results['finder'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_chunk_count': 50000,
        'field_mapping': field_mapping,
        'effective_pool_after_holdouts': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['finder'] = {'error': str(e)}

# ===== DISTRACTOR DOC COUNTS =====
print("\n" + "="*70)
print("COUNTING DISTRACTOR DOCUMENTS (streaming)")
print("="*70)

# NQ distractor: microsoft/ms_marco
print("\n[1/3] NQ distractor: microsoft/ms_marco v1.1 train")
try:
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists", streaming=False)
    nq_distractor_doc_count = len(ds)
    print(f"✓ Document count: {nq_distractor_doc_count}")
    results['nq']['base_distractor_doc_count'] = nq_distractor_doc_count
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['nq']['base_distractor_doc_count'] = f"ERROR: {e}"

# PubMedQA distractor: uiyunkim-hub/pubmed-abstract
print("\n[2/3] PubMedQA distractor: uiyunkim-hub/pubmed-abstract train")
try:
    ds = load_dataset("uiyunkim-hub/pubmed-abstract", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists", streaming=False)
    pubmed_distractor_doc_count = len(ds)
    print(f"✓ Document count: {pubmed_distractor_doc_count}")
    results['pubmedqa']['base_distractor_doc_count'] = pubmed_distractor_doc_count
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['pubmedqa']['base_distractor_doc_count'] = f"ERROR: {e}"

# FinDER distractor: TeraflopAI/SEC-EDGAR
print("\n[3/3] FinDER distractor: TeraflopAI/SEC-EDGAR train")
try:
    ds = load_dataset("TeraflopAI/SEC-EDGAR", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists", streaming=False)
    finder_distractor_doc_count = len(ds)
    print(f"✓ Document count: {finder_distractor_doc_count}")
    results['finder']['base_distractor_doc_count'] = finder_distractor_doc_count
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['finder']['base_distractor_doc_count'] = f"ERROR: {e}"

# Print final summary
print("\n\n" + "="*70)
print("FINAL SUMMARY - ALL NUMBERS")
print("="*70)
print(json.dumps(results, indent=2))

# Save to file
output_file = "scripts/extracted_numbers.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: {output_file}")
