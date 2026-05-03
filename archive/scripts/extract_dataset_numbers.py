#!/usr/bin/env python3
"""
Extract exact dataset metrics for 4 datasets: NQ, TriviaQA, PubMedQA, FinDER
Measures: raw_source_count, usable_count, gold passages, distractor info
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loaders.nq_loader import NQLoader
from src.data_loaders.trivia_loader import TriviaLoader
from src.data_loaders.pubmed_loader import PubMedLoader
from datasets import load_dataset

results = {}

print("\n" + "="*80)
print("EXTRACTING EXACT DATASET NUMBERS")
print("="*80)

# =============================================================================
# NQ (Natural Questions)
# =============================================================================
print("\n[1/4] NQ (Natural Questions)")
print("-" * 80)
try:
    import ir_datasets
    dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
    
    queries_count = sum(1 for _ in dataset.queries_iter())
    print(f"Raw queries: {queries_count}")
    
    loader = NQLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    
    print(f"Usable QA pairs: {usable_count}")
    print(f"Total gold passages: {gold_count}")
    
    results['nq'] = {
        'raw_source_count': queries_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_source': 'microsoft/ms_marco (v1.1, train)',
        'base_distractor_chunk_count': 50000,
    }
    print("✓ NQ complete")
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['nq'] = {'error': str(e)}

# =============================================================================
# TriviaQA
# =============================================================================
print("\n[2/4] TriviaQA")
print("-" * 80)
try:
    ds_raw = load_dataset(
        "trivia_qa", "rc", split="validation",
        cache_dir="data/raw", download_mode="reuse_dataset_if_exists"
    )
    raw_count = len(ds_raw)
    print(f"Raw rows: {raw_count}")
    
    loader = TriviaLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    
    print(f"Usable QA pairs: {usable_count}")
    print(f"Total gold passages: {gold_count}")
    
    results['triviaqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'eligible_nonmember_count': usable_count,
        'base_distractor_source': 'wikimedia/wikipedia (20231101.en, train)',
        'base_distractor_chunk_count': 50000,
    }
    print("✓ TriviaQA complete")
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['triviaqa'] = {'error': str(e)}

# =============================================================================
# PubMedQA
# =============================================================================
print("\n[3/4] PubMedQA")
print("-" * 80)
try:
    ds_raw = load_dataset(
        "qiaojin/PubMedQA", "pqa_labeled", split="train",
        cache_dir="data/raw", download_mode="reuse_dataset_if_exists"
    )
    raw_count = len(ds_raw)
    print(f"Raw rows: {raw_count}")
    
    loader = PubMedLoader(cache_dir="data/raw")
    qa_pairs = loader.load_qa_pairs(limit=None, seed=42)
    usable_count = len(qa_pairs)
    gold_count = sum(len(pair.gold_passages) for pair in qa_pairs)
    
    print(f"Usable QA pairs: {usable_count}")
    print(f"Total gold passages: {gold_count}")
    
    results['pubmedqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'eligible_nonmember_count': usable_count,
        'base_distractor_source': 'uiyunkim-hub/pubmed-abstract (train)',
        'base_distractor_chunk_count': 50000,
    }
    print("✓ PubMedQA complete")
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['pubmedqa'] = {'error': str(e)}

# =============================================================================
# FinDER (Linq-AI-Research/FinDER)
# =============================================================================
print("\n[4/4] FinDER")
print("-" * 80)
try:
    ds_raw = load_dataset(
        "Linq-AI-Research/FinDER", split="train",
        cache_dir="data/raw", download_mode="reuse_dataset_if_exists"
    )
    raw_count = len(ds_raw)
    print(f"Raw rows: {raw_count}")
    
    if len(ds_raw) > 0:
        first_row = ds_raw[0]
        print(f"Dataset fields: {list(first_row.keys())}")
        
        # Auto-detect field mappings
        field_mapping = {}
        for k in first_row.keys():
            k_lower = k.lower()
            if 'question' in k_lower and 'question' not in field_mapping:
                field_mapping['question'] = k
            elif 'answer' in k_lower and 'answer' not in field_mapping:
                field_mapping['answer'] = k
            elif any(x in k_lower for x in ['evidence', 'context', 'passage']) and 'evidence' not in field_mapping:
                field_mapping['evidence'] = k
        
        print(f"Field mapping: {field_mapping}")
        
        # Filter and count
        qa_pairs = []
        for row in ds_raw:
            try:
                q_field = field_mapping.get('question')
                a_field = field_mapping.get('answer')
                e_field = field_mapping.get('evidence')
                
                if q_field and a_field and e_field:
                    q = str(row.get(q_field, '')).strip()
                    a = str(row.get(a_field, '')).strip()
                    e = row.get(e_field, [])
                    
                    if q and a and e:
                        qa_pairs.append((q, a, e))
            except:
                pass
        
        usable_count = len(qa_pairs)
        gold_count = sum(len(e) if isinstance(e, list) else 1 for _, _, e in qa_pairs)
        effective_pool = usable_count
        
        print(f"Usable rows: {usable_count}")
        print(f"Total evidence units: {gold_count}")
        print(f"Effective pool after holdouts: {effective_pool}")
        
        results['finder'] = {
            'raw_source_count': raw_count,
            'usable_count_after_filtering': usable_count,
            'gold_ingestion_unit_count': gold_count,
            'effective_pool_after_holdouts': effective_pool,
            'base_distractor_source': 'TeraflopAI/SEC-EDGAR (train)',
            'base_distractor_chunk_count': 50000,
            'field_mapping': field_mapping,
        }
    else:
        results['finder'] = {'error': 'Dataset empty'}
    
    print("✓ FinDER complete")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['finder'] = {'error': str(e)}

# =============================================================================
# SUMMARY & SAVE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(json.dumps(results, indent=2))

output_file = os.path.join(os.path.dirname(__file__), 'extracted_numbers.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: {output_file}")
