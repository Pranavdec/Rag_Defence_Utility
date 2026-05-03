#!/usr/bin/env python3
"""
Extract numbers using cached datasets and existing loaders.
Does NOT trigger new downloads.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

results = {}

print("\n" + "="*70)
print("EXTRACTING DATASET NUMBERS FROM CACHE")
print("="*70)

# ===== PubMedQA (Already cached successfully) =====
print("\n[1/4] PubMedQA")
try:
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    
    # Count usable (with question and context)
    usable = sum(1 for row in ds if row.get('question') and row.get('context'))
    
    # Count gold passages
    gold = 0
    for row in ds:
        context = row.get('context')
        if context:
            if isinstance(context, dict) and context.get('contexts'):
                gold += len(context['contexts'])
            else:
                gold += 1
    
    print(f"✓ Raw count: {raw_count}")
    print(f"✓ Usable count: {usable}")
    print(f"✓ Gold passages: {gold}")
    
    results['pubmedqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_chunk_count': 50000,
        'eligible_nonmember_count': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['pubmedqa'] = {'error': str(e)}

# ===== FinDER (Already cached successfully) =====
print("\n[2/4] FinDER (Linq-AI-Research/FinDER)")
try:
    from datasets import load_dataset
    ds = load_dataset("Linq-AI-Research/FinDER", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"  Raw rows: {raw_count}")
    
    # Inspect first row for field mapping
    field_mapping = {}
    if len(ds) > 0:
        first_row = ds[0]
        print(f"  Fields available: {list(first_row.keys())}")
        
        # FinDER likely has: text (question), answer, references (evidence)
        for key in first_row.keys():
            key_lower = key.lower()
            if key == 'text' and 'question' not in field_mapping:
                field_mapping['question'] = 'text'
                print(f"  → question field: text")
            elif 'answer' in key_lower and 'answer' not in field_mapping:
                field_mapping['answer'] = key
                print(f"  → answer field: {key}")
            elif 'references' in key_lower and 'evidence' not in field_mapping:
                field_mapping['evidence'] = key
                print(f"  → evidence field: {key}")
    
    # Count usable
    q_field = field_mapping.get('question')
    a_field = field_mapping.get('answer')
    e_field = field_mapping.get('evidence')
    
    usable = 0
    gold = 0
    if q_field and a_field and e_field:
        for row in ds:
            q = str(row.get(q_field, '')).strip()
            a = str(row.get(a_field, '')).strip()
            e = row.get(e_field)
            if q and a and e:
                usable += 1
                if isinstance(e, list):
                    gold += len(e)
                else:
                    gold += 1
    
    print(f"✓ Usable count: {usable}")
    print(f"✓ Gold evidence units: {gold}")
    
    results['finder'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_chunk_count': 50000,
        'field_mapping': field_mapping,
        'effective_pool_after_holdouts': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['finder'] = {'error': str(e)}

# ===== NQ and TriviaQA (cached but large - try to get counts) =====
print("\n[3/4] NQ - (checking for cached data)")
print("  Note: NQ loader requires ir_datasets which triggers large downloads.")
print("  Skipping for now - disk space full. Please provide manually or free space.")
results['nq'] = {
    'raw_source_count': 'NEED: Total queries from dpr-w100/natural-questions/dev',
    'usable_count_after_filtering': 'NEED: QA pairs after loader filtering',
    'gold_ingestion_unit_count': 'NEED: Total gold passages',
    'base_distractor_doc_count': 'NEED: ms_marco v1.1 train doc count',
    'base_distractor_chunk_count': 50000,
    'note': 'Disk space full - unable to load ir_datasets'
}

print("\n[4/4] TriviaQA - (checking for cached data)")
print("  Note: Disk space full, cannot download. Using expected format.")
print("  Expected source: trivia_qa rc validation split")
results['triviaqa'] = {
    'raw_source_count': 'NEED: Total rows in trivia_qa/rc/validation',
    'usable_count_after_filtering': 'NEED: Rows with entity_pages.wiki_context non-empty',
    'gold_ingestion_unit_count': 'NEED: Total gold wiki context passages',
    'base_distractor_doc_count': 'SKIPPED (Wikipedia 6M rows - user to provide)',
    'base_distractor_chunk_count': 50000,
    'eligible_nonmember_count': 'TBD'
}

# ===== SUMMARY =====
print("\n\n" + "="*70)
print("EXTRACTED NUMBERS (PARTIAL - Disk Space Issue)")
print("="*70)
print(json.dumps(results, indent=2))

# Save to file
with open("scripts/extracted_numbers.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: scripts/extracted_numbers.json")

print("\n" + "="*70)
print("WHAT YOU STILL NEED TO PROVIDE:")
print("="*70)
print("""
NQ:
  - raw_source_count: Total queries from dpr-w100/natural-questions/dev
  - usable_count_after_filtering: QA pairs after loader filtering
  - gold_ingestion_unit_count: Total gold passages
  - base_distractor_doc_count: microsoft/ms_marco v1.1 train doc count

TriviaQA:
  - raw_source_count: Total rows in trivia_qa/rc/validation
  - usable_count_after_filtering: Rows with valid entity_pages.wiki_context
  - gold_ingestion_unit_count: Total wiki context passages
  - base_distractor_doc_count: Wikipedia 2023 train doc count
  
(PubMedQA and FinDER completed successfully)
""")
