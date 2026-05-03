#!/usr/bin/env python3
"""
Extract dataset numbers WITHOUT triggering large downloads.
Directly loads datasets and counts rows.
"""
from datasets import load_dataset
import json

results = {}

print("\n" + "="*70)
print("EXTRACTING DATASET NUMBERS (NO LARGE DOWNLOADS)")
print("="*70)

# ===== NQ =====
print("\n[1/4] NQ (Natural Questions)")
try:
    # Load NQ dataset directly
    ds = load_dataset("Tevatron/nq", split="test", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw count: {raw_count}")
    
    # Count usable (with both query and positive passages)
    usable = sum(1 for row in ds if row.get('query') and row.get('positive_passages'))
    print(f"✓ Usable count: {usable}")
    
    # Count gold passages
    gold = sum(len(row.get('positive_passages', [])) for row in ds)
    print(f"✓ Gold passages: {gold}")
    
    results['nq'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_chunk_count': 50000
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['nq'] = {'error': str(e)}

# ===== TriviaQA =====
print("\n[2/4] TriviaQA")
try:
    ds = load_dataset("trivia_qa", "rc", split="validation", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw count: {raw_count}")
    
    # Count usable (with question and entity_pages.wiki_context)
    usable = sum(1 for row in ds 
                 if row.get('question') and 
                 row.get('entity_pages') and 
                 row['entity_pages'].get('wiki_context') and
                 any(ctx.strip() for ctx in row['entity_pages']['wiki_context']))
    print(f"✓ Usable count: {usable}")
    
    # Count gold passages
    gold = 0
    for row in ds:
        if row.get('entity_pages') and row['entity_pages'].get('wiki_context'):
            gold += len([ctx for ctx in row['entity_pages']['wiki_context'] if ctx.strip()])
    print(f"✓ Gold passages: {gold}")
    
    results['triviaqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_doc_count': 'SKIPPED (6M rows)',
        'base_distractor_chunk_count': 50000,
        'eligible_nonmember_count': 'TBD'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    results['triviaqa'] = {'error': str(e)}

# ===== PubMedQA =====
print("\n[3/4] PubMedQA")
try:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw count: {raw_count}")
    
    # Count usable (with question and context)
    usable = sum(1 for row in ds if row.get('question') and row.get('context'))
    print(f"✓ Usable count: {usable}")
    
    # Count gold passages
    gold = 0
    for row in ds:
        context = row.get('context')
        if context:
            if isinstance(context, dict) and context.get('contexts'):
                gold += len(context['contexts'])
            else:
                gold += 1
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

# ===== FinDER =====
print("\n[4/4] FinDER (Linq-AI-Research/FinDER)")
try:
    ds = load_dataset("Linq-AI-Research/FinDER", split="train", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw count: {raw_count}")
    
    # Inspect first row
    if len(ds) > 0:
        first_row = ds[0]
        print(f"  Fields: {list(first_row.keys())}")
        
        # Auto-detect field mappings
        field_mapping = {}
        for key in first_row.keys():
            key_lower = key.lower()
            if 'question' in key_lower and 'question' not in field_mapping:
                field_mapping['question'] = key
            elif 'answer' in key_lower and 'answer' not in field_mapping:
                field_mapping['answer'] = key
            elif 'evidence' in key_lower and 'evidence' not in field_mapping:
                field_mapping['evidence'] = key
            elif 'context' in key_lower and 'evidence' not in field_mapping:
                field_mapping['evidence'] = key
        
        print(f"  Detected fields → question={field_mapping.get('question')}, answer={field_mapping.get('answer')}, evidence={field_mapping.get('evidence')}")
    
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

# Print final summary
print("\n\n" + "="*70)
print("FINAL NUMBERS")
print("="*70)
print(json.dumps(results, indent=2))

# Save to file
with open("scripts/extracted_numbers.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: scripts/extracted_numbers.json")
