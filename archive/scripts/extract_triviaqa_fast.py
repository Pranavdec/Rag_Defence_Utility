#!/usr/bin/env python3
"""
Fast TriviaQA extraction - optimized for speed.
"""
from datasets import load_dataset
import json

print("\n" + "="*70)
print("FAST TRIVIAQA EXTRACTION")
print("="*70)

results = {}

try:
    # Load dataset (validation split - should be cached)
    print("\n[1/3] Loading TriviaQA validation split...")
    ds = load_dataset("trivia_qa", "rc", split="validation", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw rows: {raw_count}")
    
    # Count usable and gold in one pass
    print("\n[2/3] Filtering and counting gold passages...")
    usable = 0
    gold = 0
    
    for i, row in enumerate(ds):
        if i % 5000 == 0:
            print(f"  Progress: {i}/{raw_count}...")
        
        question = row.get('question', '').strip()
        entity_pages = row.get('entity_pages', {})
        
        if not isinstance(entity_pages, dict):
            continue
            
        wiki_contexts = entity_pages.get('wiki_context', [])
        if not isinstance(wiki_contexts, list):
            continue
        
        # Filter empty contexts
        non_empty = [ctx for ctx in wiki_contexts if ctx and str(ctx).strip()]
        
        if question and non_empty:
            usable += 1
            gold += len(non_empty)
    
    print(f"✓ Usable rows: {usable}")
    print(f"✓ Gold passages: {gold}")
    
    results['triviaqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_chunk_count': 50000,
        'filtering_rate': f"{(usable/raw_count*100):.2f}%"
    }
    
    print("\n[3/3] Complete!")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['triviaqa'] = {'error': str(e)}

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(json.dumps(results, indent=2))

# Save
with open("scripts/triviaqa_numbers.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: scripts/triviaqa_numbers.json")
