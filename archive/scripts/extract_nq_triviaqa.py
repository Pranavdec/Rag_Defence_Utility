#!/usr/bin/env python3
"""
Extract NQ and TriviaQA numbers from cached datasets.
"""
from datasets import load_dataset
import json

results = {}

print("\n" + "="*70)
print("EXTRACTING NQ & TriviaQA FROM CACHE")
print("="*70)

# ===== NQ =====
print("\n[1/2] NQ (Natural Questions via ir_datasets)")
try:
    import ir_datasets
    dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
    
    # Count raw queries
    raw_count = 0
    for _ in dataset.queries_iter():
        raw_count += 1
    print(f"✓ Raw queries: {raw_count}")
    
    # Count qrels (query-doc mappings)
    qrels_count = 0
    for _ in dataset.qrels_iter():
        qrels_count += 1
    print(f"✓ Query-doc relevance pairs: {qrels_count}")
    
    # Estimate usable QA pairs (queries with ≥1 relevant doc)
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = 1
    usable_count = len(qrels)
    print(f"✓ Usable QA pairs (queries with gold docs): {usable_count}")
    
    # Gold passages = number of relevant docs
    gold_count = sum(1 for qrel in dataset.qrels_iter() if qrel.relevance > 0)
    print(f"✓ Gold passages (relevant docs): {gold_count}")
    
    results['nq'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable_count,
        'gold_ingestion_unit_count': gold_count,
        'base_distractor_chunk_count': 50000,
        'note': 'Distractor: ms_marco v1.1'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['nq'] = {'error': str(e)}

# ===== TriviaQA =====
print("\n[2/2] TriviaQA")
try:
    ds = load_dataset("trivia_qa", "rc", split="validation", cache_dir="data/raw", download_mode="reuse_dataset_if_exists")
    raw_count = len(ds)
    print(f"✓ Raw rows: {raw_count}")
    
    # Count usable (with question and entity_pages.wiki_context non-empty)
    usable = 0
    gold = 0
    for row in ds:
        question = row.get('question', '').strip()
        entity_pages = row.get('entity_pages', {})
        wiki_contexts = entity_pages.get('wiki_context', []) if isinstance(entity_pages, dict) else []
        non_empty_contexts = [ctx for ctx in wiki_contexts if ctx and ctx.strip()]
        
        if question and non_empty_contexts:
            usable += 1
            gold += len(non_empty_contexts)
    
    print(f"✓ Usable rows (with valid wiki_context): {usable}")
    print(f"✓ Gold passages (wiki_context entries): {gold}")
    
    results['triviaqa'] = {
        'raw_source_count': raw_count,
        'usable_count_after_filtering': usable,
        'gold_ingestion_unit_count': gold,
        'base_distractor_chunk_count': 50000,
        'note': 'Distractor: Wikipedia (6M rows - user to provide doc count)'
    }
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['triviaqa'] = {'error': str(e)}

# Print summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(json.dumps(results, indent=2))

# Save to file
with open("scripts/extracted_nq_triviaqa.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to: scripts/extracted_nq_triviaqa.json")
