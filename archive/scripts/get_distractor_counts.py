#!/usr/bin/env python3
"""
Get distractor dataset document counts.
Since datasets may be very large, we'll load with limited rows where possible.
"""
from datasets import load_dataset

distractor_configs = {
    'triviaqa': {
        'hf_path': 'wikimedia/wikipedia',
        'subset': '20231101.en',
        'split': 'train',
    },
    'pubmedqa': {
        'hf_path': 'uiyunkim-hub/pubmed-abstract',
        'subset': None,
        'split': 'train',
    },
    'finder': {
        'hf_path': 'TeraflopAI/SEC-EDGAR',
        'subset': None,
        'split': 'train',
    }
}

print("="*80)
print("DISTRACTOR DOCUMENT COUNTS")
print("="*80)

for dataset_name, config in distractor_configs.items():
    try:
        print(f"\n{dataset_name.upper()}:")
        print(f"  Source: {config['hf_path']}")
        
        # Load dataset to count
        if config['subset']:
            ds = load_dataset(
                config['hf_path'],
                config['subset'],
                split=config['split'],
                cache_dir="data/raw",
                download_mode="reuse_dataset_if_exists"
            )
        else:
            ds = load_dataset(
                config['hf_path'],
                split=config['split'],
                cache_dir="data/raw",
                download_mode="reuse_dataset_if_exists"
            )
        
        doc_count = len(ds)
        print(f"  Document count: {doc_count:,}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
