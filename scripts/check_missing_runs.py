#!/usr/bin/env python3
"""
Check Missing Experiments
Analyzes completed experiment runs and identifies missing defense configurations.
"""

import json
from pathlib import Path
from collections import defaultdict

# Expected defense configurations
EXPECTED_CONFIGS = [
    {"dp": False, "trustrag": False, "av": False, "name": "no_defense"},
    {"dp": True, "trustrag": False, "av": False, "name": "dp_only"},
    {"dp": False, "trustrag": True, "av": False, "name": "trustrag_only"},
    {"dp": False, "trustrag": False, "av": True, "name": "av_only"},
    {"dp": True, "trustrag": True, "av": False, "name": "dp_trustrag"},
    {"dp": True, "trustrag": False, "av": True, "name": "dp_av"},
    {"dp": False, "trustrag": True, "av": True, "name": "trustrag_av"},
    {"dp": True, "trustrag": True, "av": True, "name": "all_defenses"}
]

def get_defense_config(result_data):
    """Extract defense configuration from result file"""
    config = result_data.get('config', {})
    defenses = config.get('defenses', [])
    
    dp = False
    trustrag = False
    av = False
    
    for defense in defenses:
        if defense.get('name') == 'differential_privacy' and defense.get('enabled'):
            dp = True
        elif defense.get('name') == 'trustrag' and defense.get('enabled'):
            trustrag = True
        elif defense.get('name') == 'attention_filtering' and defense.get('enabled'):
            av = True
    
    return {"dp": dp, "trustrag": trustrag, "av": av}

def config_name(config):
    """Generate readable name for config"""
    if not config["dp"] and not config["trustrag"] and not config["av"]:
        return "no_defense"
    elif config["dp"] and not config["trustrag"] and not config["av"]:
        return "dp_only"
    elif not config["dp"] and config["trustrag"] and not config["av"]:
        return "trustrag_only"
    elif not config["dp"] and not config["trustrag"] and config["av"]:
        return "av_only"
    elif config["dp"] and config["trustrag"] and not config["av"]:
        return "dp_trustrag"
    elif config["dp"] and not config["trustrag"] and config["av"]:
        return "dp_av"
    elif not config["dp"] and config["trustrag"] and config["av"]:
        return "trustrag_av"
    elif config["dp"] and config["trustrag"] and config["av"]:
        return "all_defenses"
    return "unknown"

def main():
    # Analyze result files
    results_dir = Path("data/results")
    runs_by_dataset = defaultdict(list)
    
    print("Analyzing defense configurations in result files...")
    print("=" * 70)
    
    for result_file in sorted(results_dir.glob("run_*.json")):
        with open(result_file) as f:
            data = json.load(f)
        
        dataset = data.get('dataset', 'unknown')
        defense_config = get_defense_config(data)
        config_label = config_name(defense_config)
        
        runs_by_dataset[dataset].append({
            'file': result_file.name,
            'config': defense_config,
            'config_name': config_label
        })
        
        print(f"{result_file.name:<40} {dataset:<12} {config_label}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY BY DATASET")
    print("=" * 70)
    
    datasets = ["nq", "pubmedqa", "triviaqa"]
    expected_config_names = [c["name"] for c in EXPECTED_CONFIGS]
    
    missing_by_dataset = defaultdict(list)
    total_missing = 0
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-" * 70)
        
        completed_configs = set([run['config_name'] for run in runs_by_dataset[dataset]])
        missing_configs = [c for c in expected_config_names if c not in completed_configs]
        
        print(f"Completed: {len(completed_configs)}/8")
        for run in sorted(runs_by_dataset[dataset], key=lambda x: x['config_name']):
            print(f"  ✓ {run['config_name']:<20} ({run['file']})")
        
        if missing_configs:
            print(f"\nMissing: {len(missing_configs)}/8")
            for config in missing_configs:
                print(f"  ✗ {config}")
                missing_by_dataset[dataset].append(config)
                total_missing += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total completed: {sum(len(runs) for runs in runs_by_dataset.values())}/24")
    print(f"Total missing: {total_missing}/24")
    
    if total_missing > 0:
        print(f"\n{'Dataset':<12} {'Missing Configs'}")
        print("-" * 70)
        for dataset in datasets:
            if missing_by_dataset[dataset]:
                configs = ", ".join(missing_by_dataset[dataset])
                print(f"{dataset:<12} {configs}")

if __name__ == "__main__":
    main()
