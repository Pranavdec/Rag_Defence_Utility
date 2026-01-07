#!/usr/bin/env python3
"""
Find run files without evaluations and generate a script to evaluate them.
Usage: python find_and_generate_eval_script.py [run_folder] [eval_folder]
"""

import json
from pathlib import Path
from datetime import datetime
import sys
import os

def get_signature(data):
    config_val = data['config']
    if isinstance(config_val, dict):
        config_val = json.dumps(config_val, sort_keys=True)
    return (data['dataset'], config_val)

def find_runs_without_eval(run_folder: str, eval_folder: str):
    """
    Find run JSON files without matching eval files.
    Returns list of run file paths that need evaluation.
    """
    run_path = Path(run_folder)
    eval_path = Path(eval_folder)
    
    if not run_path.exists():
        print(f"Error: Run folder '{run_folder}' does not exist")
        return []
    
    if not eval_path.exists():
        print(f"Warning: Eval folder '{eval_folder}' does not exist, creating it")
        eval_path.mkdir(parents=True, exist_ok=True)
    
    # Get all run and eval JSON files
    run_files = list(run_path.glob("run_*.json"))
    eval_files = list(eval_path.glob("eval_*.json"))
    
    # 1. Load run data AND track the file paths
    runs = []
    for f in run_files:
        with open(f, 'r') as j:
            # Store as a tuple: (Path, Content)
            runs.append((f, json.load(j)))

    # 2. Load all eval signatures (same as before)
    eval_signatures = set()
    for f in eval_files:
        with open(f, 'r') as j:
            data = json.load(j)
            if 'config' in data and isinstance(data['config'], dict):
                data['config'].pop('judge_llm', None)
            eval_signatures.add(get_signature(data))

    # 3. Find paths (f) where the content (x) signature is missing
    missing_run_files = [f for f, x in runs if get_signature(x) not in eval_signatures]
    
    return missing_run_files

def generate_eval_script(missing_runs: list, output_dir: str = "./"):
    """
    Generate a shell script to evaluate missing runs.
    """
    if not missing_runs:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = f"evaluate_missing_{timestamp}.sh"
    script_path = Path(output_dir) / script_name
    

    script_content = [
        "#!/bin/bash",
        "# Auto-generated script to evaluate missing runs",
        f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Found {len(missing_runs)} runs without evaluations",
        "",
        "set -e",
        "",
        "# Check if Ollama is running, if not start it",
        'if ! pgrep -x "ollama" > /dev/null; then',
        '    echo "Ollama is not running. Starting Ollama service..."',
        "    ollama serve &",
        "    sleep 3  # Wait for service to start",
        '    echo "Starting llama3 model..."',
        "    ollama run llama3 &",
        "    sleep 2  # Wait for model to load",
        '    echo "✓ Ollama started successfully"',
        "else",
        '    echo "✓ Ollama is already running"',
        "fi",
        "",
        "echo '========================================='",
        "echo 'Evaluating missing runs...'",
        "echo '========================================='",
        "echo ''",
    ]
    
    for i, run_file in enumerate(missing_runs, 1):
        run_name = Path(run_file).name
        script_content.append(f"echo '[{i}/{len(missing_runs)}] Evaluating: {run_name}'")
        script_content.append(f"python main.py evaluate '{run_file}' || echo '✗ Failed: {run_name}'")
        script_content.append("echo ''")
    
    script_content.extend([
        "echo '========================================='",
        "echo 'All missing evaluations completed!'",
        "echo 'Check data/metrics/ for results'",
        "echo '========================================='"
    ])
    
    # Write script
    script_path.write_text('\n'.join(script_content))
    
    # Make executable
    script_path.chmod(0o755)
    
    return str(script_path)

def main():
    # Default folders
    default_run_folder = "data/results"
    default_eval_folder = "data/metrics"
    
    # Allow command line arguments
    run_folder = sys.argv[1] if len(sys.argv) > 1 else default_run_folder
    eval_folder = sys.argv[2] if len(sys.argv) > 2 else default_eval_folder
    
    print("=========================================")
    print("Finding runs without evaluations...")
    print("=========================================")
    print(f"Run folder:  {run_folder}")
    print(f"Eval folder: {eval_folder}")
    print("")
    
    # Find missing runs
    missing_runs = find_runs_without_eval(run_folder, eval_folder)
    
    if not missing_runs:
        print("✓ All runs have been evaluated!")
        return
    
    print(f"Found {len(missing_runs)} runs without evaluations:")
    for run_file in missing_runs:
        print(f"  - {Path(run_file).name}")
    print("")
    
    # Generate evaluation script
    script_path = generate_eval_script(missing_runs)
    
    if script_path:
        print("=========================================")
        print(f"Generated evaluation script: {script_path}")
        print("=========================================")
        print("")
        print("To evaluate missing runs, run:")
        print(f"  ./{Path(script_path).name}")
        print("")
        print("Or execute directly:")
        print(f"  bash {script_path}")
        print("")

if __name__ == "__main__":
    main()
