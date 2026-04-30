"""
Experiment 1 — Utility Paradox full runner.

Loops over:
  datasets × C0–C7 (DP × TrustRAG × PAD) × test_seeds {42, 123, 456, 1234, 3546}

For each run invokes comprehensive_eval.py --phase final (or --phase smoke for dry-runs),
writing per-seed evaluation JSON into a stable output tree:

  data/results/exp1_runs/<dataset>/<config_id>/seed_<test_seed>/evaluation_sequential_3stage.json

After all runs complete (or after --aggregate-only is passed), builds four summary tables
(one per dataset):

  data/results/exp1_summary/summary_<dataset>.csv
  data/results/exp1_summary/summary_<dataset>.md

Each table:
  rows = C0 … C7  (labeled with DP/T/PAD flags)
  cols = Answer Relevancy | Faithfulness | Contextual Relevancy | Contextual Recall
  cells = "mean ± std"  (sample std, ddof=1 when n>1, else population std=0)

A manifest of all completed runs is written to:
  data/results/exp1_runs/manifest.jsonl

Usage:
  # Dry-run smoke (20 queries) on nq only:
  python scripts/run_exp1_utility_suite.py --phase smoke --datasets nq

  # Full Exp1 on all datasets:
  python scripts/run_exp1_utility_suite.py --phase final

  # Re-build summary tables from existing results without re-running:
  python scripts/run_exp1_utility_suite.py --aggregate-only
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Project root and Python interpreter
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VENV_PYTHON = PROJECT_ROOT / "env" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

# Five replication seeds — test_seed varies; ingestion_seed stays fixed.
EXP1_TEST_SEEDS: List[int] = [42, 123, 456, 1234, 3546]
FIXED_INGESTION_SEED: int = 42
FIXED_INGESTION_SIZE: int = 1000

# Defense grid: C0 … C7  (DP × TrustRAG × PAD)
# Each entry: (dp_enabled, trustrag_enabled, pad_enabled)
CONFIG_GRID: List[Tuple[str, bool, bool, bool]] = [
    ("C0", False, False, False),
    ("C1", True,  False, False),
    ("C2", False, True,  False),
    ("C3", False, False, True),
    ("C4", True,  True,  False),
    ("C5", True,  False, True),
    ("C6", False, True,  True),
    ("C7", True,  True,  True),
]

CONFIG_LABEL: Dict[str, str] = {
    cid: f"{cid}  DP={'ON' if dp else 'OFF'}  T={'ON' if tr else 'OFF'}  PAD={'ON' if pad else 'OFF'}"
    for cid, dp, tr, pad in CONFIG_GRID
}

METRIC_KEYS = ["answer_relevancy", "faithfulness", "contextual_relevancy", "contextual_recall"]
METRIC_LABELS = ["Answer Relevancy", "Faithfulness", "Contextual Relevancy", "Contextual Recall"]

DEFAULT_DATASETS = ["nq", "triviaqa", "pubmedqa", "financebench"]

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "default_utility_config.yaml"

# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def _base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_run_config(
    dataset: str,
    dp_enabled: bool,
    trustrag_enabled: bool,
    pad_enabled: bool,
    test_seed: int,
    phase: str,
) -> Dict[str, Any]:
    """Return a config dict for one run, patching the base utility config."""
    cfg = _base_config()

    # Data
    cfg["data"]["dataset"] = dataset
    cfg["data"]["ingestion_seed"] = FIXED_INGESTION_SEED
    cfg["data"]["ingestion_size"] = FIXED_INGESTION_SIZE
    cfg["data"]["test_seed"] = test_seed
    cfg["data"]["test_size"] = 20 if phase == "smoke" else 200
    cfg["data"]["clear_chroma_before_run"] = False

    # DeepEval on for all Exp1 final runs (metrics are the deliverable).
    cfg["evaluation"]["skip_deepeval"] = False
    cfg["evaluation"]["deepeval_max_concurrent"] = 1  # vLLM stability

    # LLM provider: PAD requires HF; otherwise use vLLM for speed.
    if pad_enabled:
        cfg["system"]["llm"]["provider"] = "huggingface"
    else:
        cfg["system"]["llm"]["provider"] = "vllm"
    cfg["system"]["judge_llm"] = "vllm/meta-llama/Llama-3.1-8B-Instruct"

    # Defenses
    for defense in cfg.get("defenses", []):
        name = defense.get("name", "")
        if name == "differential_privacy":
            defense["enabled"] = dp_enabled
        elif name == "trustrag":
            defense["enabled"] = trustrag_enabled
        elif name == "privacy_aware_decoding":
            defense["enabled"] = pad_enabled

    # Attacks and ADO off
    for atk in cfg.get("attack", {}).values():
        if isinstance(atk, dict):
            atk["enabled"] = False
    cfg["ado"]["enabled"] = False

    return cfg


# ---------------------------------------------------------------------------
# Output layout helpers
# ---------------------------------------------------------------------------

def _run_output_dir(results_root: Path, dataset: str, config_id: str, seed: int) -> Path:
    return results_root / "exp1_runs" / dataset / config_id / f"seed_{seed}"


def _manifest_path(results_root: Path) -> Path:
    return results_root / "exp1_runs" / "manifest.jsonl"


def _summary_dir(results_root: Path) -> Path:
    return results_root / "exp1_summary"


# ---------------------------------------------------------------------------
# Running a single evaluation
# ---------------------------------------------------------------------------

def run_one(
    dataset: str,
    config_id: str,
    dp_enabled: bool,
    trustrag_enabled: bool,
    pad_enabled: bool,
    test_seed: int,
    phase: str,
    results_root: Path,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build a config, invoke comprehensive_eval.py, symlink output into stable tree.
    Returns manifest row dict, or None if the run failed.
    """
    out_dir = _run_output_dir(results_root, dataset, config_id, test_seed)
    sentinel = out_dir / "evaluation_sequential_3stage.json"
    if sentinel.exists():
        print(f"  [SKIP] {dataset}/{config_id}/seed_{test_seed} — result already exists.")
        return _manifest_row(dataset, config_id, test_seed, str(sentinel), skipped=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_run_config(dataset, dp_enabled, trustrag_enabled, pad_enabled, test_seed, phase)
    # Point results path at our stable per-seed directory so evaluator writes there directly.
    cfg["paths"]["results"] = str(out_dir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"exp1_{dataset}_{config_id}_s{test_seed}_",
        dir=PROJECT_ROOT / "config", delete=False
    ) as fh:
        yaml.dump(cfg, fh, default_flow_style=False)
        tmp_config = fh.name

    print(f"\n{'='*70}")
    print(f"  RUN: dataset={dataset}  config={config_id}  seed={test_seed}  phase={phase}")
    print(f"       DP={dp_enabled}  TrustRAG={trustrag_enabled}  PAD={pad_enabled}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [DRY RUN] would invoke: {PYTHON} scripts/comprehensive_eval.py --config {tmp_config} --phase {phase}")
        os.unlink(tmp_config)
        return None

    t0 = time.time()
    try:
        cmd = [PYTHON, str(SCRIPT_DIR / "comprehensive_eval.py"), "--config", tmp_config, "--phase", phase]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            print(f"  [ERROR] Run returned exit code {result.returncode} after {elapsed:.0f}s")
    except Exception as exc:
        print(f"  [ERROR] Run crashed: {exc}")
        success = False
        elapsed = time.time() - t0
    finally:
        try:
            os.unlink(tmp_config)
        except Exception:
            pass

    # The evaluator writes under cfg["paths"]["results"]/<timestamp>_eval/
    # Find the produced JSON and move it to our stable location.
    json_path: Optional[Path] = None
    for candidate in sorted(out_dir.glob("*_eval/evaluation_sequential_3stage.json")):
        json_path = candidate
        break  # take first (only one expected)

    if json_path and json_path.exists():
        stable = out_dir / "evaluation_sequential_3stage.json"
        json_path.rename(stable)
        json_path = stable
        # Clean up the now-empty timestamped subdir
        ts_dir = stable.parent if stable.parent != out_dir else None
    else:
        # Evaluator may have written directly if results path was already flat
        direct = out_dir / "evaluation_sequential_3stage.json"
        json_path = direct if direct.exists() else None

    row = _manifest_row(
        dataset, config_id, test_seed,
        str(json_path) if json_path else "",
        elapsed_s=int(elapsed),
        success=success,
    )
    _append_manifest(results_root, row)
    return row


def _manifest_row(
    dataset: str, config_id: str, seed: int, json_path: str,
    elapsed_s: int = 0, success: bool = True, skipped: bool = False
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "config_id": config_id,
        "test_seed": seed,
        "success": success,
        "skipped": skipped,
        "elapsed_s": elapsed_s,
        "result_json": json_path,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _append_manifest(results_root: Path, row: Dict[str, Any]) -> None:
    path = _manifest_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Aggregation: build summary tables
# ---------------------------------------------------------------------------

def _read_metrics(json_path: str) -> Optional[Dict[str, float]]:
    """Read the four DeepEval metrics from an evaluation JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        m = data.get("metrics", {})
        return {k: float(m.get(k, 0.0)) for k in METRIC_KEYS}
    except Exception as exc:
        print(f"  [WARN] Could not read {json_path}: {exc}")
        return None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def _format_cell(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def build_summary_tables(results_root: Path, datasets: List[str]) -> None:
    """
    For each dataset, collect per-seed metric scores and emit CSV + Markdown tables.
    """
    summary_dir = _summary_dir(results_root)
    summary_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"\nBuilding summary table for {dataset} ...")

        # Collect scores per (config_id, metric)
        scores: Dict[str, Dict[str, List[float]]] = {
            cid: {k: [] for k in METRIC_KEYS}
            for cid, *_ in CONFIG_GRID
        }

        for cid, dp, tr, pad in CONFIG_GRID:
            for seed in EXP1_TEST_SEEDS:
                json_path = _run_output_dir(results_root, dataset, cid, seed) / "evaluation_sequential_3stage.json"
                if not json_path.exists():
                    # Check inside timestamped subdirs (un-normalized runs)
                    matches = list((_run_output_dir(results_root, dataset, cid, seed)).glob(
                        "*_eval/evaluation_sequential_3stage.json"
                    ))
                    json_path = matches[0] if matches else json_path
                if json_path.exists():
                    m = _read_metrics(str(json_path))
                    if m:
                        for k in METRIC_KEYS:
                            scores[cid][k].append(m[k])

        # Build table
        header_row = ["Config"] + METRIC_LABELS
        rows: List[List[str]] = []
        for cid, dp, tr, pad in CONFIG_GRID:
            row = [CONFIG_LABEL[cid]]
            for k in METRIC_KEYS:
                vals = scores[cid][k]
                if vals:
                    mean, std = _mean_std(vals)
                    row.append(_format_cell(mean, std))
                else:
                    row.append("—")
            rows.append(row)

        # CSV
        csv_path = summary_dir / f"summary_{dataset}.csv"
        with open(csv_path, "w") as f:
            f.write(",".join(header_row) + "\n")
            for row in rows:
                f.write(",".join(f'"{c}"' for c in row) + "\n")
        print(f"  Saved CSV: {csv_path}")

        # Markdown
        md_path = summary_dir / f"summary_{dataset}.md"
        col_widths = [
            max(len(header_row[i]), max((len(row[i]) for row in rows), default=0))
            for i in range(len(header_row))
        ]

        def _pad(s: str, w: int) -> str:
            return s.ljust(w)

        with open(md_path, "w") as f:
            f.write(f"# Experiment 1 Utility Metrics — {dataset}\n\n")
            f.write(f"Seeds used: {EXP1_TEST_SEEDS}  |  Cells: mean ± sample std\n\n")
            header_line = "| " + " | ".join(_pad(h, col_widths[i]) for i, h in enumerate(header_row)) + " |"
            sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(header_row))) + " |"
            f.write(header_line + "\n")
            f.write(sep_line + "\n")
            for row in rows:
                f.write("| " + " | ".join(_pad(row[i], col_widths[i]) for i in range(len(row))) + " |\n")
        print(f"  Saved Markdown: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1 — Utility Paradox full runner (8 configs × 5 seeds × datasets)"
    )
    parser.add_argument(
        "--phase", choices=["smoke", "final"], default="final",
        help="smoke=20 queries (validation); final=200 queries (real measurements)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        metavar="DATASET",
        help=f"Datasets to run (default: {DEFAULT_DATASETS})"
    )
    parser.add_argument(
        "--configs", nargs="+", default=[c for c, *_ in CONFIG_GRID],
        metavar="CID",
        help="Subset of config IDs to run (e.g. C0 C1 C3); default: all C0–C7"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=EXP1_TEST_SEEDS,
        metavar="SEED",
        help=f"test_seed values (default: {EXP1_TEST_SEEDS})"
    )
    parser.add_argument(
        "--results-dir", default=str(PROJECT_ROOT / "data" / "results"),
        help="Root results directory (default: data/results)"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip running evaluations; only (re-)build summary tables from existing results"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing"
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    selected_configs = {c for c in args.configs}
    grid = [(cid, dp, tr, pad) for cid, dp, tr, pad in CONFIG_GRID if cid in selected_configs]

    print(f"\n{'#'*70}")
    print(f"# Experiment 1 — Utility Paradox Suite")
    print(f"#   phase:    {args.phase}")
    print(f"#   datasets: {args.datasets}")
    print(f"#   configs:  {[c for c, *_ in grid]}")
    print(f"#   seeds:    {args.seeds}")
    print(f"#   results:  {results_root}")
    print(f"{'#'*70}\n")

    if not args.aggregate_only:
        total = len(args.datasets) * len(grid) * len(args.seeds)
        done = 0
        failures = []

        for dataset in args.datasets:
            for cid, dp, tr, pad in grid:
                for seed in args.seeds:
                    done += 1
                    print(f"\n[{done}/{total}]", end=" ")
                    row = run_one(
                        dataset=dataset,
                        config_id=cid,
                        dp_enabled=dp,
                        trustrag_enabled=tr,
                        pad_enabled=pad,
                        test_seed=seed,
                        phase=args.phase,
                        results_root=results_root,
                        dry_run=args.dry_run,
                    )
                    if row and not row.get("success", True) and not row.get("skipped"):
                        failures.append(f"{dataset}/{cid}/seed_{seed}")

        if failures:
            print(f"\n[WARN] {len(failures)} runs failed: {failures}")
        else:
            print("\nAll runs completed successfully.")

    if not args.dry_run:
        build_summary_tables(results_root, args.datasets)

    print(f"\nManifest: {_manifest_path(results_root)}")
    print(f"Summary tables: {_summary_dir(results_root)}/")


if __name__ == "__main__":
    main()
