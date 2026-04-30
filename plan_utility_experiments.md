---
name: Utility Paradox Final
overview: Final, implementation-ready protocol + run plan derived from `utility_paradox_v2_807da446.plan.md`, aligned to current repo defaults (TrustRAG/PAD from `TrustRAG/` and `PAD/`), with your requested DP override and an efficient plan-consulting runner (vLLM batching when PAD=OFF, sequential HF when PAD=ON).
todos: []
isProject: false
---

# Utility Paradox Final

## What you’re doing (high level)
Run **utility-only** experiments (attacks/ADO OFF) for a **fixed 200-query set** per seed, comparing utility across defense subsets and then sweeping key hyperparameters—while keeping runs reproducible and getting early signals quickly.

Core principle:
- **PAD=OFF:** keep the existing **3-stage Modular RAG batching** (harvest → cleanup → vLLM batch generation → DeepEval metrics).
- **PAD=ON:** switch to a **PAD-safe sequential HF generation path** (PAD’s logits hook must be applied).

## Inputs and fixed invariants
### Utility-only / evaluation-only
- Set all attacks disabled: `attack.*.enabled=false`.
- Set ADO disabled: `ado.enabled=false`.
- Utility metrics only via DeepEval: Contextual Recall, Contextual Relevancy, Answer Relevancy, Faithfulness.

### Query set
- Always use `data.test_size=200` for final runs.
- Use `data.test_seed` to select the 200-query sample deterministically.

### Retrieval
- Keep `retrieval.top_k=5`.
- Keep chunking (`retrieval.chunk_size=512`, `retrieval.chunk_overlap=50`) unless you explicitly decide otherwise.


### DP defaults
- `differential_privacy.method = dp_pure`
- `differential_privacy.epsilon = 5.0`
- `differential_privacy.delta = 1e-3`
(Keep `candidate_multiplier=3` )

### TrustRAG defaults
Use current repo/TrustRAG implementation defaults:
- `trustrag.similarity_threshold = 0.88`
- `trustrag.rouge_threshold = 0.25`

(Keep `candidate_multiplier=3` )

### PAD defaults
Use current repo/PAD implementation defaults:
- `noise_type = adaptive`
- `epsilon = 0.2`
- `alpha = 10.0`
- `delta = 1e-5`
- `enable_screening = True`
- `enable_calibration = True`
- `noise_amplification = 3.0`
- `min_sensitivity = 0.4`
- `static_noise_scale = 0.1`

(Keep `candidate_multiplier=3`.)

Important constraint:
- When PAD is enabled, set `system.llm.provider = huggingface` (PAD logits hook does not apply under `vllm`/`ollama`).

## Experiment 1: Utility Paradox (8 static defense combinations)
### Datasets (as in your plan)
- Start with `nq`, then add `triviaqa` and `pubmedqa` (and optionally `financebench` if you want later; ensure loaders exist and DeepEval is stable).

### Defense grid (8)
Cross product over `DP`, `TrustRAG`, `PAD` enabled/disabled:
- C0: DP OFF, T OFF, PAD OFF
- C1: DP ON,  T OFF, PAD OFF
- C2: DP OFF, T ON,  PAD OFF
- C3: DP OFF, T OFF, PAD ON
- C4: DP ON,  T ON,  PAD OFF
- C5: DP ON,  T OFF, PAD ON
- C6: DP OFF, T ON,  PAD ON
- C7: DP ON,  T ON,  PAD ON

### Seeds
- Total 5 seeds per dataset-defense config.
- Include your specified seeds `{42, 123, 456, 1234, 3546}`

Within each seed:
- Use the same 200-query set for all 8 configs (only defense toggles differ).

### Corpus/in-context
- Use your explicit distractor ingestion + corpus cloning approach if you want realistic retrieval crowding.
- Golden passage ingestion must happen in the same combined DB directory that distractors were cloned into.

### Outputs and reporting
- For each dataset and each config:
  - compute mean ± variation across seeds (and bootstrap CI if you implement per-query scores).

Note on bootstrap:
- The current scripts often compute aggregated DeepEval metrics; if you want true bootstrap over 200 queries, store per-query DeepEval outputs or recompute per bootstrap sample.

## Experiment 2: Parameter Sensitivity Sweeps (single-defense at a time)
Attacks/ADO OFF.
Use the same 200-query sets as Experiment 1.

### Replication policy
- Primary sweep seed: `42`.
- Replication seeds: rerun only best and worst configs per defense on `123` and `1234`.

### DP-RAG sweep
- Sweep `epsilon ∈ {0.1, 0.5, 1.0, 3.0, 5.0, 10.0}`
- For each epsilon, test both methods:
  - `dp_pure`
  - `dp_approx`
- Keep `delta=1e-3` (as per your starting-now rule).

### TrustRAG sweep (2D grid)
- Sweep:
  - `similarity_threshold ∈ {0.75, 0.82, 0.88, 0.92}`
  - `rouge_threshold ∈ {0.15, 0.25, 0.35}`

### PAD sweep
Two sub-experiments:

A) Static mode
- `noise_type=static`
- Sweep:
  - `static_noise_scale ∈ {0.01, 0.05, 0.1, 0.5, 1.0}`
- Sweep `epsilon ∈ {0.5, 1.0, 3.0, 5.0}`
- Keep `enable_screening=False` and `enable_calibration=False` for this sweep (as per your protocol).

B) Adaptive mode ablation
- `noise_type=adaptive`
- Sweep:
  - `enable_screening ∈ {False, True}`
  - `enable_calibration ∈ {False, True}`
  - and `noise_amplification ∈ {1.0, 2.0}`
- Keep other PAD parameters at defaults from `PAD/`.

## Fastest path to early results (no code change)
Run a 3-phase schedule:

Phase 0 (smoke):
- `data.test_size=20`
- 1 dataset (start with `nq`).
- 5 defense configs only: `C0, C1, C2, C3, C7`.

Phase 1 (directional):
- `data.test_size=50`
- same dataset; full 8 configs.

Phase 2 (final):
- `data.test_size=200`
- run full Experiment 1/2.

If PAD-enabled runs are too slow, keep PAD configs only for Phase 1 early checks (then scale).

### Dispatch logic
- If `privacy_aware_decoding.enabled=false`:
  - set `system.llm.provider=vllm`
  - reuse your existing Modular pipeline 3-stage harvest→cleanup→vLLM batch generation and DeepEval metrics.
- If PAD is enabled:
  - set `system.llm.provider=huggingface`
  - use a PAD-safe sequential evaluation that calls `ModularRAG.run_single()` per query and aggregates DeepEval metrics.

### Run manifest
Every run writes a manifest entry:
- dataset, experiment, seed, config ID
- all defense hyperparameters
- whether PAD is enabled
- output JSON path
- the plan path used

This provides traceability without changing the core evaluation logic.

## How to phrase “modular design” (paper language)
You can still claim modular design:
- Retrieval and defenses are toggled via a defense manager layer.
- Generation is modular: `create_generator()` selects a generator backend.
- PAD changes the generator backend’s execution (HF logits processor) rather than being a separate “hook-only” retrieval layer.

Recommended phrasing:
> “Our system maintains a modular architecture where retrieval-time defenses are controlled independently from the generation backend; PAD is implemented as a generator-level privacy mechanism using HF logits processors, enabling configuration-driven swapping of the generation execution path while preserving modular defense orchestration.”

## Suggested files to create/modify
- Update defaults:
  - `config/default_utility_config.yaml`
- Add runner:
  - `scripts/run_utility_protocol_from_plan.py`
- Add PAD-safe sequential runner (utility-only):
  - `scripts/utility_runner_pad_sequential.py`
- Optional (but recommended):
  - `scripts/early_smoke_schedule.sh` or a helper mode in the runner to switch `test_size` and config subsets.

## Todos (implementation checklist)
2. Implement plan-driven orchestrator:
   - parse `/home/crk/.cursor/plans/utility_paradox_v2_807da446.plan.md`
   - generate per-run configs (8 configs × seeds + sweep configs)
   - write run manifest entries
3. Implement PAD-safe sequential utility runner:
   - uses HF generation + per-query `ModularRAG.run_single()`
   - computes the same DeepEval utility metrics as the vLLM path
4. Add staged early run mode:
   - auto-run smoke (`test_size=20`) then full (`test_size=200`)
5. (If you truly need bootstrap CI over 200 queries)
   - store per-query DeepEval metric scores so bootstrap can be computed without re-calling the judge repeatedly.
