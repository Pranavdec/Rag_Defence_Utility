# Experiment 1 — Implementation notes

## Supported orchestration path

The planned driver `run_utility_protocol_from_plan.py` was never implemented.
The supported orchestration is now:

```
scripts/run_exp1_utility_suite.py   ← Exp1 runner (all datasets × C0–C7 × 5 seeds)
scripts/comprehensive_eval.py       ← Per-run evaluator (invoked by the suite runner)
```

Speedup infrastructure already in-repo:
- vLLM judge: `src/evaluation/vllm_judge.py`
- Shared vLLM engine: `scripts/comprehensive_eval.py` (`_shared_vllm_llm`)
- Retrieval cache: `src/core/retrieval_cache.py`
- Judge score cache: `src/evaluation/judge_cache.py`
- Smoke/final phase: `--phase smoke|final` flag on `comprehensive_eval.py`

## Distractor ingestion (combined-corpus protocol)

`ModularRAG.ingest()` indexes benchmark gold passages only (from the QA loader).
For the combined-corpus methodology in `plan_utility_experiments.md §Corpus/in-context`:

1. Run `scripts/ingest_distractors.py` first to build the distractor Chroma base DB.
2. Clone/merge that base DB into the per-dataset persist directory used by `ModularRAG`.
   The persist dir is `paths.chroma_db/<dataset_name>_seed<ingestion_seed>/`.
3. Then run `scripts/run_exp1_utility_suite.py` with `data.clear_chroma_before_run: false`
   (the default); `ModularRAG.ingest()` will detect the collection as already populated
   (gold passages are present alongside distractors) and skip re-ingestion.

If you want gold-only (no distractors) for an ablation, simply do not run step 1–2;
`run_exp1_utility_suite.py` works identically — only the retrieval crowding differs.
