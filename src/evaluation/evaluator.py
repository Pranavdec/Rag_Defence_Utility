"""
Evaluator module for RAG pipeline.
Integrates RAGAS and DeepEval metrics.
"""
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("ollama").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Try importing RAGAS
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas")

# Try importing DeepEval
try:
    from deepeval.metrics import (
        GEval,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric
    )
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate as deepeval_evaluate
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not available. Install with: pip install deepeval")

from .custom_metrics import detect_refusal_simple
from .judge_cache import JudgeCache

os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600000"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "3000"

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    dataset: str
    num_samples: int
    metrics: Dict[str, float]
    per_sample: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None


class RAGEvaluator:
    """
    Evaluator for RAG pipeline results.
    Supports RAGAS and DeepEval metrics.
    """
    
    def __init__(
        self,
        llm_model: str = "ollama/llama3",
        embedding_model: str = "ollama/nomic-embed-text",
        shared_vllm_llm=None,
        cache_dir: str = "data/raw/judge_cache",
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_model: Model for LLM-based metrics (RAGAS/DeepEval)
            embedding_model: Model for embedding-based metrics (RAGAS)
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.shared_vllm_llm = shared_vllm_llm
        self.judge_cache = JudgeCache(cache_dir)
        
        logger.info(f"RAGEvaluator initialized. RAGAS={RAGAS_AVAILABLE}, DeepEval={DEEPEVAL_AVAILABLE}")
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """Load results from a JSON file."""
        with open(results_path, "r") as f:
            return json.load(f)
    
    def compute_latency_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Compute latency statistics."""
        latencies = [r.get("latency_ms", 0) for r in results]
        
        if not latencies:
            return {"avg_latency_ms": 0, "p50_latency_ms": 0, "p99_latency_ms": 0}
        
        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
        }
    
    def compute_refusal_rate(self, results: List[Dict]) -> Dict[str, float]:
        """Compute refusal rate using simple heuristics."""
        refusals = 0
        
        for r in results:
            answer = r.get("answer", "")
            if detect_refusal_simple(answer):
                refusals += 1
        
        rate = refusals / len(results) if results else 0
        
        return {
            "refusal_count": refusals,
            "refusal_rate": rate,
            "total_samples": len(results)
        }
    
    def evaluate_with_ragas(
        self,
        results: List[Dict],
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics with Ollama LLM and local embeddings.
        
        Args:
            results: List of result dicts with 'question', 'answer', 'contexts', 'ground_truth'
            metrics: List of RAGAS metrics to use (default: all available)
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available,  skipping RAGAS evaluation")
            return {}
        
        # Prepare data for RAGAS
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for r in results:
            data["question"].append(r.get("question", ""))
            # RAGAS expects 'answer' not 'generated_answer'
            data["answer"].append(r.get("generated_answer", r.get("answer", "")))
            data["contexts"].append(r.get("contexts", []))
            data["ground_truth"].append(r.get("ground_truth", ""))
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        if metrics is None:
            metrics = [faithfulness, answer_correctness, context_recall]
        
        logger.info(f"Running RAGAS evaluation with {len(metrics)} metrics...")
        
        try:
            # Configure Ollama LLM for RAGAS
            from langchain_ollama import ChatOllama
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            # Extract model name from "ollama/llama3" format
            model_name = self.llm_model.replace("ollama/", "")
            
            # Ollama LLM wrapper
            llm = ChatOllama(model=model_name, temperature=0)
            wrapped_llm = LangchainLLMWrapper(llm)
            
            # Local embeddings (same as used for ingestion)
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
            
            # Set LLM and embeddings for each metric
            for m in metrics:
                if hasattr(m, 'llm'):
                    m.llm = wrapped_llm
                if hasattr(m, 'embeddings'):
                    m.embeddings = wrapped_embeddings
            
            result = ragas_evaluate(
                dataset, 
                metrics=metrics,
                llm=wrapped_llm,
                embeddings=wrapped_embeddings
            )
            
            # Robust Extraction Strategy
            final_metrics = {}
            
            # Strategy 1: Iterate over result (if it behaves like a dict of aggregates)
            try:
                for k, v in result.items():
                   if isinstance(v, (int, float)):
                       final_metrics[k] = float(v)
            except (AttributeError, TypeError, KeyError):
                pass
            
            # Strategy 2: If result.scores exists and is a list (per-sample scores), aggregate manually
            if not final_metrics and hasattr(result, 'scores'):
                scores = result.scores
                if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], dict):
                    logger.info(f"Aggregating {len(scores)} per-sample scores manually...")
                    # Get all keys
                    keys = scores[0].keys()
                    for k in keys:
                        # Filter out non-numeric
                        values = [s[k] for s in scores if isinstance(s.get(k), (int, float))]
                        if values:
                            final_metrics[k] = float(sum(values) / len(values))
            
            # Strategy 3: Try converting to dict as last resort
            if not final_metrics:
                try:
                     final_metrics = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
                except Exception:
                    pass

            return final_metrics
        except ImportError as e:
            logger.error(f"Missing dependency for RAGAS with Ollama: {e}")
            logger.info("Install with: pip install langchain-ollama langchain-community")
            return {"ragas_error": f"Missing dependency: {e}"}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"ragas_error": str(e)}
    
    def evaluate_with_deepeval(
        self,
        results: List[Dict],
        metrics: Optional[List] = None,
        max_concurrent: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate using DeepEval metrics with a configurable judge LLM.
        Supported:
        - ollama/<model>
        - vllm/<hf_model_path>
        
        Args:
            results: List of result dicts
            metrics: List of DeepEval metrics to use (default: RAG standard)
            max_concurrent: Maximum number of concurrent evaluations (default: 5)
        """
        if not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval not available, skipping evaluation")
            return {}
        
        # Configure judge model (required for both default and subset metric selection)
        deepeval_model = None
        model_spec = (self.llm_model or "").strip()
        if model_spec.startswith("vllm/"):
            model_path = model_spec[len("vllm/"):]
            try:
                from .vllm_judge import VLLMJudgeModel
                # Faithfulness / multi-step metrics emit large structured JSON; 512 truncates
                # mid-string and breaks trimAndLoadJson ("Unterminated string").
                deepeval_model = VLLMJudgeModel(
                    model=model_path,
                    shared_llm=self.shared_vllm_llm,
                    temperature=0.0,
                    # Faithfulness truths/claims JSON can be large; must stay under
                    # max_model_len minus prompt tokens (see comprehensive_eval floor).
                    max_tokens=8192,
                )
            except Exception as e:
                logger.error(f"Failed to initialize VLLMJudgeModel: {e}")
                return {"deepeval_error": f"VLLM judge init failed: {e}"}
        else:
            model_name = model_spec.replace("ollama/", "")
            try:
                from deepeval.models import OllamaModel
                deepeval_model = OllamaModel(model=model_name)
            except ImportError:
                logger.warning(
                    "Could not import OllamaModel from deepeval.models. Falling back to default (might fail if API key missing)."
                )
                deepeval_model = self.llm_model
            except Exception as e:
                logger.error(f"Failed to initialize OllamaModel: {e}")
                return {"deepeval_error": f"Model init failed: {e}"}

        # Configure metrics:
        # - metrics=None -> default 4 metrics
        # - metrics=[\"contextual_recall\",\"faithfulness\"] -> subset for sweeps
        if metrics is None or (metrics and all(isinstance(m, str) for m in metrics)):
            metric_map = {
                "answer_relevancy": AnswerRelevancyMetric,
                "faithfulness": FaithfulnessMetric,
                "contextual_relevancy": ContextualRelevancyMetric,
                "contextual_recall": ContextualRecallMetric,
            }
            if metrics and all(isinstance(m, str) for m in metrics):
                selected = [str(m).strip().lower() for m in metrics]
                unknown = [m for m in selected if m not in metric_map]
                if unknown:
                    return {
                        "deepeval_error": f"Unknown deepeval_metrics: {unknown}. Supported: {list(metric_map.keys())}"
                    }
                metric_classes = [metric_map[m] for m in selected]
            else:
                metric_classes = [
                    AnswerRelevancyMetric,
                    FaithfulnessMetric,
                    ContextualRelevancyMetric,
                    ContextualRecallMetric,
                ]

            metrics = []
            for cls in metric_classes:
                if cls is FaithfulnessMetric and model_spec.startswith("vllm/"):
                    # Default None -> "comprehensive list" can explode JSON size and hit
                    # max_tokens / max_model_len mid-string (invalid JSON).
                    metrics.append(
                        FaithfulnessMetric(
                            threshold=0,
                            model=deepeval_model,
                            truths_extraction_limit=12,
                        )
                    )
                else:
                    metrics.append(cls(threshold=0, model=deepeval_model))

        test_cases = []
        for r in results:
            test_case = LLMTestCase(
                input=r.get("question", ""),
                actual_output=r.get("generated_answer", r.get("answer", "")),
                expected_output=r.get("ground_truth", ""),
                retrieval_context=r.get("contexts", [])
            )
            test_cases.append(test_case)
            
        logger.info(f"Running DeepEval evaluation on {len(test_cases)} samples with {len(metrics)} metrics...")

        model_spec_for_conc = (self.llm_model or "").strip()
        if model_spec_for_conc.startswith("vllm/") and max_concurrent > 1:
            # vLLM in-process client isn't reliably thread-safe across many concurrent calls.
            # Prefer stability; vLLM still runs fast per-call and caches help a lot.
            logger.warning("Clamping DeepEval max_concurrent to 1 for vLLM judge stability.")
            max_concurrent = 1
        logger.info(f"Using max_concurrent={max_concurrent} for parallel evaluation")

        # Cache-aware metric evaluation:
        # DeepEval's `evaluate()` does not skip already-scored samples, so we
        # compute per-sample scores ourselves and cache them on disk.
        import asyncio

        def _judge_name() -> str:
            spec = (self.llm_model or "").strip()
            if spec.startswith("vllm/"):
                return spec[len("vllm/") :]
            if spec.startswith("ollama/"):
                return spec
            return spec

        async def _score_metric(metric, metric_name: str) -> float:
            sem = asyncio.Semaphore(max(1, int(max_concurrent)))
            scores: List[float] = []
            failures: List[str] = []
            judge_model = _judge_name()

            async def _one(tc: LLMTestCase) -> None:
                payload = {
                    "judge_model": judge_model,
                    "metric": metric_name,
                    "question": tc.input,
                    "expected_output": tc.expected_output,
                    "actual_output": tc.actual_output,
                    "retrieval_context": tc.retrieval_context,
                }
                cached = self.judge_cache.get(payload)
                if cached is not None:
                    scores.append(float(cached))
                    return
                async with sem:
                    try:
                        s = await metric.a_measure(
                            tc, _show_indicator=False, _log_metric_to_confident=False
                        )
                        if isinstance(s, (int, float)):
                            s = float(s)
                            self.judge_cache.put(payload, s)
                            scores.append(s)
                        else:
                            msg = f"non-numeric return ({type(s).__name__})"
                            failures.append(msg)
                            logger.debug(f"DeepEval {metric_name}: {msg} for input={tc.input[:60]!r}")
                    except Exception as exc:
                        import traceback
                        tb = traceback.format_exc()
                        failures.append(str(exc))
                        logger.warning(
                            f"DeepEval {metric_name}: a_measure failed for input={tc.input[:60]!r}: "
                            f"{exc}\n{tb}"
                        )

            await asyncio.gather(*[_one(tc) for tc in test_cases])
            if failures:
                logger.warning(
                    f"DeepEval {metric_name}: {len(failures)}/{len(test_cases)} samples failed. "
                    f"First error: {failures[0]}"
                )
            return float(sum(scores) / len(scores)) if scores else 0.0

        try:
            async def _run_all() -> Dict[str, float]:
                out: Dict[str, float] = {}
                for metric in metrics:
                    raw_name = getattr(metric, "name", None) or metric.__class__.__name__
                    out_key = f"deepeval_{raw_name.lower().replace(' ', '_')}"
                    out[out_key] = await _score_metric(metric, raw_name)
                return out

            return asyncio.run(_run_all())
        except RuntimeError:
            # Already in an event loop (e.g. notebooks) - do a blocking fallback.
            final: Dict[str, float] = {}
            judge_model = _judge_name()
            for metric in metrics:
                raw_name = getattr(metric, "name", None) or metric.__class__.__name__
                out_key = f"deepeval_{raw_name.lower().replace(' ', '_')}"
                vals: List[float] = []
                for tc in test_cases:
                    payload = {
                        "judge_model": judge_model,
                        "metric": raw_name,
                        "question": tc.input,
                        "expected_output": tc.expected_output,
                        "actual_output": tc.actual_output,
                        "retrieval_context": tc.retrieval_context,
                    }
                    cached = self.judge_cache.get(payload)
                    if cached is not None:
                        vals.append(float(cached))
                        continue
                    try:
                        s = metric.measure(tc, _show_indicator=False, _log_metric_to_confident=False)
                        if isinstance(s, (int, float)):
                            s = float(s)
                            self.judge_cache.put(payload, s)
                            vals.append(s)
                        else:
                            logger.debug(
                                f"DeepEval {raw_name}: non-numeric return ({type(s).__name__}) "
                                f"for input={tc.input[:60]!r}"
                            )
                    except Exception as exc:
                        import traceback
                        logger.warning(
                            f"DeepEval {raw_name}: measure failed for input={tc.input[:60]!r}: "
                            f"{exc}\n{traceback.format_exc()}"
                        )
                        continue
                if not vals:
                    logger.warning(f"DeepEval {raw_name}: all {len(test_cases)} samples failed — score will be 0.0")
                final[out_key] = float(sum(vals) / len(vals)) if vals else 0.0
            return final
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"deepeval_error": str(e)}

    def evaluate_all(
        self,
        results_path: str,
        use_ragas: bool = False,
        use_deepeval: bool = True,
        deepeval_max_concurrent: int = 5,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Run all evaluations on a results file.
        
        Args:
            results_path: Path to the results JSON file
            use_ragas: Whether to run RAGAS metrics
            use_deepeval: Whether to run DeepEval metrics
            deepeval_max_concurrent: Max concurrent evaluations for DeepEval
            evaluation_config: Optional dictionary containing evaluation-time configuration
        """
        data = self.load_results(results_path)
        results = data.get("results", [])
        dataset_name = data.get("dataset", "unknown")
        
        logger.info(f"Evaluating {len(results)} results from {dataset_name}...")
        
        all_metrics = {}
        
        # Latency stats
        latency_stats = self.compute_latency_stats(results)
        all_metrics.update(latency_stats)
        logger.info(f"Latency: {latency_stats}")
        
        # Refusal rate
        refusal_stats = self.compute_refusal_rate(results)
        all_metrics.update(refusal_stats)
        logger.info(f"Refusal: {refusal_stats}")
        
        # RAGAS metrics
        if use_ragas and RAGAS_AVAILABLE:
            ragas_metrics = self.evaluate_with_ragas(results)
            all_metrics.update({f"ragas_{k}": v for k, v in ragas_metrics.items()})
            logger.info(f"RAGAS: {ragas_metrics}")
            
        # DeepEval metrics
        if use_deepeval and DEEPEVAL_AVAILABLE:
            deepeval_metrics = self.evaluate_with_deepeval(
                results, 
                max_concurrent=deepeval_max_concurrent
            )
            all_metrics.update(deepeval_metrics)
            logger.info(f"DeepEval: {deepeval_metrics}")
        
        # Extract config from results if available
        # This is the config used during INFERENCE (retrieval/generation)
        result_config = data.get("config", {})
        
        # Add evaluation-specific config if provided
        if evaluation_config:
            # We can either merge it at the top level or add a subsection
            # Adding a subsection is safer to distinguish inference vs evaluation params
            result_config["evaluation_config"] = evaluation_config
            
            # Also ensure specific params used are reflected
            if "evaluation" not in result_config:
                result_config["evaluation"] = {}
            result_config["evaluation"]["deepeval_max_concurrent"] = deepeval_max_concurrent
        
        result_config["judge_llm"] = self.llm_model
        
        return EvaluationResult(
            dataset=dataset_name,
            num_samples=len(results),
            metrics=all_metrics,
            per_sample=results,
            config=result_config
        )
    
    def save_evaluation(
        self,
        eval_result: EvaluationResult,
        output_path: str
    ):
        """Save evaluation results to JSON and metrics-only CSV."""
        # Save full JSON
        output = {
            "dataset": eval_result.dataset,
            "num_samples": eval_result.num_samples,
            "config": eval_result.config,
            "metrics": eval_result.metrics,
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Evaluation saved to {output_path}")
        
        # Save metrics-only CSV
        import csv
        csv_path = output_path.replace(".json", "_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            # Write metrics
            for k, v in eval_result.metrics.items():
                writer.writerow([k, v])
            
            # Write config metadata (flattened)
            if eval_result.config:
                writer.writerow([]) # Empty row separator
                writer.writerow(["CONFIG", ""])
                
                # Helper to flatten dict
                def flatten_dict(d, parent_key='', sep='.'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flat_config = flatten_dict(eval_result.config)
                for k, v in flat_config.items():
                     writer.writerow([k, v])
        
        logger.info(f"Metrics CSV saved to {csv_path}")


def main():
    """CLI for running evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output", "-o", help="Output path for evaluation results")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS evaluation")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_all(
        args.results_file,
        use_ragas=not args.no_ragas
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset: {result.dataset}")
    print(f"Samples: {result.num_samples}")
    print("\nMetrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if args.output:
        evaluator.save_evaluation(result, args.output)


if __name__ == "__main__":
    main()
