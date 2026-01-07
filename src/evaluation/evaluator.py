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
        embedding_model: str = "ollama/nomic-embed-text"
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_model: Model for LLM-based metrics (RAGAS/DeepEval)
            embedding_model: Model for embedding-based metrics (RAGAS)
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
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
        Evaluate using DeepEval metrics with Ollama LLM.
        
        Args:
            results: List of result dicts
            metrics: List of DeepEval metrics to use (default: RAG standard)
            max_concurrent: Maximum number of concurrent evaluations (default: 5)
        """
        if not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval not available, skipping evaluation")
            return {}
        
        # Configure metrics
        if metrics is None:
            # Initialize Ollama model
            # Extract model name from "ollama/llama3" -> "llama3"
            model_name = self.llm_model.replace("ollama/", "")
            try:
                # Late import to avoid top-level dependency issues if not installed
                from deepeval.models import OllamaModel
                deepeval_model = OllamaModel(model=model_name)
            except ImportError:
                logger.warning("Could not import OllamaModel from deepeval.models. Falling back to default (might fail if API key missing).")
                deepeval_model = self.llm_model # Fallback to string
            except Exception as e:
                logger.error(f"Failed to initialize OllamaModel: {e}")
                return {"deepeval_error": f"Model init failed: {e}"}

            metrics = [
                AnswerRelevancyMetric(threshold=0, model=deepeval_model),
                FaithfulnessMetric(threshold=0, model=deepeval_model),
                ContextualRelevancyMetric(threshold=0, model=deepeval_model),
                # ContextualPrecisionMetric(threshold=0, model=deepeval_model),
                ContextualRecallMetric(threshold=0, model=deepeval_model)
            ]

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
        logger.info(f"Using max_concurrent={max_concurrent} for parallel evaluation")
        
        try:
            # Import AsyncConfig and ErrorConfig to control parallelization and error handling
            from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
            
            # Configure async settings to reduce parallelization
            async_config = AsyncConfig(
                run_async=True,
                max_concurrent=max_concurrent,
                throttle_value=0
            )
            
            # Configure error handling to ignore errors and continue evaluation
            error_config = ErrorConfig(ignore_errors=True)
            
            eval_results = deepeval_evaluate(
                test_cases, 
                metrics=metrics,
                async_config=async_config,
                error_config=error_config
            )
            
            final_metrics = {}
            metric_sums = {}
            metric_counts = {}
            
            # DeepEval returns an EvaluationResult object with test_results attribute
            # or the result might be directly iterable
            test_results = eval_results
            if hasattr(eval_results, 'test_results'):
                test_results = eval_results.test_results
            
            # Extract scores from each test result
            for result in test_results:
                # Try to access metrics_data or metrics_metadata
                metrics_list = None
                if hasattr(result, 'metrics_data'):
                    metrics_list = result.metrics_data
                elif hasattr(result, 'metrics_metadata'):
                    metrics_list = result.metrics_metadata
                elif hasattr(result, 'metrics'):
                    metrics_list = result.metrics
                
                if metrics_list:
                    for metric_data in metrics_list:
                        # Extract name and score
                        name = getattr(metric_data, 'name', None) or getattr(metric_data, 'metric', None) or metric_data.__class__.__name__
                        score = getattr(metric_data, 'score', None)
                        
                        if score is not None and name:
                            if name not in metric_sums:
                                metric_sums[name] = 0.0
                                metric_counts[name] = 0
                            
                            metric_sums[name] += float(score)
                            metric_counts[name] += 1
            
            # If still empty, try extracting from the metrics objects directly
            if not metric_sums:
                for metric in metrics:
                    name = getattr(metric, 'name', metric.__class__.__name__)
                    score = getattr(metric, 'score', None)
                    if score is not None:
                        metric_sums[name] = float(score)
                        metric_counts[name] = 1

            for name, total in metric_sums.items():
                if metric_counts[name] > 0:
                    key = f"deepeval_{name.lower().replace(' ', '_')}"
                    final_metrics[key] = total / metric_counts[name]
            
            return final_metrics
            
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
