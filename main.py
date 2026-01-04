"""
RAG Defence Utility - Main CLI
Commands:
  - run <dataset>: Run retrieval + generation on test set (NO ingestion)
  - evaluate <results_file>: Evaluate saved results
"""
import argparse
import os
import sys
import json
import yaml
import logging
from datetime import datetime

# Suppress noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("ollama").setLevel(logging.ERROR)

from src.core.retrieval import VectorStore
from src.core.generation import OllamaGenerator
from src.defenses.manager import DefenseManager


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def cmd_run(args):
    """Run retrieval + generation on test set (assumes ingestion is done)."""
    config = load_config()
    dataset = args.dataset
    test_size = args.limit or config["data"]["test_size"]
    
    print("=" * 60)
    print(f"RAG RUN: {dataset.upper()}")
    print("=" * 60)
    
    # Load saved QA pairs
    qa_file = f"data/qa_pairs/{dataset}.json"
    if not os.path.exists(qa_file):
        print(f"✗ QA pairs not found: {qa_file}")
        print("  Run 'python scripts/ingest_data.py' first!")
        return
    
    with open(qa_file) as f:
        qa_pairs = json.load(f)
    
    # Random sample for testing
    import random
    test_seed = config["data"]["test_seed"]
    random.seed(test_seed)
    
    if len(qa_pairs) > test_size:
        qa_pairs = random.sample(qa_pairs, test_size)
    else:
        qa_pairs = qa_pairs[:test_size]
    
    print(f"Testing on {len(qa_pairs)} questions (random seed={test_seed})")
    
    # Get collection name from loader
    collection_names = {"nq": "nq-corpus", "pubmedqa": "pubmedqa", "triviaqa": "triviaqa"}
    collection_name = collection_names.get(dataset, dataset)
    
    # Initialize retrieval
    vs = VectorStore(
        collection_name=collection_name,
        persist_directory=config["paths"]["chroma_db"],
        embedding_model=config["system"]["embedding_model"]
    )
    
    if not vs.is_populated():
        print(f"✗ Collection '{collection_name}' is empty!")
        print("  Run 'python scripts/ingest_data.py' first!")
        return
    
    print(f"Using collection: {collection_name} ({vs.collection.count()} chunks)")
    
    # Initialize generator
    gen = OllamaGenerator(
        model_name=config["system"]["llm"]["model_name"],
        temperature=config["system"]["llm"]["temperature"]
    )

    # Initialize Defense Manager
    defense_config = config.get("defenses", [])
    if not defense_config and "defense" in config:
        old_conf = config["defense"]
        if old_conf and old_conf.get("method"):
            old_conf["name"] = "differential_privacy"
            defense_config = [old_conf]
    
    defense_manager = DefenseManager(defense_config)
    
    # Run retrieval + generation
    results = []
    top_k = config["retrieval"]["top_k"]
    
    print("\nProcessing questions...")
    for i, qa in enumerate(qa_pairs, 1):
        # Clean progress indicator
        print(f"\rProgress: {i}/{len(qa_pairs)} ({i*100//len(qa_pairs)}%)", end="", flush=True)
        
        # Defense Pre-Retrieval
        query_text, fetch_k = defense_manager.apply_pre_retrieval(qa["question"], top_k)

        # Retrieve
        retrieved = vs.query(
            query_text, 
            top_k=fetch_k
        )
        
        # Defense Post-Retrieval
        retrieved = defense_manager.apply_post_retrieval(retrieved, qa["question"])
        
        contexts = [r["content"] for r in retrieved]
        
        # Defense Pre-Generation
        sys_p, user_p, mod_contexts = defense_manager.apply_pre_generation(
            system_prompt="",
            user_prompt=qa["question"],
            contexts=contexts
        )
        
        # Generate
        gen_result = gen.generate(
            question=user_p, 
            contexts=mod_contexts,
            system_prompt=sys_p if sys_p else None
        )
        
        # Defense Post-Generation
        gen_result["answer"] = defense_manager.apply_post_generation(gen_result["answer"])
        
        results.append({
            "pair_id": qa["pair_id"],
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "generated_answer": gen_result["answer"],
            "contexts": contexts,
            "latency_ms": gen_result["latency_ms"],
            "metadata": qa.get("metadata", {})
        })
    print()  # New line after progress
    
    # Save results
    os.makedirs(config["paths"]["results"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{config['paths']['results']}/run_{dataset}_{timestamp}.json"
    
    # Calculate summary stats
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    
    output_data = {
        "dataset": dataset,
        "timestamp": timestamp,
        "config": {
            **config,  # Include full config.yaml content
            "runtime_params": {
                "test_size": len(results),
                "top_k": top_k,
                "model": config["system"]["llm"]["model_name"]
            }
        },
        "summary": {
            "avg_latency_ms": round(avg_latency, 2)
        },
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Questions tested: {len(results)}")
    print(f"Avg latency: {avg_latency:.2f} ms")
    print(f"Results saved: {output_file}")


def cmd_evaluate(args):
    """Evaluate saved results using RAGAS and custom metrics."""
    from src.evaluation.evaluator import RAGEvaluator
    
    print("=" * 60)
    print("EVALUATION (RAGAS + Custom Metrics)")
    print("=" * 60)
    
    if not os.path.exists(args.results_file):
        print(f"✗ File not found: {args.results_file}")
        return
    
    # Load config
    config = load_config()
    
    # Initialize evaluator with judge LLM
    judge_llm = config["system"].get("judge_llm", config["system"]["llm"]["model_name"])
    evaluator = RAGEvaluator(
        llm_model=f"ollama/{judge_llm}",
        embedding_model=config["system"]["embedding_model"]
    )
    
    # Get evaluation config
    deepeval_max_concurrent = config.get("evaluation", {}).get("deepeval_max_concurrent", 5)
    
    # Run evaluation
    print(f"\nEvaluating: {args.results_file}")
    result = evaluator.evaluate_all(
        args.results_file,
        use_ragas=args.use_ragas,
        use_deepeval=not args.no_deepeval,
        deepeval_max_concurrent=deepeval_max_concurrent,
        evaluation_config=config
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {result.dataset}")
    print(f"Samples: {result.num_samples}")
    print()
    
    # Group metrics
    latency_keys = [k for k in result.metrics if "latency" in k]
    refusal_keys = [k for k in result.metrics if "refusal" in k]
    ragas_keys = [k for k in result.metrics if k.startswith("ragas_")]
    other_keys = [k for k in result.metrics if k not in latency_keys + refusal_keys + ragas_keys]
    
    if latency_keys:
        print("📊 Latency:")
        for k in latency_keys:
            v = result.metrics[k]
            print(f"   {k}: {v:.2f}" if isinstance(v, float) else f"   {k}: {v}")
    
    if refusal_keys:
        print("\n🚫 Refusal Detection:")
        count = result.metrics.get("refusal_count", 0)
        total = result.metrics.get("total_samples", 0)
        rate = result.metrics.get("refusal_rate", 0)
        print(f"   Refusals: {count}/{total} ({rate:.1%})")
        if count == 0:
            print("   (Note: 0% is good! It means the model answered all benign questions)")
    
    if ragas_keys:
        print("\n📈 RAGAS Metrics:")
        if not any(result.metrics[k] for k in ragas_keys):
             print("   (No RAGAS metrics computed - possible local LLM timeouts)")
        for k in ragas_keys:
            v = result.metrics[k]
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    if other_keys:
        print("\n📋 Other:")
        for k in other_keys:
            v = result.metrics[k]
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    # Save evaluation results
    output_file = args.output
    if not output_file:
        # Auto-generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/metrics", exist_ok=True)
        output_file = f"data/metrics/eval_{result.dataset}_{timestamp}.json"
    
    evaluator.save_evaluation(result, output_file)
    print(f"\n✓ Evaluation saved to: {output_file}")
    print(f"✓ Metrics CSV saved to: {output_file.replace('.json', '_metrics.csv')}")


def main():
    parser = argparse.ArgumentParser(description="RAG Defence Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run retrieval + generation")
    run_parser.add_argument("dataset", choices=["nq", "pubmedqa", "triviaqa"])
    run_parser.add_argument("--limit", type=int, help="Override test_size")
    run_parser.set_defaults(func=cmd_run)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results with DeepEval/RAGAS")
    eval_parser.add_argument("results_file", help="Path to results JSON")
    eval_parser.add_argument("--use-ragas", action="store_true", help="Enable RAGAS eval")
    eval_parser.add_argument("--no-deepeval", action="store_true", help="Skip DeepEval eval")
    eval_parser.add_argument("-o", "--output", help="Save evaluation to file")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
