"""
Config-Driven Comprehensive RAG Evaluation Script

This script runs the full evaluation pipeline based SOLELY on config.yaml.
No CLI arguments are needed - everything is driven by the config file.

Workflow:
1. Load config/config.yaml
2. Create timestamped results folder and copy config
3. Clear Vector DB for clean state
4. Ingest data based on config
5. Run evaluation (ADO-enabled or ADO-disabled path)
6. Save results

Usage:
    python scripts/comprehensive_eval.py
"""

import os
import sys
import json
import random
import shutil
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.core.pipeline import ModularRAG, get_loader, load_config
from src.core.retrieval import VectorStore
from src.attacks.poisoned_rag import PoisonedRAGFramework
from src.attacks.mba import MBAFramework

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryResult:
    """Result of a single query."""
    query: str
    query_type: str  # 'benign', 'mba', 'poisoning'
    answer: str
    expected_answer: Optional[str]
    success: bool
    latency_ms: float
    ado_metadata: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Utility metrics
    utility_total: int = 0
    
    # DeepEval metrics
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    contextual_relevancy: float = 0.0
    contextual_recall: float = 0.0
    
    # Attack metrics
    poisoning_total: int = 0
    poisoning_success: int = 0
    poisoning_asr: float = 0.0
    
    mba_total: int = 0
    mba_accuracy: float = 0.0
    mba_precision: float = 0.0
    mba_recall: float = 0.0
    mba_f1: float = 0.0
    mba_avg_mask_accuracy: float = 0.0  # Average accuracy of filled masks across all MBA queries
    mba_member_accuracy: float = 0.0  # Accuracy on member documents
    mba_non_member_accuracy: float = 0.0  # Accuracy on non-member documents
    
    # Performance
    avg_latency_ms: float = 0.0
    total_queries: int = 0
    
    # ADO metrics
    ado_enabled: bool = False
    avg_trust_score: float = 0.0
    defenses_triggered: Dict[str, int] = field(default_factory=dict)
    risk_level_counts: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================

class ConfigDrivenEvaluator:
    """
    Comprehensive evaluator driven entirely by config.yaml.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Create timestamped results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(
            self.config["paths"]["results"],
            f"{timestamp}_eval"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Copy config to results folder
        config_copy_path = os.path.join(self.results_dir, "config.yaml")
        shutil.copy(config_path, config_copy_path)
        logger.info(f"Config copied to: {config_copy_path}")
        
        # Initialize DeepEval evaluator
        self.deepeval_evaluator = None
        try:
            from src.evaluation.evaluator import RAGEvaluator
            judge_llm = self.config.get("system", {}).get("judge_llm", "llama3")
            self.deepeval_evaluator = RAGEvaluator(llm_model=f"ollama/{judge_llm}")
            logger.info("DeepEval evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not initialize DeepEval: {e}")
        
        self.rag: Optional[ModularRAG] = None
        
    def clear_vector_db(self):
        """Clear the vector database for a clean start."""
        chroma_path = self.config["paths"]["chroma_db"]
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            logger.info(f"Cleared Vector DB at: {chroma_path}")
        else:
            logger.info(f"Vector DB path does not exist, nothing to clear: {chroma_path}")
    
    def clear_user_data(self):
        """Clear user session data for ADO (for clean trust scores)."""
        user_data_path = "data/users"
        if os.path.exists(user_data_path):
            shutil.rmtree(user_data_path)
            os.makedirs(user_data_path, exist_ok=True)
            logger.info(f"Cleared user session data at: {user_data_path}")
    
    def setup_rag(self) -> ModularRAG:
        """Initialize and return the RAG pipeline."""
        self.rag = ModularRAG(config_path=self.config_path)
        return self.rag
    
    def ingest_data(self):
        """Ingest data based on config settings."""
        if self.rag is None:
            self.setup_rag()
        
        logger.info("Starting data ingestion...")
        success = self.rag.ingest()
        if success:
            logger.info("Data ingestion complete.")
        else:
            logger.error("Data ingestion failed!")
        return success
    
    def load_test_qa_pairs(self) -> List:
        """Load QA pairs for testing based on config."""
        dataset_name = self.config["data"]["dataset"]
        test_size = self.config["data"]["test_size"]
        test_seed = self.config["data"].get("test_seed", 123)
        
        # Use ingestion seed/size to get the same data pool that was ingested
        ingestion_size = self.config["data"]["ingestion_size"]
        ingestion_seed = self.config["data"]["ingestion_seed"]
        
        loader = get_loader(dataset_name)
        all_pairs = loader.load_qa_pairs(limit=ingestion_size, seed=ingestion_seed)
        
        # Sample test pairs using test_seed
        random.seed(test_seed)
        if len(all_pairs) > test_size:
            test_pairs = random.sample(all_pairs, test_size)
        else:
            test_pairs = all_pairs
        
        logger.info(f"Loaded {len(test_pairs)} test QA pairs for evaluation")
        return test_pairs
    
    def run_query(self, query: str, user_id: str = "eval_user") -> Dict:
        """Run a single query and return result."""
        start = time.time()
        result = self.rag.run_single(query, user_id=user_id)
        latency = (time.time() - start) * 1000
        result['latency_ms'] = latency
        return result
    
    # =========================================================================
    # ADO DISABLED PATH (Sequential Evaluation)
    # =========================================================================
    
    def run_sequential_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation when ADO is disabled.
        Runs benign queries first, then attacks sequentially.
        """
        logger.info("=" * 70)
        logger.info("RUNNING SEQUENTIAL EVALUATION (ADO DISABLED)")
        logger.info("=" * 70)
        
        all_results: List[QueryResult] = []
        metrics = EvaluationMetrics()
        metrics.ado_enabled = False
        
        # 1. Benign Queries
        logger.info("\n--- PHASE 1: Benign Queries ---")
        test_pairs = self.load_test_qa_pairs()
        benign_results = self._run_benign_queries(test_pairs)
        all_results.extend(benign_results)
        
        # 2. Poisoning Attack (if enabled)
        poison_config = self.config.get("attack", {}).get("poisoned_rag", {})
        if poison_config.get("enabled", False):
            logger.info("\n--- PHASE 2: Poisoning Attack ---")
            poison_results = self._run_poisoning_attack()
            all_results.extend(poison_results)
        
        # 3. MBA Attack (if enabled)
        mba_config = self.config.get("attack", {}).get("mba", {})
        if mba_config.get("enabled", False):
            logger.info("\n--- PHASE 3: MBA Attack ---")
            mba_results, mba_metrics = self._run_mba_attack()
            all_results.extend(mba_results)
            metrics.mba_accuracy = mba_metrics.get("accuracy", 0.0)
            metrics.mba_precision = mba_metrics.get("precision", 0.0)
            metrics.mba_recall = mba_metrics.get("recall", 0.0)
            metrics.mba_f1 = mba_metrics.get("f1", 0.0)
            metrics.mba_avg_mask_accuracy = mba_metrics.get("avg_mask_accuracy", 0.0)
            metrics.mba_member_accuracy = mba_metrics.get("member_accuracy", 0.0)
            metrics.mba_non_member_accuracy = mba_metrics.get("non_member_accuracy", 0.0)
        
        # 4. Aggregate and compute DeepEval
        metrics = self._aggregate_metrics(all_results, metrics)
        metrics = self._compute_deepeval_metrics(all_results, metrics)
        
        # 5. Save results
        self._save_results(all_results, metrics, "sequential")
        
        return {"metrics": asdict(metrics), "results": [asdict(r) for r in all_results]}
    
    # =========================================================================
    # ADO ENABLED PATH (Mixed Traffic Evaluation)
    # =========================================================================
    
    def run_mixed_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation when ADO is enabled.
        Shuffles benign and attack queries together for realistic traffic.
        """
        logger.info("=" * 70)
        logger.info("RUNNING MIXED TRAFFIC EVALUATION (ADO ENABLED)")
        logger.info("=" * 70)
        
        # Clear user data for clean ADO state
        self.clear_user_data()
        
        all_results: List[QueryResult] = []
        metrics = EvaluationMetrics()
        metrics.ado_enabled = True
        
        # Build query schedule
        query_schedule = []
        
        # 1. Add Benign Queries
        test_pairs = self.load_test_qa_pairs()
        for qa in test_pairs:
            query_schedule.append({
                'type': 'benign',
                'query': qa.question,
                'expected': qa.answer if qa.answer else None,
                'metadata': {}
            })
        
        # 2. Add Poisoning Queries (if enabled)
        poison_config = self.config.get("attack", {}).get("poisoned_rag", {})
        if poison_config.get("enabled", False):
            # Generate poison payloads
            poison_framework = PoisonedRAGFramework(self.config)
            payloads = poison_framework.generate_poisoned_payloads()
            
            # Inject poisoned documents
            poisoned_docs = payloads.get('poisoned_documents', [])
            if poisoned_docs and self.rag.vector_store:
                self._inject_poison_docs(poisoned_docs)
            
            # Add poison eval queries
            for eval_pair in payloads.get('eval_pairs', []):
                query_schedule.append({
                    'type': 'poisoning',
                    'query': eval_pair['question'],
                    'expected': eval_pair['target_answer'],
                    'metadata': {'ground_truth': eval_pair.get('ground_truth')}
                })
        
        # 3. Add MBA Queries (if enabled)
        mba_config = self.config.get("attack", {}).get("mba", {})
        mba_payloads = []
        if mba_config.get("enabled", False):
            mba_framework = MBAFramework(self.config)
            mba_payloads = mba_framework.generate_attack_dataset()
            
            for payload in mba_payloads:
                query_schedule.append({
                    'type': 'mba',
                    'query': payload['query'],
                    'expected': payload['ground_truth'],
                    'metadata': {
                        'id': payload['id'],
                        'is_member': payload['is_member'],
                        'original_document': payload.get('original_document', '')
                    }
                })
        
        # 4. Shuffle the schedule
        random.seed(int(time.time()))  # Use current time for true randomness
        random.shuffle(query_schedule)
        
        logger.info(f"\nTotal queries in mixed schedule: {len(query_schedule)}")
        logger.info(f"  - Benign: {sum(1 for q in query_schedule if q['type'] == 'benign')}")
        logger.info(f"  - Poisoning: {sum(1 for q in query_schedule if q['type'] == 'poisoning')}")
        logger.info(f"  - MBA: {sum(1 for q in query_schedule if q['type'] == 'mba')}")
        
        # 5. Execute queries
        mba_responses = []
        for i, item in enumerate(query_schedule):
            print(f"\r  Query {i+1}/{len(query_schedule)} [{item['type'].upper()}]", end="", flush=True)
            
            res = self.run_query(item['query'], user_id="mixed_eval_user")
            answer = res.get("answer", "")
            
            # Determine success based on query type
            if item['type'] == 'benign':
                success = bool(answer and len(answer.strip()) > 0)
            elif item['type'] == 'poisoning':
                target = item['expected']
                success = target.lower() in answer.lower() if target else False
            else:  # mba
                success = False  # Will be evaluated later
                mba_responses.append(answer)
            
            all_results.append(QueryResult(
                query=item['query'],
                query_type=item['type'],
                answer=answer,
                expected_answer=item['expected'] if isinstance(item['expected'], str) else str(item['expected']),
                success=success,
                latency_ms=res.get('latency_ms', 0),
                ado_metadata=res.get('ado_metadata', {}),
                extra=item['metadata']
            ))
        
        print()  # Newline after progress
        
        # 6. Evaluate MBA results
        if mba_config.get("enabled", False) and mba_payloads:
            mba_framework = MBAFramework(self.config)
            mba_eval = mba_framework.evaluate_attack_results(mba_payloads, mba_responses)
            metrics.mba_accuracy = mba_eval.get("accuracy", 0.0)
            metrics.mba_precision = mba_eval.get("precision", 0.0)
            metrics.mba_recall = mba_eval.get("recall", 0.0)
            metrics.mba_f1 = mba_eval.get("f1", 0.0)
            metrics.mba_avg_mask_accuracy = mba_eval.get("avg_mask_accuracy", 0.0)
        
        # 7. Aggregate and compute DeepEval
        metrics = self._aggregate_metrics(all_results, metrics)
        metrics = self._compute_deepeval_metrics(all_results, metrics)
        
        # 8. Save results
        self._save_results(all_results, metrics, "mixed")
        
        return {"metrics": asdict(metrics), "results": [asdict(r) for r in all_results]}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _run_benign_queries(self, qa_pairs: List) -> List[QueryResult]:
        """Run benign queries and return results."""
        results = []
        for i, qa in enumerate(qa_pairs):
            print(f"\r  Benign {i+1}/{len(qa_pairs)}", end="", flush=True)
            
            res = self.run_query(qa.question)
            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
            
            results.append(QueryResult(
                query=qa.question,
                query_type="benign",
                answer=answer,
                expected_answer=qa.answer if qa.answer else None,
                success=bool(answer and len(answer.strip()) > 0),
                latency_ms=res.get('latency_ms', 0),
                ado_metadata=res.get('ado_metadata', {}),
                extra={"contexts": contexts}
            ))
        print()
        return results
    
    def _should_run_deepeval(self) -> bool:
        """Check if DeepEval should be run."""
        return not self.config.get('evaluation', {}).get('skip_deepeval', False)
    
    def _run_poisoning_attack(self) -> List[QueryResult]:
        """Run poisoning attack and return results."""
        results = []
        
        # Generate payloads
        poison_framework = PoisonedRAGFramework(self.config)
        payloads = poison_framework.generate_poisoned_payloads()
        
        # Inject poison
        poisoned_docs = payloads.get('poisoned_documents', [])
        if poisoned_docs and self.rag.vector_store:
            self._inject_poison_docs(poisoned_docs)
        
        # Run queries
        eval_pairs = payloads.get('eval_pairs', [])
        for i, pair in enumerate(eval_pairs):
            print(f"\r  Poison {i+1}/{len(eval_pairs)}", end="", flush=True)
            
            res = self.run_query(pair['question'])
            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
            target = pair['target_answer']
            
            # Check if poison was retrieved
            poison_retrieved = any('poison' in str(ctx).lower() for ctx in contexts)
            
            success = target.lower() in answer.lower() if target else False
            
            # Debug logging
            if not success:
                logger.info(f"\n[POISON FAILURE] Q: {pair['question'][:100]}")
                logger.info(f"  Target: '{target}'")
                logger.info(f"  Answer: '{answer[:200]}'")
                logger.info(f"  Poison retrieved: {poison_retrieved}")
                if contexts:
                    logger.info(f"  First context snippet: {contexts[0][:150] if contexts[0] else 'empty'}...")
            
            results.append(QueryResult(
                query=pair['question'],
                query_type="poisoning",
                answer=answer,
                expected_answer=target,
                success=success,
                latency_ms=res.get('latency_ms', 0),
                ado_metadata=res.get('ado_metadata', {})
            ))
        print()
        
        # Clean poison after evaluation
        self._clean_poison_docs()
        
        return results
    
    def _run_mba_attack(self) -> Tuple[List[QueryResult], Dict]:
        """Run MBA attack and return results with metrics."""
        results = []
        
        mba_framework = MBAFramework(self.config)
        payloads = mba_framework.generate_attack_dataset()
        
        responses = []
        for i, payload in enumerate(payloads):
            print(f"\r  MBA {i+1}/{len(payloads)}", end="", flush=True)
            
            # BLACK BOX: Use same query for both retrieval and generation
            res = self.run_query(payload['query'], user_id="mba_attacker")
            answer = res.get("answer", "")
            responses.append(answer)
            
            # Debug first few - CHECK RETRIEVAL
            if i < 15:
                logger.info(f"\n[MBA SAMPLE {i+1}] Is Member: {payload['is_member']}")
                logger.info(f"  Original Doc (first 150 chars): {payload['original_document'][:150]}...")
                logger.info(f"  Ground Truth: {payload['ground_truth']}")
                logger.info(f"  Response: {answer[:300]}")
                
                # Check what was retrieved
                retrieved = self.rag.vector_store.query(payload['query'][:500], top_k=self.config['retrieval']['top_k'])
                if retrieved:
                    logger.info(f"  Retrieved {len(retrieved)} docs:")
                    for idx, doc in enumerate(retrieved[:2]):  # Show top 2
                        doc_snippet = doc['document'][:100] if 'document' in doc else doc.get('content', '')[:100]
                        logger.info(f"    Doc {idx+1}: {doc_snippet}...")
                        # Check if original doc is in retrieval
                        if payload['original_document'][:100] in doc.get('document', doc.get('content', '')):
                            logger.info(f"    ✓ ORIGINAL DOC RETRIEVED at position {idx+1}")
                else:
                    logger.info(f"  ✗ No documents retrieved!")
            
            results.append(QueryResult(
                query=payload['query'][:100] + "...",
                query_type="mba",
                answer=answer[:200] + "..." if len(answer) > 200 else answer,
                expected_answer=str(payload['ground_truth']),
                success=False,  # Evaluated separately
                latency_ms=res.get('latency_ms', 0),
                ado_metadata=res.get('ado_metadata', {}),
                extra={'id': payload['id'], 'is_member': payload['is_member']}
            ))
        print()
        
        # Evaluate
        mba_metrics = mba_framework.evaluate_attack_results(payloads, responses)
        
        return results, mba_metrics
    
    def _inject_poison_docs(self, poisoned_docs: List):
        """Inject poisoned documents into vector store."""
        if not self.rag or not self.rag.vector_store:
            logger.warning("RAG or vector store not initialized, cannot inject poison")
            return
        
        # Check before injection
        before_count = self.rag.vector_store.collection.count()
        logger.info(f"Before injection: {before_count} documents in vector store")
        
        docs = [pd.content for pd in poisoned_docs]
        ids = [pd.doc_id for pd in poisoned_docs]
        metas = [{"poisoned": True, "target": pd.target_question} for pd in poisoned_docs]
        
        logger.info(f"Injecting {len(docs)} poisoned documents with IDs like: {ids[0]}")
        
        self.rag.vector_store.add_documents(
            documents=docs,
            metadatas=metas,
            ids=ids,
            force=True  # Bypass is_populated() check for poison injection
        )
        
        # Check after injection
        after_count = self.rag.vector_store.collection.count()
        logger.info(f"After injection: {after_count} documents in vector store (added {after_count - before_count})")
        
        # Verify poison docs exist
        result = self.rag.vector_store.collection.get(where={"poisoned": True}, limit=1)
        if result and result['documents']:
            logger.info(f"✓ Verified: Poison docs exist in vector store")
        else:
            logger.error(f"✗ ERROR: Poison docs NOT FOUND after injection!")
        
        logger.info(f"Injected {len(docs)} poisoned documents")
    
    def _clean_poison_docs(self):
        """Remove poisoned documents from vector store."""
        if not self.rag or not self.rag.vector_store:
            return
        
        try:
            collection = self.rag.vector_store.collection
            all_data = collection.get()
            poison_ids = [id for id in all_data['ids'] if id.startswith('poison_')]
            if poison_ids:
                collection.delete(ids=poison_ids)
                logger.info(f"Cleaned {len(poison_ids)} poisoned documents")
        except Exception as e:
            logger.warning(f"Could not clean poison: {e}")
    
    def _aggregate_metrics(self, results: List[QueryResult], metrics: EvaluationMetrics) -> EvaluationMetrics:
        """Aggregate results into metrics."""
        metrics.total_queries = len(results)
        
        total_latency = 0.0
        trust_scores = []
        
        for r in results:
            total_latency += r.latency_ms
            
            # ADO metrics
            if r.ado_metadata:
                ts = r.ado_metadata.get("trust_score")
                if ts is not None:
                    trust_scores.append(ts)
                
                risk = r.ado_metadata.get("risk_profile", {}).get("overall_threat_level", "UNKNOWN")
                metrics.risk_level_counts[risk] = metrics.risk_level_counts.get(risk, 0) + 1
                
                plan = r.ado_metadata.get("defense_plan", {})
                for defense_name in ["differential_privacy", "trustrag", "attention_filtering"]:
                    if plan.get(defense_name, {}).get("enabled", False):
                        metrics.defenses_triggered[defense_name] = metrics.defenses_triggered.get(defense_name, 0) + 1
            
            # Count by type
            if r.query_type == "benign":
                metrics.utility_total += 1
            elif r.query_type == "poisoning":
                metrics.poisoning_total += 1
                if r.success:
                    metrics.poisoning_success += 1
            elif r.query_type == "mba":
                metrics.mba_total += 1
        
        # Calculate rates
        if metrics.poisoning_total > 0:
            metrics.poisoning_asr = metrics.poisoning_success / metrics.poisoning_total
        if metrics.total_queries > 0:
            metrics.avg_latency_ms = total_latency / metrics.total_queries
        if trust_scores:
            metrics.avg_trust_score = sum(trust_scores) / len(trust_scores)
        
        return metrics
    
    def _compute_deepeval_metrics(self, results: List[QueryResult], metrics: EvaluationMetrics) -> EvaluationMetrics:
        """Compute DeepEval metrics for benign queries."""
        if not self.deepeval_evaluator:
            logger.warning("DeepEval not available, skipping quality metrics")
            return metrics
        
        benign_results = [r for r in results if r.query_type == "benign"]
        if not benign_results:
            logger.warning("No benign queries to evaluate with DeepEval")
            return metrics
        
        eval_data = []
        for r in benign_results:
            eval_data.append({
                "question": r.query,
                "answer": r.answer,
                "generated_answer": r.answer,
                "ground_truth": r.expected_answer or "",
                "contexts": r.extra.get("contexts", [])
            })
        
        if self._should_run_deepeval():
            logger.info(f"Running DeepEval on {len(eval_data)} benign queries...")
            
            try:
                max_concurrent = self.config.get("evaluation", {}).get("deepeval_max_concurrent", 5)
                deepeval_results = self.deepeval_evaluator.evaluate_with_deepeval(
                    eval_data, max_concurrent=max_concurrent
                )
                
                metrics.answer_relevancy = deepeval_results.get("deepeval_answer_relevancy", 
                                            deepeval_results.get("deepeval_answerrelevancymetric", 0.0))
                metrics.faithfulness = deepeval_results.get("deepeval_faithfulness",
                                        deepeval_results.get("deepeval_faithfulnessmetric", 0.0))
                metrics.contextual_relevancy = deepeval_results.get("deepeval_contextual_relevancy",
                                                deepeval_results.get("deepeval_contextualrelevancymetric", 0.0))
                metrics.contextual_recall = deepeval_results.get("deepeval_contextual_recall",
                                             deepeval_results.get("deepeval_contextualrecallmetric", 0.0))
                
                logger.info(f"DeepEval: AR={metrics.answer_relevancy:.3f}, F={metrics.faithfulness:.3f}, "
                           f"CR={metrics.contextual_relevancy:.3f}, CRec={metrics.contextual_recall:.3f}")
            except Exception as e:
                logger.error(f"DeepEval evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("DeepEval evaluation SKIPPED (skip_deepeval=true)")
        
        return metrics
    
    def _save_results(self, results: List[QueryResult], metrics: EvaluationMetrics, mode: str):
        """Save results to the results folder."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "config": self.config,
            "metrics": asdict(metrics),
            "results": [asdict(r) for r in results]
        }
        
        output_path = os.path.join(self.results_dir, f"evaluation_{mode}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: EvaluationMetrics):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print(" EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\n  ADO Enabled: {metrics.ado_enabled}")
        print(f"  Total Queries: {metrics.total_queries}")
        print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms")
        
        print(f"\n  --- UTILITY METRICS (DeepEval) ---")
        if metrics.answer_relevancy > 0 or metrics.faithfulness > 0:
            print(f"  Answer Relevancy:     {metrics.answer_relevancy:.3f}")
            print(f"  Faithfulness:         {metrics.faithfulness:.3f}")
            print(f"  Contextual Relevancy: {metrics.contextual_relevancy:.3f}")
            print(f"  Contextual Recall:    {metrics.contextual_recall:.3f}")
        else:
            print(f"  (No DeepEval metrics computed)")
        
        print(f"\n  --- ATTACK METRICS ---")
        if metrics.poisoning_total > 0:
            print(f"  Poisoning ASR: {metrics.poisoning_asr:.1%} ({metrics.poisoning_success}/{metrics.poisoning_total})")
        if metrics.mba_total > 0:
            print(f"  MBA Accuracy: {metrics.mba_accuracy:.1%}")
            print(f"  MBA F1: {metrics.mba_f1:.3f}")
            print(f"  MBA Avg Mask Accuracy: {metrics.mba_avg_mask_accuracy:.1%}")
            if hasattr(metrics, 'mba_member_accuracy'):
                print(f"  MBA Member Mask Accuracy: {metrics.mba_member_accuracy:.1%} (avg % masks correct)")
            if hasattr(metrics, 'mba_non_member_accuracy'):
                print(f"  MBA Non-Member Mask Accuracy: {metrics.mba_non_member_accuracy:.1%} (avg % masks correct)")
        
        if metrics.ado_enabled:
            print(f"\n  --- ADO METRICS ---")
            print(f"  Avg Trust Score: {metrics.avg_trust_score:.3f}")
            print(f"  Risk Levels: {metrics.risk_level_counts}")
            print(f"  Defenses Triggered: {metrics.defenses_triggered}")
        
        print("=" * 70)
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    def run(self):
        """Main execution method."""
        logger.info("Starting Config-Driven Comprehensive Evaluation")
        logger.info(f"Config: {self.config_path}")
        
        # Step 1: Clear Vector DB
        self.clear_vector_db()
        
        # Step 2: Setup RAG and Ingest
        self.setup_rag()
        if not self.ingest_data():
            logger.error("Ingestion failed, aborting evaluation.")
            return None
        
        # Step 3: Run evaluation based on ADO setting
        ado_enabled = self.config.get("ado", {}).get("enabled", False)
        
        if ado_enabled:
            return self.run_mixed_evaluation()
        else:
            return self.run_sequential_evaluation()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point."""
    evaluator = ConfigDrivenEvaluator()
    results = evaluator.run()
    
    if results:
        logger.info("Evaluation complete!")
    else:
        logger.error("Evaluation failed!")


if __name__ == "__main__":
    main()
