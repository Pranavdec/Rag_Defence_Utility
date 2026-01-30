"""
Main RAG Pipeline orchestrator.
Handles ingestion and batch inference with result logging.
"""
import os
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict
import yaml
import logging

from ..data_loaders.base_loader import BaseLoader, QAPair
from ..data_loaders.nq_loader import NQLoader
from ..data_loaders.pubmed_loader import PubMedLoader
from ..data_loaders.trivia_loader import TriviaLoader
from .retrieval import VectorStore
from .generation import create_generator
from ..defenses.manager import DefenseManager
from .persistence import UserTrustManager
from .sensing import MetricsCollector
from .ado import Sentinel, Strategist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_loader(name: str) -> BaseLoader:
    """Factory function to get loader by name."""
    loaders = {
        "nq": NQLoader,
        "pubmedqa": PubMedLoader,
        "triviaqa": TriviaLoader,
    }
    if name not in loaders:
        raise ValueError(f"Unknown loader: {name}. Available: {list(loaders.keys())}")
    return loaders[name]()


class ModularRAG:
    """
    Modular RAG Pipeline.
    
    Supports:
    - Ingestion from multiple data sources
    - Retrieval using ChromaDB with local embeddings
    - Generation using HuggingFace (default) or Ollama LLM
    - Result logging for evaluation
    """
    
    def __init__(self, config_path: str = "config/config.yaml", attack_module: Optional[Any] = None):
        self.config = load_config(config_path)
        self.attack = attack_module
        
        # Extract config values (with fallbacks for different config formats)
        self.top_k = self.config.get("retrieval", {}).get("top_k", self.config.get("experiment", {}).get("top_k_retrieval", 5))
        self.test_sample_size = self.config.get("data", {}).get("test_size", self.config.get("experiment", {}).get("test_sample_size", 10))
        self.sample_size = self.test_sample_size  # Alias for compatibility
        self.ingestion_limit = self.config.get("data", {}).get("ingestion_size", self.config.get("ingestion", {}).get("limit", 1000))
        
        self.chunk_size = self.config.get("retrieval", {}).get("chunk_size", self.config.get("ingestion", {}).get("chunk_size", 512))
        self.chunk_overlap = self.config.get("retrieval", {}).get("chunk_overlap", self.config.get("ingestion", {}).get("chunk_overlap", 50))
        
        self.chroma_path = self.config["paths"]["chroma_db"]
        self.results_path = self.config["paths"]["results"]
        
        # Get embedding model from config
        self.embedding_model = self.config.get("system", {}).get("embedding_model", "all-MiniLM-L6-v2")
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
        
        # Initialize Defense Manager FIRST (so it can be passed to generator)
        # Support both old 'defense' (dict) and new 'defenses' (list) config
        defense_config = self.config.get("defenses", [])
        if not defense_config and "defense" in self.config:
            # Migration path: wrap old single defense config in list
            old_conf = self.config["defense"]
            if old_conf and old_conf.get("method"):
                old_conf["name"] = "differential_privacy" # Map old generic defense to specific one
                defense_config = [old_conf]
        
        
        self.defense_manager = DefenseManager(defense_config)
        
        # Initialize ADO Components
        self.ado_enabled = self.config.get("ado", {}).get("enabled", False)
        if self.ado_enabled:
            logger.info("Initializing ADO Components...")
            self.trust_manager = UserTrustManager()
            self.metrics_collector = MetricsCollector()
            
            ado_config = self.config.get("ado", {})
            self.sentinel = Sentinel(
                model_name=ado_config.get("sentinel_model", "llama3"),
                use_ollama=True # Currently default
            )
            self.strategist = Strategist(config=ado_config)
            
            # Default test user for batch runs
            self.default_user_id = ado_config.get("user_id", "test_user_001")


        # Initialize generator (with potential model sharing from AV defense)
        self.generator = create_generator(self.config, defense_manager=self.defense_manager)
        
        # Vector store will be initialized per dataset
        self.vector_store: Optional[VectorStore] = None
        self.current_dataset: Optional[str] = None
        
        logger.info(f"ModularRAG initialized with config from {config_path}")
    
    def ingest(self, dataset_name: str, sample_size: Optional[int] = None) -> bool:
        """
        Ingest documents from a dataset into the vector store.
        Skips if already populated.
        
        Args:
            dataset_name: One of 'nq', 'pubmedqa', 'triviaqa'
            sample_size: Override config sample_size for documents
        """
        sample_size = sample_size or self.sample_size
        
        logger.info(f"Starting ingestion for {dataset_name}...")
        

        # Load loader to get correct name
        loader = get_loader(dataset_name)
        
        # Initialize vector store for this dataset using loader.name
        self.vector_store = VectorStore(
            collection_name=loader.name,
            persist_directory=self.chroma_path,
            embedding_model=self.embedding_model
        )
        self.current_dataset = dataset_name
        
        # Check if already populated
        if self.vector_store.is_populated():
            logger.info(f"Dataset {dataset_name} already ingested. Skipping.")
            return True

        # Load QA pairs (gold passages)
        qa_pairs = loader.load_qa_pairs(limit=sample_size)
        
        if not qa_pairs:
            logger.warning(f"No QA pairs loaded for {dataset_name}")
            return False
        
        # Chunk gold passages
        chunked_docs = []
        chunked_metas = []
        chunked_ids = []
        
        for qa in qa_pairs:
            for p_idx, passage in enumerate(qa.gold_passages):
                chunks = self._chunk_text(passage)
                
                for c_idx, chunk in enumerate(chunks):
                    chunked_docs.append(chunk)
                    chunked_metas.append({
                        "pair_id": qa.pair_id,
                        "question": qa.question[:200],
                        "passage_idx": p_idx,
                        "chunk_idx": c_idx
                    })
                    chunked_ids.append(f"{qa.pair_id}_p{p_idx}_c{c_idx}")
        
        logger.info(f"Chunked {len(qa_pairs)} QA pairs into {len(chunked_docs)} chunks")
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=chunked_docs,
            metadatas=chunked_metas,
            ids=chunked_ids
        )
        
        return True

    def ingest_with_attack(self, dataset_name: str, poison_ratio: float = 0.1, sample_size: Optional[int] = None):
        """
        Ingest legitimate documents and inject poisoned documents.
        """
        # Ingest legitimate documents first
        self.ingest(dataset_name, sample_size=sample_size)
        
        if self.attack and self.vector_store:
            logger.info("Injecting poisoned documents...")
            
            # Estimate dataset size (or use specific count if available)
            # using sample_size if provided, else use current count
            current_count = self.vector_store.collection.count()
            target_poison_count = int(current_count * poison_ratio)
            
            if target_poison_count == 0 and poison_ratio > 0:
                target_poison_count = 5 # Minimum fallback
            
            logger.info(f"Generating {target_poison_count} poisoned documents...")
            poisoned_docs = self.attack.generate_poisoned_corpus(target_size=target_poison_count)
            
            # Add to vector store with metadata marking them as poisoned
            self.vector_store.add_documents(
                documents=poisoned_docs,
                metadatas=[{"poisoned": True, "source": "attack_module"} for _ in poisoned_docs],
                ids=[f"poison_{i}" for i in range(len(poisoned_docs))]
            )
            logger.info("Poisoned documents injected.")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple text chunking with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def run_single(self, question: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run RAG on a single question.
        
        Returns:
            dict with 'question', 'answer', 'contexts', 'latency_ms'
        """
        if self.vector_store is None:
            raise RuntimeError("No dataset ingested. Call ingest() first.")
        
        # Determine User ID
        if user_id is None:
            user_id = getattr(self, "default_user_id", "anonymous")

        # --- ADO STAGE 1: SENSE & REASON ---
        ado_metadata = {}
        if self.ado_enabled:
            # 1. Get Trust Context from previous interactions
            user_ctx = self.trust_manager.get_user_context(user_id)
            trust_score = user_ctx.global_trust_score
            trust_history = getattr(user_ctx, 'trust_history', [])
            query_history = getattr(user_ctx, 'query_history', [])
            metrics_history = getattr(user_ctx, 'metrics_history', [])
            
            # 2. Calculate CURRENT PRE-retrieval metrics (from query text)
            # These can be calculated immediately without inference:
            # - M_LEX: Lexical overlap with previous queries
            # - M_CMP: Query complexity (special chars)
            # - M_INT: Intent velocity (time between queries)
            current_pre_metrics = self.metrics_collector.calculate_pre_retrieval(
                question, 
                history=query_history[-5:] if query_history else []
            )
            
            # 3. Get PREVIOUS POST-retrieval metrics (from last query's retrieval)
            # These can only be calculated after retrieval completes:
            # - M_DIS: Embedding dispersion
            # - M_DRP: Score drop-off
            prev_post_metrics = {}
            if metrics_history and len(metrics_history) > 0:
                prev_post_metrics = metrics_history[-1].get('post_retrieval', {})
            
            # Combine both metric sets
            combined_metrics = {
                **current_pre_metrics,      # Current query analysis
                **prev_post_metrics         # Previous retrieval patterns
            }
            logger.info(f"Metrics - Current Pre: {current_pre_metrics}, Previous Post: {prev_post_metrics}")
            
            # 4. Sentinel Analysis (uses current pre + previous post metrics)
            risk_profile = self.sentinel.analyze(
                query=question,
                trust_score=trust_score,        # From previous interactions
                metrics=combined_metrics,       # Current pre + previous post
                history_window=query_history[-5:] if query_history else [],
                trust_history=trust_history     # Trust trend over time
            )
            
            logger.info(f"ADO Risk Assessment: {risk_profile.overall_threat_level} | Trust: {trust_score:.2f} | Trend: {'DECLINING' if len(trust_history) >= 2 and trust_history[-1].get('delta', 0) < 0 else 'STABLE'}")

            # 4. Strategist Configuration
            defense_plan = self.strategist.generate_defense_plan(risk_profile)
            
            # Apply Defense Plan (Dynamic Configuration)
            self.defense_manager.set_dynamic_config(defense_plan)
            
            ado_metadata = {
                "risk_profile": asdict(risk_profile),
                "trust_score": trust_score,
                "defense_plan": defense_plan
            }
            
            # Update Trust Score (Persistence) - do this BEFORE inference
            self.trust_manager.update_trust_score(
                user_id, 
                risk_profile.new_global_score_delta, 
                reason=risk_profile.reasoning_trace
            )

        
        # Defense Pre-Retrieval
        query_text, fetch_k = self.defense_manager.apply_pre_retrieval(question, self.top_k)
        
        # Determine if embeddings are needed (for TrustRAG defense or ADO metrics)
        need_embeddings = self.defense_manager.needs_embeddings or self.ado_enabled
        
        # Retrieve
        retrieved = self.vector_store.query(query_text, top_k=fetch_k, include_embeddings=need_embeddings)

        # --- ADO STAGE 2: Calculate POST-retrieval metrics ---
        post_retrieval_metrics = {}
        if self.ado_enabled:
             # Calculate dispersion/drop-off from retrieval results
             # VectorStore returns 'distance' (cosine distance), convert to similarity score
             distances = [r.get("distance", 0.0) for r in retrieved if r.get("distance") is not None]
             scores = [1.0 - d for d in distances]  # Convert distance to similarity score
             embeddings = [r.get("embedding") for r in retrieved if r.get("embedding") is not None]
             post_retrieval_metrics = self.metrics_collector.calculate_retrieval(scores, embeddings)
             logger.info(f"POST-retrieval metrics: {post_retrieval_metrics}")
             ado_metadata["retrieval_metrics"] = post_retrieval_metrics

        
        # Defense Post-Retrieval
        retrieved = self.defense_manager.apply_post_retrieval(retrieved, question)
        
        contexts = [r["content"] for r in retrieved]
        
        # Defense Pre-Generation
        sys_p, user_p, mod_contexts = self.defense_manager.apply_pre_generation(
            system_prompt="", # Default empty
            user_prompt=question,
            contexts=contexts
        )
        
        # Generate
        result = self.generator.generate(
            question=user_p, 
            contexts=mod_contexts,
            system_prompt=sys_p if sys_p else None
        )
        
        # Defense Post-Generation
        result["answer"] = self.defense_manager.apply_post_generation(result["answer"])
        
        # --- ADO STAGE 3: Store metrics for next round ---
        if self.ado_enabled:
            # Store both pre and post retrieval metrics
            # Pre-metrics are already calculated (current_pre_metrics)
            # Post-metrics were calculated after retrieval (post_retrieval_metrics)
            combined_metrics_for_storage = {
                'pre_retrieval': current_pre_metrics,
                'post_retrieval': post_retrieval_metrics
            }
            self.trust_manager.update_query_history(user_id, question, combined_metrics_for_storage)
            logger.debug(f"Stored query with pre+post metrics for user {user_id}")
        
        return {
            "question": question,
            "answer": result["answer"],
            "contexts": contexts,
            "latency_ms": result["latency_ms"],
            "model": result["model"],
            "ado_metadata": ado_metadata
        }
    
    def run_batch(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run RAG on a batch of test cases.
        
        Args:
            dataset_name: Dataset to use for test set
            sample_size: Number of test cases (uses config default if None)
            save_results: Whether to save results to JSON
        
        Returns:
            List of result dicts
        """
        sample_size = sample_size or self.sample_size
        
        # Ensure dataset is ingested
        if self.current_dataset != dataset_name:
            self.ingest(dataset_name, sample_size=sample_size * 10)  # Ingest more docs than test cases
        
        # Load test QA pairs
        loader = get_loader(dataset_name)
        qa_pairs = loader.load_qa_pairs(limit=sample_size)
        
        logger.info(f"Running batch on {len(qa_pairs)} test cases...")
        
        results = []
        total_latency = 0
        
        for i, qa in enumerate(qa_pairs):
            logger.info(f"Processing {i+1}/{len(qa_pairs)}: {qa.question[:50]}...")
            
            result = self.run_single(qa.question)
            result["ground_truth"] = qa.answer
            result["metadata"] = qa.metadata
            
            results.append(result)
            total_latency += result["latency_ms"]
        
        # Summary stats
        avg_latency = total_latency / len(results) if results else 0
        logger.info(f"Batch complete. Avg latency: {avg_latency:.2f}ms")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.results_path,
                f"run_{dataset_name}_{timestamp}.json"
            )
            
            output_data = {
                "dataset": dataset_name,
                "timestamp": timestamp,
                "config": {
                    **self.config,  # Include full config.yaml content
                    "runtime_overrides": {
                        "test_sample_size": sample_size,
                        "timestamp": timestamp
                    }
                },
                "summary": {
                    "total_cases": len(results),
                    "avg_latency_ms": avg_latency
                },
                "results": results
            }
            
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        return results
