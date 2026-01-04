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

from ..data_loaders.base_loader import BaseLoader, Document, TestCase
from ..data_loaders.nq_loader import NQLoader
from ..data_loaders.pubmed_loader import PubMedLoader
from ..data_loaders.trivia_loader import TriviaLoader
from .retrieval import VectorStore
from .generation import create_generator
from ..defenses.manager import DefenseManager

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
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        
        # Extract config values
        self.top_k = self.config["experiment"]["top_k_retrieval"]
        self.test_sample_size = self.config["experiment"]["test_sample_size"]
        self.ingestion_limit = self.config["ingestion"].get("limit", 1000)
        
        self.chunk_size = self.config["ingestion"].get("chunk_size", 512)
        self.chunk_overlap = self.config["ingestion"].get("chunk_overlap", 50)
        
        self.chroma_path = self.config["paths"]["chroma_db"]
        self.results_path = self.config["paths"]["results"]
        
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

        # Load documents
        documents = loader.load_documents(sample_size=sample_size)
        
        if not documents:
            logger.warning(f"No documents loaded for {dataset_name}")
            return False
        
        # Chunk documents (simple splitting for now)
        chunked_docs = []
        chunked_metas = []
        chunked_ids = []
        
        for doc in documents:
            # Simple chunking by character count
            content = doc.content
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append(chunk)
                chunked_metas.append({**doc.metadata, "chunk_id": i})
                chunked_ids.append(f"{doc.doc_id}_chunk_{i}")
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=chunked_docs,
            metadatas=chunked_metas,
            ids=chunked_ids
        )
        
        return True
    
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
    
    def run_single(self, question: str) -> Dict[str, Any]:
        """
        Run RAG on a single question.
        
        Returns:
            dict with 'question', 'answer', 'contexts', 'latency_ms'
        """
        if self.vector_store is None:
            raise RuntimeError("No dataset ingested. Call ingest() first.")
        
        # Defense Pre-Retrieval
        query_text, fetch_k = self.defense_manager.apply_pre_retrieval(question, self.top_k)
        
        # Retrieve
        retrieved = self.vector_store.query(query_text, top_k=fetch_k)
        
        # Defense Post-Retrieval
        retrieved = self.defense_manager.apply_post_retrieval(retrieved, question)
        
        contexts = [r["content"] for r in retrieved]
        
        # Defense Pre-Generation
        # Default system prompt is None in generator, but we can pass one if defense wants to inject
        system_prompt = None 
        # But wait, generator.generate takes system_prompt.
        # We need to construct user_prompt too? 
        # Actually generator.generate constructs user prompt from context+question.
        # Defense might want to modify the QUESTION that goes into prompt? or the CONTEXTS?
        
        # My BaseDefense.pre_generation signature is (system_prompt, user_prompt, contexts)
        # But generator builds the prompt internally. 
        # I should probably update generator to accept inputs more flexibly, OR
        # I can just pass modified contexts and question?
        # The user wanted "post-gen defense".
        # Let's see what base.py says: pre_generation(system_prompt, user_prompt, contexts)
        # Here I have `question` and `contexts`.
        # I can pass `question` as `user_prompt` effectively for now.
        
        sys_p, user_p, mod_contexts = self.defense_manager.apply_pre_generation(
            system_prompt="", # Default empty
            user_prompt=question,
            contexts=contexts
        )
        
        # Generator expects question. If user_p changed, we treat it as new question.
        # If sys_p changed, we pass it.
        
        # Generate
        result = self.generator.generate(
            question=user_p, 
            contexts=mod_contexts,
            system_prompt=sys_p if sys_p else None
        )
        
        # Defense Post-Generation
        result["answer"] = self.defense_manager.apply_post_generation(result["answer"])
        
        return {
            "question": question,
            "answer": result["answer"],
            "contexts": contexts,
            "latency_ms": result["latency_ms"],
            "model": result["model"]
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
        
        # Load test set
        loader = get_loader(dataset_name)
        test_cases = loader.load_test_set(sample_size=sample_size)
        
        logger.info(f"Running batch on {len(test_cases)} test cases...")
        
        results = []
        total_latency = 0
        
        for i, tc in enumerate(test_cases):
            logger.info(f"Processing {i+1}/{len(test_cases)}: {tc.question[:50]}...")
            
            result = self.run_single(tc.question)
            result["ground_truth"] = tc.ground_truth
            result["metadata"] = tc.metadata
            
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
