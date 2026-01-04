"""
Retrieval module for ChromaDB vector store with local embeddings.
Uses sentence-transformers for fast batch embedding (no Ollama overhead).
"""
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def sample_dp_threshold_pure(scores: np.ndarray, k: int, 
                             epsilon: float, min_score: float = -1.0, 
                             max_score: float = 1.0) -> float:
    """
    Pure differential privacy using exponential mechanism.
    Returns a threshold score sampled with probability proportional to exp(ε·utility/2).
    """
    # Clip and pad score range
    sorted_scores = np.sort(scores)
    sorted_scores = np.insert(sorted_scores, 0, min_score)
    sorted_scores = np.insert(sorted_scores, len(sorted_scores), max_score)
    
    # Utility: negative distance from selecting exactly k documents
    utilities = -np.abs(len(sorted_scores) - k - np.arange(len(sorted_scores)))
    
    # Exponential mechanism probabilities weighted by interval widths
    pdf = np.exp(epsilon * utilities[:-1] / 2) * np.diff(sorted_scores)
    
    # Normalize PDF
    if np.sum(pdf) == 0:
         return min_score # Fallback if probabilities are zero (should vary rarely happen with valid eps)
    pdf /= np.sum(pdf)
    
    return np.random.choice(sorted_scores[:-1], p=pdf)


def sample_dp_threshold_approximate(scores: np.ndarray, k: int,
                                    epsilon: float, delta: float,
                                    min_score: float = -1.0) -> float:
    """
    Approximate differential privacy using Gaussian mechanism.
    Adds calibrated Gaussian noise to the k-th score.
    """
    if delta <= 0:
        raise ValueError("Delta must be > 0 for approximate DP")
    
    # Sensitivity of top-k selection is 1 (adding/removing one document)
    sensitivity = 1.0
    
    # Gaussian noise scale for (ε,δ)-DP
    noise_scale = (2 * sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    
    # Get k-th highest score (or min_score if fewer than k documents)
    sorted_scores = np.sort(scores)
    kth_score = sorted_scores[-k] if k <= len(scores) else min_score
    
    # Add Gaussian noise
    noisy_threshold = kth_score + np.random.normal(0, noise_scale)
    
    return noisy_threshold


class LocalEmbedder:
    """Fast local embeddings using sentence-transformers (supports batching)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a list of texts using batch processing (FAST!)."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.model.encode(text, show_progress_bar=False).tolist()


class VectorStore:
    """ChromaDB-based vector store with persistence."""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "nomic-embed-text"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = LocalEmbedder(model_name=embedding_model)
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"VectorStore initialized: {collection_name} at {persist_directory}")
    
    def is_populated(self, min_docs: int = 10) -> bool:
        """Check if the collection already has documents (skip re-ingestion)."""
        count = self.collection.count()
        logger.info(f"Collection '{self.collection_name}' has {count} documents.")
        return count >= min_docs
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        Add documents to the vector store in batches.
        Skips if already populated.
        """
        if self.is_populated():
            logger.info(f"Collection already populated, skipping ingestion.")
            return
        
        total = len(documents)
        logger.info(f"Adding {total} documents in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else [f"doc_{j}" for j in range(i, i + len(batch_docs))]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None
            
            # Generate embeddings
            embeddings = self.embedder.embed(batch_docs)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
        
        logger.info(f"Ingestion complete. Total documents: {self.collection.count()}")
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        defense_config: Optional[dict] = None
    ) -> List[dict]:
        """
        Query the vector store for similar documents.
        Returns list of dicts with 'content', 'metadata', 'distance'.
        
        Args:
            defense_config: Dictionary with keys 'method', 'epsilon', 'delta', 'candidate_multiplier'.
        """
        query_embedding = self.embedder.embed_single(query_text)
        
        # Determine candidate retrieval size
        fetch_k = top_k
        if defense_config and defense_config.get("method"):
            multiplier = defense_config.get("candidate_multiplier", 3)
            fetch_k = top_k * multiplier
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []
        
        # Apply defense mechanism if active
        if defense_config and defense_config.get("method") and docs:
            method = defense_config["method"]
            epsilon = defense_config["epsilon"]
            
            # Convert distances to similarity scores (Cosine Similarity = 1 - Cosine Distance)
            scores = 1.0 - np.array(dists)
            
            if method == "dp_pure":
                threshold = sample_dp_threshold_pure(
                    scores, top_k, epsilon, min_score=-1.0, max_score=1.0
                )
            elif method == "dp_approx":
                delta = defense_config.get("delta", 0.01)
                threshold = sample_dp_threshold_approximate(
                    scores, top_k, epsilon, delta, min_score=-1.0
                )
            else:
                 logger.warning(f"Unknown defense method: {method}. Proceeding without filtering.")
                 threshold = -float('inf')

            # Filter documents above threshold
            filtered_indices = [i for i, s in enumerate(scores) if s > threshold]
            
            # If nothing passes, fallback (e.g., return top 1 or none)
            # Here we return empty if nothing passes, or maybe we should return at least 1?
            # Let's stick to the user's logic: return all above threshold
            
            # Reconstruct lists based on filtered indices
            # But we also need to respect the original top_k as a cap?
            # User said: "filter top k". So we return up to top_k from the filtered set?
            # "Return all documents above threshold (up to max_retrieve)"
            
            final_indices = filtered_indices[:top_k]
            
            docs = [docs[i] for i in final_indices]
            metas = [metas[i] for i in final_indices]
            dists = [dists[i] for i in final_indices]

        # Format results
        formatted = []
        for i in range(len(docs)):
            formatted.append({
                "content": docs[i],
                "metadata": metas[i] if metas else {},
                "distance": dists[i] if dists else None
            })
        
        return formatted
    
    def clear(self):
        """Clear the collection (useful for testing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{self.collection_name}' cleared.")
