"""
Retrieval module for ChromaDB vector store with local embeddings.
Uses sentence-transformers for fast batch embedding (no Ollama overhead).
"""
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP and library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

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
        include_embeddings: bool = False
    ) -> List[dict]:
        """
        Query the vector store for similar documents.
        Returns list of dicts with 'content', 'metadata', 'distance'.
        
        Args:
            query_text: The query string
            top_k: Number of documents to retrieve
            include_embeddings: Whether to include embeddings in results (needed by some defenses)
        """
        query_embedding = self.embedder.embed_single(query_text)
        
        # Only include embeddings if needed (significantly improves performance when not needed)
        include_fields = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=include_fields
        )
        
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []
        embeddings = results["embeddings"][0] if results["embeddings"] else []
        
        # Format results
        formatted = []
        for i in range(len(docs)):
            formatted.append({
                "content": docs[i],
                "metadata": metas[i] if metas else {},
                "distance": dists[i] if dists else None,
                "embedding": embeddings[i] if embeddings is not None and len(embeddings) > 0 else None
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

