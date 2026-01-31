import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from rouge_score import rouge_scorer
from .base import BaseDefense

logger = logging.getLogger(__name__)

class TrustRAGDefense(BaseDefense):
    """
    TrustRAG Defense mechanism.
    Uses K-means clustering on retrieved document embeddings to identify and remove 
    potentially malicious documents (e.g., injection attacks) based on cluster similarity and N-gram overlap.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Ensure thresholds are floats (LLM might pass strings)
        sim_thresh = config.get("similarity_threshold", 0.88)
        rouge_thresh = config.get("rouge_threshold", 0.25)
        
        # Convert to float if string (e.g., "0.88-0.95" -> 0.88)
        if isinstance(sim_thresh, str):
            try:
                sim_thresh = float(sim_thresh.split('-')[0].strip())
            except:
                sim_thresh = 0.88
        if isinstance(rouge_thresh, str):
            try:
                rouge_thresh = float(rouge_thresh.split('-')[0].strip())
            except:
                rouge_thresh = 0.25
        
        self.similarity_threshold = float(sim_thresh)
        self.rouge_threshold = float(rouge_thresh)
        self.candidate_multiplier = config.get("candidate_multiplier", 3)
        self.target_top_k = 5  # Will be updated in pre_retrieval

    def pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Request more candidates than needed to allow for TrustRAG filtering.
        """
        self.target_top_k = top_k
        fetch_k = top_k * self.candidate_multiplier
        logger.info(f"[TrustRAG] Increasing retrieval limit from {top_k} to {fetch_k}")
        return query, fetch_k

    def post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter documents using TrustRAG's K-means defense mechanism.
        """
        if len(documents) <= 2:
            return documents

        # unexpected case (no embeddings) - return raw documents
        if not all(d.get("embedding") is not None for d in documents):
            logger.warning("[TrustRAG] Missing embeddings in some documents. Skipping defense.")
            return documents
        
        # specific handling to preserve original dict structure
        doc_texts = [d["content"] for d in documents]
        embeddings = [d["embedding"] for d in documents]
        
        filtered_texts, _ = self.trustrag_kmeans_filter(
            doc_texts, 
            embeddings,
            self.similarity_threshold,
            self.rouge_threshold
        )
        
        # reconstruct the list of dicts, preserving the order of the remaining documents 
        # (or at least finding them back).
        # Since the filter might return a subset or reordered list, we need to map back.
        # However, the filter might merge clusters, so order might change.
        # We'll just build a new list based on the returned texts. 
        # CAUTION: if there are duplicate texts, this might pick the first one matching. 
        # Assuming unique content for simplicity or acceptable behavior.
        
        filtered_docs = []
        original_map = {d["content"]: d for d in documents}
        
        for text in filtered_texts:
            if text in original_map:
                filtered_docs.append(original_map[text])
            else:
                # Should not happen unless filter modifies text, which it doesn't seem to do
                logger.warning(f"[TrustRAG] Filtered text not found in original documents: {text[:50]}...")
                
        logger.info(f"[TrustRAG] Filtered {len(documents)} -> {len(filtered_docs)} docs")
        
        # Return filtered docs (will be limited to top_k by manager)
        return filtered_docs

    def trustrag_kmeans_filter(self, documents, embeddings, similarity_threshold=0.88, rouge_threshold=0.25):
        """
        Filter retrieved documents using TrustRAG's K-means defense mechanism.
        
        Args:
            documents: List of retrieved document texts
            embeddings: List of document embeddings (numpy array or list)
            similarity_threshold: Threshold for intra-cluster similarity (default: 0.88)
            rouge_threshold: Threshold for ROUGE-L similarity (default: 0.25)
        
        Returns:
            filtered_docs: List of filtered documents
            filtered_embeddings: List of corresponding embeddings
        """
        if len(documents) <= 2:
            return documents, embeddings
        
        # Convert to numpy array if needed
        embeddings = np.array(embeddings)
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        
        # Avoid division by zero
        norms = np.sqrt((embeddings_norm**2).sum(axis=1))[:, None]
        norms[norms == 0] = 1e-10
        embeddings_norm = embeddings_norm / norms
        
        # K-means clustering (k=2)
        # Handle case where n_samples < n_clusters
        n_clusters = 2
        if len(documents) < n_clusters:
             return documents, embeddings

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=500, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_norm)
        
        # Split into clusters
        cluster_0_docs = [documents[i] for i, label in enumerate(cluster_labels) if label == 0]
        cluster_0_emb = [embeddings[i] for i, label in enumerate(cluster_labels) if label == 0]
        cluster_1_docs = [documents[i] for i, label in enumerate(cluster_labels) if label == 1]
        cluster_1_emb = [embeddings[i] for i, label in enumerate(cluster_labels) if label == 1]
        
        # Calculate intra-cluster similarities
        def calculate_intra_cluster_similarity(cluster_emb):
            if len(cluster_emb) < 2:
                return 0.0
            similarities = []
            for i in range(len(cluster_emb)):
                for j in range(i + 1, len(cluster_emb)):
                    sim = cosine_similarity([cluster_emb[i]], [cluster_emb[j]])[0][0]
                    similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0
        
        sim_0 = calculate_intra_cluster_similarity(cluster_0_emb)
        sim_1 = calculate_intra_cluster_similarity(cluster_1_emb)
        
        logger.info(f"[TrustRAG] Cluster 0 sim: {sim_0:.4f}, Cluster 1 sim: {sim_1:.4f}")

        # N-gram filtering function
        def ngram_filter(docs):
            if len(docs) < 2:
                return docs, list(range(len(docs)))
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            del_indices = set()
            
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    score = scorer.score(docs[i], docs[j])['rougeL'].fmeasure
                    if score > rouge_threshold:
                        del_indices.add(j)
            
            filtered = [doc for idx, doc in enumerate(docs) if idx not in del_indices]
            return filtered, del_indices
        
        # Defense decision logic
        if sim_0 > similarity_threshold and sim_1 > similarity_threshold:
            # Both clusters suspicious - discard all
            logger.info("[TrustRAG] Both clusters suspicious - discarding all")
            return [], []
        
        elif sim_0 > similarity_threshold:
            # Cluster 0 is malicious, keep cluster 1
            logger.info("[TrustRAG] Cluster 0 suspicious - keeping cluster 1")
            filtered_docs, _ = ngram_filter(cluster_1_docs)
            filtered_emb = [cluster_1_emb[i] for i, doc in enumerate(cluster_1_docs) if doc in filtered_docs]
            return filtered_docs, filtered_emb
        
        elif sim_1 > similarity_threshold:
            # Cluster 1 is malicious, keep cluster 0
            logger.info("[TrustRAG] Cluster 1 suspicious - keeping cluster 0")
            filtered_docs, _ = ngram_filter(cluster_0_docs)
            filtered_emb = [cluster_0_emb[i] for i, doc in enumerate(cluster_0_docs) if doc in filtered_docs]
            return filtered_docs, filtered_emb
        
        else:
            # Neither cluster exceeds threshold - apply n-gram filtering and merge
            logger.info("[TrustRAG] Neither cluster suspicious - merging")
            filtered_0, del_0 = ngram_filter(cluster_0_docs)
            filtered_1, del_1 = ngram_filter(cluster_1_docs)
            
            # Keep embeddings for non-deleted docs
            kept_emb_0 = [cluster_0_emb[i] for i, doc in enumerate(cluster_0_docs) if doc in filtered_0]
            kept_emb_1 = [cluster_1_emb[i] for i, doc in enumerate(cluster_1_docs) if doc in filtered_1]
            
            return filtered_0 + filtered_1, kept_emb_0 + kept_emb_1
