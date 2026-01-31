import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from .base import BaseDefense

logger = logging.getLogger(__name__)

class DifferentialPrivacyDefense(BaseDefense):
    """
    Implements Differential Privacy mechanism for RAG retrieval.
    Adds noise to the filtering threshold to protect document presence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.method = config.get("method", "dp_pure")
        self.epsilon = config.get("epsilon", 1.0)
        self.delta = config.get("delta", 0.01)
        self.candidate_multiplier = config.get("candidate_multiplier", 3)
        self.target_top_k = 5 # Will be updated in pre_retrieval

    def pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Request more candidates than needed to allow for DP filtering.
        """
        self.target_top_k = top_k
        fetch_k = top_k * self.candidate_multiplier
        logger.info(f"[DP Defense] Increasing retrieval limit from {top_k} to {fetch_k}")
        return query, fetch_k

    def post_retrieval(self, documents: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Filter documents based on DP threshold.
        """
        if not documents:
            return []

        # Extract distances to calculate scores
        # Assuming Cosine Distance: Score = 1 - Distance
        # VectorStore returns "distance", handle None
        valid_docs = [d for d in documents if d.get("distance") is not None]
        
        if not valid_docs:
            logger.warning("[DP Defense] No valid distances found in documents. Skipping DP.")
            return documents[:self.target_top_k]

        dists = np.array([d["distance"] for d in valid_docs])
        scores = 1.0 - dists
        
        # Calculate threshold
        if self.method == "dp_pure":
            threshold = self._sample_dp_threshold_pure(
                scores, self.target_top_k, self.epsilon, min_score=-1.0, max_score=1.0
            )
        elif self.method == "dp_approx":
            threshold = self._sample_dp_threshold_approximate(
                scores, self.target_top_k, self.epsilon, self.delta, min_score=-1.0
            )
        else:
            logger.warning(f"Unknown DP method: {self.method}")
            return documents[:self.target_top_k]
            
        logger.info(f"[DP Defense] DP Threshold: {threshold:.4f}")

        # Filter
        filtered_docs = []
        for d in valid_docs:
            score = 1.0 - d["distance"]
            if score > threshold:
                filtered_docs.append(d)
        
        logger.info(f"[DP Defense] Filtered {len(documents)} -> {len(filtered_docs)} docs")
        
        # Return filtered docs without limiting - manager will cap at top_k after all defenses
        # This allows documents to flow freely to next defense in stacked configuration
        return filtered_docs

    def _sample_dp_threshold_pure(self, scores: np.ndarray, k: int, 
                                 epsilon: float, min_score: float = -1.0, 
                                 max_score: float = 1.0) -> float:
        """Pure differential privacy using exponential mechanism."""
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, min_score)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), max_score)
        
        utilities = -np.abs(len(sorted_scores) - k - np.arange(len(sorted_scores)))
        pdf = np.exp(epsilon * utilities[:-1] / 2) * np.diff(sorted_scores)
        
        if np.sum(pdf) == 0:
            return min_score
        pdf /= np.sum(pdf)
        
        return np.random.choice(sorted_scores[:-1], p=pdf)

    def _sample_dp_threshold_approximate(self, scores: np.ndarray, k: int,
                                        epsilon: float, delta: float,
                                        min_score: float = -1.0) -> float:
        """Approximate differential privacy using Gaussian mechanism."""
        if delta <= 0:
            raise ValueError("Delta must be > 0 for approximate DP")
        
        sensitivity = 1.0
        noise_scale = (2 * sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
        
        sorted_scores = np.sort(scores)
        kth_score = sorted_scores[-k] if k <= len(scores) else min_score
        
        noisy_threshold = kth_score + np.random.normal(0, noise_scale)
        return noisy_threshold

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_delta(self, delta: float):
        self.delta = delta

    def get_epsilon(self):
        return self.epsilon

    def get_delta(self):
        return self.delta
