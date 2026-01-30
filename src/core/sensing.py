import re
import time
import math
import numpy as np
from typing import List, Dict, Any

class MetricsCollector:
    """
    Implements the ADO 'Sensing Layer'. 
    Calculates tight, objective metrics before and after retrieval.
    """
    
    def __init__(self):
        self.last_query_time = 0.0

    # --- Pre-Retrieval Metrics ---

    def calculate_pre_retrieval(self, query: str, history: List[str] = []) -> Dict[str, float]:
        """
        Calculates lexical, complexity, and timing metrics on the raw query.
        """
        metrics = {}
        
        # M_LEX: Lexical Overlap (Jaccard) with immediate history
        # (Detects repetitive probing)
        if history:
            last_query = history[-1]
            metrics['m_lex'] = self._jaccard_similarity(query, last_query)
        else:
            metrics['m_lex'] = 0.0

        # M_CMP: Complexity Score (Special chars ratio)
        # (Detects obfuscation/jailbreaks)
        metrics['m_cmp'] = self._complexity_score(query)

        # M_INT: Intent Velocity (Time delta)
        # (Detects bot/automated attacks)
        current_time = time.time()
        if self.last_query_time > 0:
            delta = current_time - self.last_query_time
            # Normalize: < 0.5s is suspicious (1.0 risk), > 2.0s is normal (0.0 risk)
            metrics['m_int'] = 1.0 if delta < 0.5 else max(0.0, 1.0 - (delta / 2.0))
        else:
            metrics['m_int'] = 0.0
        
        self.last_query_time = current_time
        return metrics

    # --- Retrieval Metrics ---

    def calculate_retrieval(self, top_docs_scores: List[float], top_docs_embeddings: List[List[float]]) -> Dict[str, float]:
        """
        Calculates metrics based on retrieved vector distribution.
        """
        metrics = {}
        
        if not top_docs_scores:
            return {'m_dis': 0.0, 'm_drp': 0.0}

        # M_DRP: Score Drop-off (Top-1 vs Top-5)
        # Large drop = Top-1 is an outlier (potential probed specific point)
        # Small drop = Dense cluster (broad topic)
        if len(top_docs_scores) > 1:
            try:
                # Assuming distances, so lower is better. normalized 0-1 usually.
                # If these are cosine SIMILARITY (1.0 is identical):
                # drop = top1 - topK
                metrics['m_drp'] = abs(top_docs_scores[0] - top_docs_scores[-1])
            except:
                metrics['m_drp'] = 0.0
        else:
            metrics['m_drp'] = 0.0

        # M_DIS: Vector Dispersion
        # Variance of distances among retrieved chunks.
        # High variance = Conflicting contexts (Poisoning risk) or hallucination prone
        if top_docs_embeddings and len(top_docs_embeddings) > 1:
            try:
                # Use actual embedding variance when available
                emb_array = np.array(top_docs_embeddings)
                # Calculate variance across embedding dimensions, then mean
                # High value = embeddings are spread out (diverse/conflicting contexts)
                metrics['m_dis'] = float(np.mean(np.var(emb_array, axis=0)))
            except Exception:
                # Fallback to score variance
                metrics['m_dis'] = float(np.var(top_docs_scores)) if top_docs_scores else 0.0
        elif top_docs_scores and len(top_docs_scores) > 1:
            # Fallback: use score variance as proxy
            metrics['m_dis'] = float(np.var(top_docs_scores))
        else:
            metrics['m_dis'] = 0.0
            
        return metrics

    # --- Helpers ---

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def _complexity_score(self, text: str) -> float:
        if not text:
            return 0.0
        # Count non-alphanumeric chars (excluding spaces)
        special_chars = len(re.sub(r'[a-zA-Z0-9\s]', '', text))
        return special_chars / len(text)
