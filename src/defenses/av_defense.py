import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from .base import BaseDefense
from ..core.hf_model import create_model
import torch

logger = logging.getLogger(__name__)

def wrap_prompt(question, contexts):
    """
    Constructs the prompt with contexts.
    Adapting for standard RAG prompt format.
    """
    prompt = f"Question: {question}\n\n"
    for i, ctx in enumerate(contexts):
        prompt += f"[{i+1}] {ctx}\n\n"
    
    prompt += "Answer based on the above contexts found in [...]."
    return prompt

class AttentionFilteringDefense(BaseDefense):
    """
    Filters retrieved documents based on attention scores from a local LLM.
    Identifies "benign" passages that actually contribute to the answer.
    """
    
    # Class-level shared model (singleton pattern to avoid loading multiple instances)
    _shared_model = None
    _shared_model_path = None
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        # Config extraction
        self.model_path = config.get("model_path", "meta-llama/Llama-3.1-8B-Instruct")
        
        # Alpha (top_tokens): supports "all" for infinite tokens (paper setting)
        alpha = config.get("top_tokens", 100)
        self.top_tokens = None if str(alpha).lower() == "all" else int(alpha)
        
        self.max_corruptions = config.get("max_corruptions", 5)
        self.threshold = config.get("threshold", 26.2)
        
        self.candidate_multiplier = config.get("candidate_multiplier", 3)
        self.target_top_k = 5  # Will be updated in pre_retrieval
        
        # Check if shared model was provided
        shared_model = config.get("shared_model", None)
        
        if shared_model is not None:
            # Use the provided shared model
            self.llm = shared_model
            logger.info(f"[AV Defense] Using shared model instance")
            return
        
        # Check class-level singleton
        if (AttentionFilteringDefense._shared_model is not None and 
            AttentionFilteringDefense._shared_model_path == self.model_path):
            self.llm = AttentionFilteringDefense._shared_model
            logger.info(f"[AV Defense] Reusing singleton model: {self.model_path}")
            return
        
        # Model config wrapper
        model_config = {
            "model_info": {"provider": "llama", "name": self.model_path},
            "params": {
                "max_output_tokens": 512,
                "temperature": config.get("temperature", 0.0)
            }
        }
        
        # Initialize local LLM
        # Determine device from config, default to cuda if available, else cpu
        configured_device = config.get("device", "auto")
        if configured_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = configured_device
            
        logger.info(f"[AV Defense] Initializing local model {self.model_path} on {device}")
        
        try:
            self.llm = create_model(model_config, device=device)
            # Store as singleton
            AttentionFilteringDefense._shared_model = self.llm
            AttentionFilteringDefense._shared_model_path = self.model_path
        except Exception as e:
            logger.error(f"[AV Defense] Failed to initialize model: {e}")
            raise e

    def pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Request more candidates than needed to allow for attention filtering.
        """
        self.target_top_k = top_k
        fetch_k = top_k * self.candidate_multiplier
        logger.info(f"[AV Defense] Increasing retrieval limit from {top_k} to {fetch_k}")
        return query, fetch_k

    def post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter documents using attention scores.
        """
        if not documents:
            return []
            
        original_contents = [d.get("content", "") for d in documents]

        # if len(original_contents) >= self.target_top_k * self.candidate_multiplier:
        #     original_contents = original_contents[:self.target_top_k * self.candidate_multiplier]
        
        # If any content is empty, just return docs to avoid errors
        if not all(original_contents):
            return documents

        logger.info(f"[AV Defense] Filtering {len(documents)} documents using attention scores...")
        
        try:
            filtered_contents = self.filter_by_attention_score(
                topk_contents=original_contents,
                question=query
            )
            
            # Map back to documents
            # Note: We return docs whose content is in filtered_contents
            # This handles duplicates if content is identical
            filtered_docs = [d for d in documents if d.get("content") in filtered_contents]
            
            logger.info(f"[AV Defense] Retained {len(filtered_docs)}/{len(documents)} documents.")
            
            # Return filtered docs without limiting - manager will cap at top_k after all defenses
            # This allows documents to flow freely to next defense in stacked configuration
            return filtered_docs
            
        except Exception as e:
            logger.error(f"[AV Defense] Error during filtering: {e}")
            return documents # Fail safe

    def filter_by_attention_score(self, topk_contents, question):
        removed_count = 0
        contents = topk_contents

        # sorting the passages first time according to the attention scores
        query_prompt = wrap_prompt(question, contents)
        _, passage_scores, _ = self.llm.query(query_prompt, self.top_tokens)
        attention_score_sum = sum(passage_scores)
        sort_normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]

        sorted_contents = [c for c, s in sorted(zip(contents, sort_normalized_passage_scores),key=lambda x: x[1])]

        # sorting the passages second time according to the attention scores
        query_prompt = wrap_prompt(question, sorted_contents)
        _, passage_scores, _ = self.llm.query(query_prompt, self.top_tokens)
        attention_score_sum = sum(passage_scores)
        sort_normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]

        sorted_contents = [c for c, s in sorted(zip(sorted_contents, sort_normalized_passage_scores),key=lambda x: x[1])]

        while removed_count < self.max_corruptions:
            # using the finally sorted passages to compute the final query prompt used for filtering
            query_prompt = wrap_prompt(question, sorted_contents)
            _, passage_scores, _ = self.llm.query(query_prompt, self.top_tokens)
            attention_score_sum = sum(passage_scores)
            normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]

            if np.var(normalized_passage_scores) <= self.threshold:
                break

            max_score = max(normalized_passage_scores)
            max_index = normalized_passage_scores.index(max_score)
            sorted_contents.pop(max_index)
            contents = sorted_contents
            removed_count +=1

        return contents

    def set_max_corruptions(self, max_corruptions):
        self.max_corruptions = max_corruptions

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_top_tokens(self, top_tokens):
        self.top_tokens = top_tokens

    def get_max_corruptions(self):
        return self.max_corruptions

    def get_threshold(self):
        return self.threshold

    def get_top_tokens(self):
        return self.top_tokens