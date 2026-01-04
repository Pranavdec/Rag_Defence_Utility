from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class BaseDefense(ABC):
    """
    Abstract base class for all RAG defenses.
    Defines hooks for intercepting RAG pipeline stages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "UnnamedDefense")

    def pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Hook executed before retrieval.
        Can modify the query or the number of documents to retrieve (fetch_k).
        
        Args:
            query: The user query
            top_k: The target number of documents
            
        Returns:
            Tuple of (modified_query, fetch_k)
        """
        return query, top_k

    def post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Hook executed after retrieval but before generation.
        Can filter, mask, or modify retrieved documents.
        
        Args:
            documents: List of retrieved document dicts (content, metadata, etc.)
            query: The original user query
            
        Returns:
            Modified list of documents
        """
        return documents

    def pre_generation(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        contexts: List[str]
    ) -> Tuple[str, str, List[str]]:
        """
        Hook executed before LLM generation.
        Can modify prompts or context.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt containing the question
            contexts: List of context strings
            
        Returns:
            Tuple of (modified_system_prompt, modified_user_prompt, modified_contexts)
        """
        return system_prompt, user_prompt, contexts

    def post_generation(self, response: str) -> str:
        """
        Hook executed after LLM generation.
        Can sanitize or validate the response.
        
        Args:
            response: The implementation generated text
            
        Returns:
            Modified response
        """
        return response
