import logging
from typing import List, Dict, Any, Tuple
from .base import BaseDefense
from .dp_rag import DifferentialPrivacyDefense
from .trustrag import TrustRAGDefense

# Registry of available defenses
DEFENSE_REGISTRY = {
    "differential_privacy": DifferentialPrivacyDefense,
    "trustrag": TrustRAGDefense,
}

logger = logging.getLogger(__name__)

class DefenseManager:
    """
    Manager to handle initialization and execution of multiple defenses.
    Acts as a middleware in the RAG pipeline.
    """
    
    def __init__(self, config: List[Dict[str, Any]]):
        """
        Initialize defenses from configuration list.
        
        Args:
            config: List of defense configurations, e.g.:
                   [{'name': 'differential_privacy', 'method': 'dp_pure', ...}]
        """
        self.defenses: List[BaseDefense] = []
        
        if not config:
            return

        for defense_conf in config:
            name = defense_conf.get("name")
            if not name:
                logger.warning("Defense configuration missing 'name'. Strategies: %s", defense_conf)
                continue
                
            defense_class = DEFENSE_REGISTRY.get(name)
            if defense_class:
                if defense_conf.get("enabled", True):
                    try:
                        defense_instance = defense_class(defense_conf)
                        self.defenses.append(defense_instance)
                        logger.info(f"Defense initialized: {name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize defense {name}: {e}")
            else:
                logger.warning(f"Unknown defense type: {name}")

    def apply_pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """Apply all pre-retrieval hooks sequentially."""
        current_query = query
        current_fetch_k = top_k
        
        for defense in self.defenses:
            current_query, current_fetch_k = defense.pre_retrieval(current_query, current_fetch_k)
            
        return current_query, current_fetch_k

    def apply_post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply all post-retrieval hooks sequentially."""
        current_docs = documents
        
        for defense in self.defenses:
            current_docs = defense.post_retrieval(current_docs, query)
            
        return current_docs

    def apply_pre_generation(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        contexts: List[str]
    ) -> Tuple[str, str, List[str]]:
        """Apply all pre-generation hooks sequentially."""
        s_prompt, u_prompt, ctxs = system_prompt, user_prompt, contexts
        
        for defense in self.defenses:
            s_prompt, u_prompt, ctxs = defense.pre_generation(s_prompt, u_prompt, ctxs)
            
        return s_prompt, u_prompt, ctxs

    def apply_post_generation(self, response: str) -> str:
        """Apply all post-generation hooks sequentially."""
        current_response = response
        
        for defense in self.defenses:
            current_response = defense.post_generation(current_response)
            
        return current_response
