import logging
from typing import List, Dict, Any, Tuple
from .base import BaseDefense
from .dp_rag import DifferentialPrivacyDefense
from .trustrag import TrustRAGDefense
from .av_defense import AttentionFilteringDefense

# Registry of available defenses
DEFENSE_REGISTRY = {
    "differential_privacy": DifferentialPrivacyDefense,
    "trustrag": TrustRAGDefense,
    "attention_filtering": AttentionFilteringDefense,
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
        self.original_config = config # Store for reset if needed
        self.defenses: List[BaseDefense] = []
        self.needs_embeddings = False
        self._init_defenses(config)

    def _init_defenses(self, config: List[Dict[str, Any]]):
        """Helper to initialize defense list."""
        self.defenses = []
        self.needs_embeddings = False
        
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
                        # TrustRAG needs embeddings for clustering
                        if name == "trustrag":
                            self.needs_embeddings = True
                    except Exception as e:
                        logger.error(f"Failed to initialize defense {name}: {e}")
            else:
                logger.warning(f"Unknown defense type: {name}")

    def set_dynamic_config(self, defense_plan: Dict[str, Any]):
        """
        Reconfigure defenses at runtime based on Strategist plan.
        
        Args:
           defense_plan: Dict mapping defense_name -> {enabled: bool, ...params}
        """
        # We need to construct a new config list based on the plan
        # The plan is a dict, but we need a list of configs.
        # We start with the original config so we keep static tokens/secrets if any
        # But for this system, most config is simple parameters.
        
        # Strategy: 
        # 1. Iterate through known defenses in registry (or just supported ones in ADO)
        # 2. Build a new config list.
        
        new_config_list = []
        
        # We assume defense_plan keys match registry keys
        for name, plan_settings in defense_plan.items():
            if name not in DEFENSE_REGISTRY:
                continue
            
            # Create a clean config dict for this defense
            defense_conf = {"name": name}
            defense_conf.update(plan_settings) # Add enabled, epsilon, etc.
            
            # If enabled is false, we might ideally skip adding it 
            # OR we add it with enabled=False if the class supports that (checked in _init).
            # _init checks `defense_conf.get("enabled", True)`. 
            # So passing enabled=False works.
            
            new_config_list.append(defense_conf)
            
        # Re-initialize
        logger.info(f"Applying Dynamic Defense Config: {defense_plan}")
        self._init_defenses(new_config_list)

    def apply_pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Apply all pre-retrieval hooks.
        - Query modifications are applied sequentially
        - fetch_k calculation:
          * Single defense: fetch_k = top_k * multiplier
          * Stacked defenses: fetch_k = top_k * multiplier1 * (multiplier2/num_defenses) * (multiplier3/num_defenses) ...
            This ensures sufficient documents flow through all defense stages while avoiding excessive retrieval.
        - target_top_k for DP Defense only: Set higher intermediate target for DP if stacked
        """
        current_query = query
        num_defenses = len(self.defenses)
        
        # Special handling for DP Defense target_top_k in stacked configuration
        # DP uses target_top_k in its algorithm, so it needs to aim higher when not the last defense
        if num_defenses > 1:
            for idx, defense in enumerate(self.defenses):
                if hasattr(defense, 'target_top_k') and defense.__class__.__name__ == 'DifferentialPrivacyDefense':
                    remaining_defenses = num_defenses - idx - 1
                    if remaining_defenses > 0:
                        # DP not last: target more docs (e.g., 2x top_k for downstream filtering)
                        defense.target_top_k = top_k * 2
                        logger.info(f"[Manager] DP at position {idx+1}/{num_defenses}: target_top_k={defense.target_top_k}")
                    else:
                        # DP is last: target final top_k
                        defense.target_top_k = top_k
        else:
            # Single defense: use final top_k
            for defense in self.defenses:
                if hasattr(defense, 'target_top_k'):
                    defense.target_top_k = top_k
        
        # Apply query modifications sequentially
        for defense in self.defenses:
            current_query, _ = defense.pre_retrieval(current_query, top_k)
        
        # Calculate fetch_k based on single vs stacked defense logic
        if len(self.defenses) == 0:
            fetch_k = top_k
        elif len(self.defenses) == 1:
            # Single defense: use its multiplier directly
            _, fetch_k = self.defenses[0].pre_retrieval(query, top_k)
        else:
            # Stacked defenses: cascading multiplier formula
            # fetch_k = top_k * multiplier1 * (multiplier2/n) * (multiplier3/n) ...
            fetch_k = top_k
            
            for idx, defense in enumerate(self.defenses):
                _, defense_fetch_k = defense.pre_retrieval(query, top_k)
                multiplier = defense_fetch_k / top_k if top_k > 0 else 1.0
                
                if idx == 0:
                    # First defense multiplier applied directly
                    fetch_k = fetch_k * multiplier
                else:
                    # Subsequent defenses: divide by num_defenses
                    fetch_k = fetch_k * (multiplier / num_defenses)
            
            fetch_k = int(fetch_k)
            
        logger.info(f"[Manager] Pre-retrieval: top_k={top_k}, fetch_k={fetch_k}, num_defenses={len(self.defenses)}")
        return current_query, fetch_k

    def apply_post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Apply all post-retrieval hooks sequentially.
        Documents flow through each defense without intermediate limiting.
        Final cap at top_k happens only after all defenses have processed.
        """
        current_docs = documents
        
        # Sequential filtering through all defenses - no intermediate limiting
        for defense in self.defenses:
            current_docs = defense.post_retrieval(current_docs, query)
        
        # Final limit to top_k after all defenses have filtered
        # Use target_top_k from any defense (they should all be consistent with global top_k)
        if self.defenses and len(current_docs) > 0:
            if hasattr(self.defenses[0], 'target_top_k'):
                target_top_k = self.defenses[0].target_top_k
                if len(current_docs) > target_top_k:
                    logger.info(f"[Manager] Final limiting: {len(current_docs)} documents -> top_k={target_top_k}")
                    current_docs = current_docs[:target_top_k]
            
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

    def get_shared_av_model(self):
        """
        Get the Llama model instance from AttentionFilteringDefense if enabled.
        This allows sharing the model with the generator to save GPU memory.
        
        Returns:
            Llama model instance or None
        """
        for defense in self.defenses:
            if isinstance(defense, AttentionFilteringDefense):
                return defense.llm
        return None
