import logging
from typing import List, Dict, Any, Tuple, Optional
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
    
    def __init__(self, config: List[Dict[str, Any]], ado_enabled: bool = False):
        """
        Initialize defenses from configuration list.
        
        Args:
            config: List of defense configurations, e.g.:
                   [{'name': 'differential_privacy', 'method': 'dp_pure', ...}]
            ado_enabled: If True, pre-initialize ALL defenses for dynamic switching
        """
        self.original_config = config
        self.ado_enabled = ado_enabled
        
        # Defense pools
        self.all_defenses: Dict[str, BaseDefense] = {}  # Pool of all instances (keyed by name)
        self.active_defenses: List[BaseDefense] = []     # Currently active defenses
        
        # Manager-level state
        self.target_top_k: int = 5
        self.needs_embeddings: bool = False
        
        if ado_enabled:
            # Pre-initialize ALL defenses for ADO mode
            self._preinit_all_defenses(config)
        else:
            # Static mode: only initialize enabled defenses
            self._init_enabled_defenses(config)

    def _preinit_all_defenses(self, config: List[Dict[str, Any]]):
        """
        Pre-initialize ALL defenses for ADO mode.
        This allows instant activation/deactivation without recreating instances.
        """
        logger.info("[Manager] ADO mode: Pre-initializing all defenses...")
        
        # Build config map by name
        config_map = {d.get("name"): d for d in config if d.get("name")}
        
        for name, defense_class in DEFENSE_REGISTRY.items():
            # Get config for this defense, or use defaults
            defense_conf = config_map.get(name, {"name": name})
            defense_conf["name"] = name  # Ensure name is set
            
            try:
                defense_instance = defense_class(defense_conf)
                self.all_defenses[name] = defense_instance
                logger.info(f"[Manager] Pre-initialized defense: {name}")
                
                # If enabled in config, add to active list
                if defense_conf.get("enabled", False):
                    self.active_defenses.append(defense_instance)
                    
            except Exception as e:
                logger.error(f"[Manager] Failed to pre-initialize defense {name}: {e}")
        
        # Update dependency flags
        self._update_dependency_flags()
        logger.info(f"[Manager] Active defenses: {[d.name for d in self.active_defenses]}")

    def _init_enabled_defenses(self, config: List[Dict[str, Any]]):
        """
        Initialize only enabled defenses (static mode).
        """
        self.all_defenses = {}
        self.active_defenses = []
        
        if not config:
            return

        for defense_conf in config:
            name = defense_conf.get("name")
            if not name:
                logger.warning("Defense configuration missing 'name'. Config: %s", defense_conf)
                continue
                
            defense_class = DEFENSE_REGISTRY.get(name)
            if defense_class:
                if defense_conf.get("enabled", True):
                    try:
                        defense_instance = defense_class(defense_conf)
                        self.all_defenses[name] = defense_instance
                        self.active_defenses.append(defense_instance)
                        logger.info(f"[Manager] Defense initialized: {name}")
                    except Exception as e:
                        logger.error(f"[Manager] Failed to initialize defense {name}: {e}")
            else:
                logger.warning(f"[Manager] Unknown defense type: {name}")
        
        self._update_dependency_flags()

    def _update_dependency_flags(self):
        """Update flags based on active defenses."""
        self.needs_embeddings = any(d.name == "trustrag" for d in self.active_defenses)

    def _update_defense_params(self, defense: BaseDefense, settings: Dict[str, Any]):
        """
        Update a defense's parameters using setters or direct attribute assignment.
        """
        # Update the config dict
        defense.config.update(settings)
        
        for param, value in settings.items():
            if param in ("enabled", "name"):
                continue
            
            # Try setter first (e.g., set_epsilon)
            setter_name = f"set_{param}"
            if hasattr(defense, setter_name):
                try:
                    getattr(defense, setter_name)(value)
                    logger.debug(f"[Manager] Updated {defense.name}.{param} via {setter_name}")
                except Exception as e:
                    logger.error(f"[Manager] Failed to set {defense.name}.{param}: {e}")
            elif hasattr(defense, param):
                # Direct attribute update
                setattr(defense, param, value)
                logger.debug(f"[Manager] Updated {defense.name}.{param} directly")

    def set_dynamic_config(self, defense_plan: Dict[str, Any]):
        """
        Reconfigure defenses at runtime based on Strategist plan (ADO mode).
        Uses the pre-initialized pool instead of creating new instances.
        
        Args:
           defense_plan: Dict mapping defense_name -> {enabled: bool, ...params}
        """
        new_active = []
        
        for name, plan_settings in defense_plan.items():
            if name not in DEFENSE_REGISTRY:
                continue
            
            # Check enabled status
            if not plan_settings.get("enabled", False):
                continue
            
            # Get from pool (should exist if ADO mode was used)
            if name in self.all_defenses:
                defense = self.all_defenses[name]
                logger.info(f"[Manager] Activating defense from pool: {name}")
                
                # Update parameters
                self._update_defense_params(defense, plan_settings)
                new_active.append(defense)
                
            else:
                # Fallback: create new instance (shouldn't happen in ADO mode)
                logger.warning(f"[Manager] Defense {name} not in pool, creating new instance")
                defense_conf = {"name": name}
                defense_conf.update(plan_settings)
                
                defense_class = DEFENSE_REGISTRY.get(name)
                try:
                    defense_instance = defense_class(defense_conf)
                    self.all_defenses[name] = defense_instance
                    new_active.append(defense_instance)
                except Exception as e:
                    logger.error(f"[Manager] Failed to initialize defense {name}: {e}")
        
        # Update the active list
        self.active_defenses = new_active
        self._update_dependency_flags()
        
        logger.info(f"[Manager] Active defenses updated: {[d.name for d in self.active_defenses]}")

    def activate_defense(self, name: str) -> bool:
        """
        Activate a pre-initialized defense by name.
        
        Returns:
            True if activated, False if not found or already active
        """
        if name not in self.all_defenses:
            logger.warning(f"[Manager] Cannot activate unknown defense: {name}")
            return False
        
        defense = self.all_defenses[name]
        if defense in self.active_defenses:
            logger.debug(f"[Manager] Defense already active: {name}")
            return True
        
        self.active_defenses.append(defense)
        self._update_dependency_flags()
        logger.info(f"[Manager] Activated defense: {name}")
        return True

    def deactivate_defense(self, name: str) -> bool:
        """
        Deactivate a defense by name (keeps instance in pool).
        
        Returns:
            True if deactivated, False if not found or not active
        """
        if name not in self.all_defenses:
            logger.warning(f"[Manager] Cannot deactivate unknown defense: {name}")
            return False
        
        defense = self.all_defenses[name]
        if defense not in self.active_defenses:
            logger.debug(f"[Manager] Defense not active: {name}")
            return True
        
        self.active_defenses.remove(defense)
        self._update_dependency_flags()
        logger.info(f"[Manager] Deactivated defense: {name}")
        return True

    def is_defense_active(self, name: str) -> bool:
        """Check if a defense is currently active."""
        if name not in self.all_defenses:
            return False
        return self.all_defenses[name] in self.active_defenses

    def get_defense_status(self) -> Dict[str, Any]:
        """Get current status of all defenses."""
        active_names = [d.name for d in self.active_defenses]
        return {
            "active": active_names,
            "inactive": [n for n in self.all_defenses if n not in active_names],
            "needs_embeddings": self.needs_embeddings,
            "target_top_k": self.target_top_k
        }

    def apply_pre_retrieval(self, query: str, top_k: int) -> Tuple[str, int]:
        """
        Calculate fetch_k based on active defenses.
        
        Logic:
        - No defenses: fetch_k = top_k
        - Single defense: fetch_k = top_k * candidate_multiplier
        - Multiple defenses: fetch_k = top_k * max(multipliers)
          (Use max to avoid excessive retrieval while ensuring enough docs)
        """
        self.target_top_k = top_k
        
        if not self.active_defenses:
            return query, top_k
        
        # Collect multipliers from all active defenses
        multipliers = []
        for defense in self.active_defenses:
            multiplier = defense.config.get("candidate_multiplier", 1)
            multipliers.append(multiplier)
        
        # Calculate fetch_k
        if len(self.active_defenses) == 1:
            # Single defense: use its multiplier
            fetch_k = top_k * multipliers[0]
        else:
            # Multiple defenses: use max multiplier
            # This prevents fetch_k explosion while ensuring enough docs for all defenses
            for i in range(len(self.active_defenses)):
                fetch_k = top_k * multipliers[i] / (i + 1)
        
        fetch_k = int(fetch_k)
        logger.info(f"[Manager] Pre-retrieval: top_k={top_k}, fetch_k={fetch_k}, "
                   f"active_defenses={[d.name for d in self.active_defenses]}")
        
        return query, fetch_k

    def apply_post_retrieval(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Apply all post-retrieval hooks sequentially.
        
        Flow:
        1. Pass documents through each active defense's filter
        2. Re-sort by original similarity score (distance) - ascending order
        3. Limit to target_top_k
        """
        current_docs = documents
        
        # Sequential filtering through all active defenses
        for defense in self.active_defenses:
            current_docs = defense.post_retrieval(current_docs, query)
        
        # Re-sort by original distance (ascending = most similar first)
        # This ensures we return the most relevant docs after filtering
        if current_docs:
            current_docs = sorted(
                current_docs,
                key=lambda d: d.get("distance", float('inf'))
            )
        
        # Final limit to target_top_k
        if len(current_docs) > self.target_top_k:
            logger.info(f"[Manager] Final limiting: {len(current_docs)} -> top_k={self.target_top_k}")
            current_docs = current_docs[:self.target_top_k]
        
        return current_docs

    def apply_pre_generation(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        contexts: List[str]
    ) -> Tuple[str, str, List[str]]:
        """Apply all pre-generation hooks sequentially."""
        s_prompt, u_prompt, ctxs = system_prompt, user_prompt, contexts
        
        for defense in self.active_defenses:
            s_prompt, u_prompt, ctxs = defense.pre_generation(s_prompt, u_prompt, ctxs)
            
        return s_prompt, u_prompt, ctxs

    def apply_post_generation(self, response: str) -> str:
        """Apply all post-generation hooks sequentially."""
        current_response = response
        
        for defense in self.active_defenses:
            current_response = defense.post_generation(current_response)
            
        return current_response

    def get_shared_av_model(self):
        """
        Get the Llama model instance from AttentionFilteringDefense if available.
        This allows sharing the model with the generator to save GPU memory.
        
        Returns:
            Llama model instance or None
        """
        # Check in pool first (for ADO mode)
        if "attention_filtering" in self.all_defenses:
            defense = self.all_defenses["attention_filtering"]
            if isinstance(defense, AttentionFilteringDefense):
                return defense.llm
        
        # Fallback: check active defenses
        for defense in self.active_defenses:
            if isinstance(defense, AttentionFilteringDefense):
                return defense.llm
        
        return None

    # Backward compatibility properties
    @property
    def defenses(self) -> List[BaseDefense]:
        """Backward compatibility: return active defenses."""
        return self.active_defenses
