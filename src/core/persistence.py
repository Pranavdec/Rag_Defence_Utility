import os
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class UserContext:
    user_id: str
    global_trust_score: float = 0.5  # Neutral start
    total_interactions: int = 0
    last_active: float = 0.0
    history_summary: str = ""  # Placeholder for long-term summary
    trust_history: list = None  # Track last N trust scores
    metrics_history: list = None  # Track metrics from previous queries
    query_history: list = None  # Track last N queries
    
    def __post_init__(self):
        if self.trust_history is None:
            self.trust_history = []
        if self.metrics_history is None:
            self.metrics_history = []
        if self.query_history is None:
            self.query_history = []

class UserTrustManager:
    """
    Manages user trust scores and persistence.
    Operates as the 'Persistence Layer' in ADO.
    """
    def __init__(self, storage_dir: str = "data/users"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.cache: Dict[str, UserContext] = {}

    def get_user_context(self, user_id: str) -> UserContext:
        """Load user context from disk or cache."""
        if user_id in self.cache:
            return self.cache[user_id]

        file_path = os.path.join(self.storage_dir, f"{user_id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Handle list deserialization
                if 'trust_history' not in data:
                    data['trust_history'] = []
                if 'metrics_history' not in data:
                    data['metrics_history'] = []
                if 'query_history' not in data:
                    data['query_history'] = []
                context = UserContext(**data)
            except Exception as e:
                print(f"Error loading user {user_id}: {e}. Resetting.")
                context = UserContext(user_id=user_id, last_active=time.time())
        else:
            context = UserContext(user_id=user_id, last_active=time.time())
        
        self.cache[user_id] = context
        return context

    def update_trust_score(self, user_id: str, delta: float, reason: str = ""):
        """
        Update the Global Trust Score for a user.
        Delta is added to current score. Clamped between 0.0 and 1.0.
        """
        context = self.get_user_context(user_id)
        old_score = context.global_trust_score
        
        # Apply delta and clamp
        new_score = max(0.0, min(1.0, old_score + delta))
        
        # Add to history (keep last 10 entries)
        if not hasattr(context, 'trust_history') or context.trust_history is None:
            context.trust_history = []
        context.trust_history.append({
            'score': old_score,
            'delta': delta,
            'timestamp': time.time(),
            'reason': reason
        })
        # Keep only last 10 entries
        if len(context.trust_history) > 10:
            context.trust_history = context.trust_history[-10:]
        
        context.global_trust_score = new_score
        context.total_interactions += 1
        context.last_active = time.time()

        self._save_user(context)
        return new_score

    def update_query_history(self, user_id: str, query: str, metrics: Dict[str, Any]):
        """
        Store query and metrics for next round's defense decisions.
        Metrics should contain both 'pre_retrieval' and 'post_retrieval' dicts.
        Keep last 10 entries.
        """
        context = self.get_user_context(user_id)
        
        # Ensure lists exist
        if not hasattr(context, 'query_history') or context.query_history is None:
            context.query_history = []
        if not hasattr(context, 'metrics_history') or context.metrics_history is None:
            context.metrics_history = []
        
        # Add current query and metrics
        context.query_history.append(query)
        context.metrics_history.append({
            'timestamp': time.time(),
            'pre_retrieval': metrics.get('pre_retrieval', {}),
            'post_retrieval': metrics.get('post_retrieval', {})
        })
        
        # Keep only last 10
        if len(context.query_history) > 10:
            context.query_history = context.query_history[-10:]
        if len(context.metrics_history) > 10:
            context.metrics_history = context.metrics_history[-10:]
        
        self._save_user(context)
    
    def _save_user(self, context: UserContext):
        """Persist user context to disk."""
        file_path = os.path.join(self.storage_dir, f"{context.user_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(context), f, indent=2)
        except Exception as e:
            print(f"Failed to save user {context.user_id}: {e}")
