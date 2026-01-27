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
        
        context.global_trust_score = new_score
        context.total_interactions += 1
        context.last_active = time.time()
        
        # Log significant changes (optional, could be expanded to a history list)
        if reason:
             # In a real system, we'd append to a structured log. 
             # For now, we trust the caller to handle detailed logging
             pass

        self._save_user(context)
        return new_score

    def _save_user(self, context: UserContext):
        """Persist user context to disk."""
        file_path = os.path.join(self.storage_dir, f"{context.user_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(context), f, indent=2)
        except Exception as e:
            print(f"Failed to save user {context.user_id}: {e}")
