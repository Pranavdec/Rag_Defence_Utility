from abc import ABC, abstractmethod
from typing import List, Any

class BaseAttack(ABC):
    """Abstract base class for RAG attacks."""
    
    @abstractmethod
    def generate_poisoned_corpus(self, target_size: int) -> List[Any]:
        """Generate a list of poisoned documents."""
        pass
    
    @abstractmethod
    def generate_poisoned_document(self, question: str, target_answer: str) -> str:
        """Generate a single poisoned document for a specific QA pair."""
        pass
