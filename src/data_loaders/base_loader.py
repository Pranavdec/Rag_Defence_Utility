"""
Base loader for smart indexing approach.
Each loader returns QAPairs: (question, answer, gold_passages)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class QAPair:
    """A question-answer pair with its gold passage(s)."""
    question: str
    answer: str
    gold_passages: List[str]  # The passages that contain the answer
    metadata: dict = field(default_factory=dict)
    pair_id: str = ""


class BaseLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name (valid for ChromaDB: 3+ chars)."""
        pass
    
    @abstractmethod
    def load_qa_pairs(self, limit: Optional[int] = None) -> List[QAPair]:
        """
        Load QA pairs with their gold passages.
        
        Args:
            limit: Maximum number of pairs to load
            
        Returns:
            List of QAPair objects
        """
        pass
    
    def _log(self, msg: str):
        self.logger.info(f"[{self.name}] {msg}")

