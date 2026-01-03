"""
TriviaQA loader for smart indexing.
Uses validation split - each row has question + entity_pages (gold passages) + answer.
"""
from typing import List, Optional
from datasets import load_dataset
from .base_loader import BaseLoader, QAPair


class TriviaLoader(BaseLoader):
    """Loader for TriviaQA rc validation."""
    
    @property
    def name(self) -> str:
        return "triviaqa"
    
    def load_qa_pairs(self, limit: Optional[int] = None) -> List[QAPair]:
        """Load QA pairs from rc validation split."""
        self._log(f"Loading QA pairs (limit={limit})...")
        
        ds = load_dataset("trivia_qa", "rc", split="validation",
                          cache_dir=self.cache_dir)
        
        qa_pairs = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            
            question = row.get("question", "")
            
            # Answer
            answer_data = row.get("answer", {})
            answer = answer_data.get("value", "") if isinstance(answer_data, dict) else ""
            aliases = answer_data.get("aliases", []) if isinstance(answer_data, dict) else []
            
            # Gold passages from entity_pages
            entity_pages = row.get("entity_pages", {})
            wiki_contexts = entity_pages.get("wiki_context", []) if isinstance(entity_pages, dict) else []
            
            # Filter out empty contexts
            gold_passages = [ctx for ctx in wiki_contexts if ctx and ctx.strip()]
            
            if not question or not gold_passages:
                continue
            
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                gold_passages=gold_passages,
                metadata={
                    "question_id": row.get("question_id", ""),
                    "aliases": aliases
                },
                pair_id=f"trivia_{row.get('question_id', i)}"
            ))
        
        self._log(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs
