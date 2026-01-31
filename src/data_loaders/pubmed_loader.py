"""
PubMedQA loader for smart indexing.
Each row has question + context (gold passage) + answer.
"""
from typing import List, Optional
from datasets import load_dataset
from .base_loader import BaseLoader, QAPair


class PubMedLoader(BaseLoader):
    """Loader for PubMedQA pqa_labeled."""
    
    @property
    def name(self) -> str:
        return "pubmedqa"
    
    def load_qa_pairs(self, limit: Optional[int] = None, seed: int = 42) -> List[QAPair]:
        """Load QA pairs from pqa_labeled."""
        self._log(f"Loading QA pairs (limit={limit}, seed={seed})...")
        
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                          cache_dir=self.cache_dir)
        
        # Shuffle deterministically
        ds = ds.shuffle(seed=seed)
        
        qa_pairs = []
        for i, row in enumerate(ds):
            if limit and len(qa_pairs) >= limit:
                break
            
            question = row.get("question", "")
            answer = row.get("long_answer", "")
            
            # Context is a dict with 'contexts' key
            context_data = row.get("context", {})
            if isinstance(context_data, dict):
                contexts_list = context_data.get("contexts", [])
                # Join into single passage (or keep separate)
                gold_passages = [" ".join(contexts_list)] if contexts_list else []
            else:
                gold_passages = [str(context_data)] if context_data else []
            
            if not question or not gold_passages:
                continue
            
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                gold_passages=gold_passages,
                metadata={
                    "pubid": str(row.get("pubid", "")),
                    "final_decision": row.get("final_decision", "")
                },
                pair_id=f"pubmed_{row.get('pubid', i)}"
            ))
        
        self._log(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs
