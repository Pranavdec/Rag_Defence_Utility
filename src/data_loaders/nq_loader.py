"""
Natural Questions loader using ir_datasets for smart indexing.
Uses qrels to get gold passages for each query.
"""
import os
from typing import List, Optional
import ir_datasets
from .base_loader import BaseLoader, QAPair


class NQLoader(BaseLoader):
    """Loader for Natural Questions via ir_datasets."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        super().__init__(cache_dir)
        # Set ir_datasets home
        os.environ['IR_DATASETS_HOME'] = os.path.abspath(
            os.path.join(cache_dir, 'ir_datasets')
        )
    
    @property
    def name(self) -> str:
        return "nq-corpus"
    
    def load_qa_pairs(self, limit: Optional[int] = None) -> List[QAPair]:
        """Load QA pairs using qrels to match queries to gold docs."""
        self._log(f"Loading QA pairs (limit={limit})...")
        
        dataset = ir_datasets.load("beir/nq")
        
        # Build doc lookup
        self._log("Building document lookup...")
        docs_lookup = {}
        for doc in dataset.docs_iter():
            docs_lookup[doc.doc_id] = doc.text
        self._log(f"Loaded {len(docs_lookup)} documents")
        
        # Build query lookup
        queries_lookup = {}
        for query in dataset.queries_iter():
            queries_lookup[query.query_id] = query.text
        self._log(f"Loaded {len(queries_lookup)} queries")
        
        # Build qrels: query_id -> list of relevant doc_ids
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = []
                qrels[qrel.query_id].append(qrel.doc_id)
        self._log(f"Loaded {len(qrels)} query-doc mappings")
        
        # Create QA pairs
        qa_pairs = []
        for query_id, doc_ids in qrels.items():
            if limit and len(qa_pairs) >= limit:
                break
            
            question = queries_lookup.get(query_id, "")
            if not question:
                continue
            
            # Get gold passages
            gold_passages = []
            for doc_id in doc_ids:
                if doc_id in docs_lookup:
                    gold_passages.append(docs_lookup[doc_id])
            
            if not gold_passages:
                continue
            
            qa_pairs.append(QAPair(
                question=question,
                answer="",  # NQ doesn't have explicit answers in BeIR format
                gold_passages=gold_passages,
                metadata={"query_id": query_id, "doc_ids": doc_ids},
                pair_id=f"nq_{query_id}"
            ))
        
        self._log(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs
