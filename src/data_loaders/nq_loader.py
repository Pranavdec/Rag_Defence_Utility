"""
Natural Questions loader using ir_datasets for smart indexing.
Uses qrels to get gold passages and short answers for each query.
"""
import os
from typing import List, Optional

# CRITICAL: Set IR_DATASETS_HOME before importing ir_datasets
# ir_datasets reads this environment variable once on import and caches it
os.environ['IR_DATASETS_HOME'] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'ir_datasets')
)
os.makedirs(os.environ['IR_DATASETS_HOME'], exist_ok=True)

import ir_datasets
from .base_loader import BaseLoader, QAPair


class NQLoader(BaseLoader):
    """Loader for Natural Questions via ir_datasets."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        super().__init__(cache_dir)
        # IR_DATASETS_HOME is already set at module level before import
    
    @property
    def name(self) -> str:
        return "nq-corpus"
    
    def load_qa_pairs(self, limit: Optional[int] = None) -> List[QAPair]:
        """Load QA pairs using qrels to match queries to gold docs and extract answers."""
        self._log(f"Loading QA pairs (limit={limit})...")
        
        dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
        
        # Build query lookup with answers (CRITICAL: answers are in queries, not qrels)
        queries_lookup = {}
        for query in dataset.queries_iter():
            queries_lookup[query.query_id] = {
                'text': query.text,
                'answers': query.answers  # DprW100Query has answers tuple
            }
        self._log(f"Loaded {len(queries_lookup)} queries with answers")
        
        # Build qrels: query_id -> doc_ids
        qrels = {}
        all_doc_ids = set()
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = []
                qrels[qrel.query_id].append(qrel.doc_id)
                all_doc_ids.add(qrel.doc_id)
        self._log(f"Loaded {len(qrels)} query-doc mappings")
        
        # Fetch documents
        self._log(f"Fetching {len(all_doc_ids)} relevant documents...")
        docs_store = dataset.docs_store()
        docs_lookup = docs_store.get_many(all_doc_ids)
        
        # Extract text from DprW100Doc objects
        docs_lookup = {doc_id: doc.text for doc_id, doc in docs_lookup.items()}
        self._log(f"Loaded {len(docs_lookup)} documents")
        
        # Create QA pairs
        qa_pairs = []
        for query_id, doc_ids in qrels.items():
            if limit and len(qa_pairs) >= limit:
                break
            
            query_data = queries_lookup.get(query_id)
            if not query_data:
                continue
            
            question = query_data['text']
            
            # Get gold passages
            gold_passages = []
            for doc_id in doc_ids:
                if doc_id in docs_lookup:
                    gold_passages.append(docs_lookup[doc_id])
            
            if not gold_passages:
                continue
            
            # Extract answer from query.answers (tuple of strings)
            answer = " | ".join(query_data['answers']) if query_data['answers'] else ""
            
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                gold_passages=gold_passages,
                metadata={
                    "query_id": query_id,
                    "doc_ids": doc_ids,
                    "answer_count": len(query_data['answers'])
                },
                pair_id=f"nq_{query_id}"
            ))
        
        self._log(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs