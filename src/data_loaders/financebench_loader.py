"""
FinanceBench loader for smart indexing.
Maps question, answer, and evidence fields to QAPair.
"""
from typing import List, Optional, Any
from datasets import load_dataset

from .base_loader import BaseLoader, QAPair


class FinanceBenchLoader(BaseLoader):
    """Loader for PatronusAI/financebench."""

    @property
    def name(self) -> str:
        return "financebench"

    def _extract_gold_passages(self, evidence: Any) -> List[str]:
        """
        Build gold passages from the evidence list.
        Policy: prefer evidence_text_full_page, then fallback to evidence_text.
        """
        if not isinstance(evidence, list):
            return []

        passages: List[str] = []
        seen = set()

        for item in evidence:
            if not isinstance(item, dict):
                continue

            full_page = str(item.get("evidence_text_full_page") or "").strip()
            short_text = str(item.get("evidence_text") or "").strip()

            for text in (full_page, short_text):
                if text and text not in seen:
                    passages.append(text)
                    seen.add(text)

        return passages

    def load_qa_pairs(self, limit: Optional[int] = None, seed: int = 42) -> List[QAPair]:
        """Load QA pairs from FinanceBench train split."""
        self._log(f"Loading QA pairs (limit={limit}, seed={seed})...")

        ds = load_dataset(
            "PatronusAI/financebench",
            split="train",
            cache_dir=self.cache_dir,
        )

        ds = ds.shuffle(seed=seed)

        qa_pairs: List[QAPair] = []
        for i, row in enumerate(ds):
            if limit and len(qa_pairs) >= limit:
                break

            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            evidence = row.get("evidence") or []

            gold_passages = self._extract_gold_passages(evidence)
            if not question or not gold_passages:
                continue

            evidence_doc_names = []
            evidence_page_nums = []
            if isinstance(evidence, list):
                for item in evidence:
                    if not isinstance(item, dict):
                        continue
                    doc_name = item.get("doc_name")
                    page_num = item.get("evidence_page_num")
                    if doc_name:
                        evidence_doc_names.append(str(doc_name))
                    if page_num is not None:
                        evidence_page_nums.append(page_num)

            financebench_id = str(row.get("financebench_id") or i)

            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    gold_passages=gold_passages,
                    metadata={
                        "financebench_id": financebench_id,
                        "company": row.get("company", ""),
                        "doc_name": row.get("doc_name", ""),
                        "doc_type": row.get("doc_type", ""),
                        "doc_period": row.get("doc_period"),
                        "doc_link": row.get("doc_link", ""),
                        "question_type": row.get("question_type", ""),
                        "question_reasoning": row.get("question_reasoning", ""),
                        "dataset_subset_label": row.get("dataset_subset_label", ""),
                        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
                        "evidence_doc_names": evidence_doc_names,
                        "evidence_page_nums": evidence_page_nums,
                    },
                    pair_id=f"financebench_{financebench_id}",
                )
            )

        self._log(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs