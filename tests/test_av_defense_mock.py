import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.defenses.av_defense import AttentionFilteringDefense

class TestAVDefense(unittest.TestCase):
    
    @patch("src.defenses.av_defense.create_model")
    def test_filtering_logic(self, mock_create_model):
        # Mock LLM
        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        
        # Setup config
        config = {
            "model_path": "dummy/path",
            "top_tokens": 10,
            "max_corruptions": 1,
            "short_answer_threshold": 50
        }
        
        defense = AttentionFilteringDefense(config)
        
        # Mock query return: (result, passage_scores, total_attention_score)
        # Scenario: 3 documents. Doc 2 has high attention (bad?), Doc 1 and 3 are good.
        # Defense logic removes max score.
        # First pass scores: [10, 50, 20] -> sum 80 -> [12.5, 62.5, 25.0]
        # Sorted by attention: Doc 0 (12.5), Doc 2 (25.0), Doc 1 (62.5)
        # Second pass (on sorted): [10, 20, 50] (assuming attention follows content)
        
        # We need to simulate the query returning scores consistent with the content
        
        def query_side_effect(prompt, top_tokens=100):
            # Simple heuristic: if prompt contains "Bad", give high score
            # if prompt contains "Good", give low score
            
            # The prompt structure from wrap_prompt:
            # [1] Good Content
            # [2] Bad Content
            
            # Let's just return fixed scores based on how many docs are in prompt
            num_docs = prompt.count("[") - 1 # Rough count of [N]
            if num_docs == 3:
                return "result", [10, 50, 20], 80
            elif num_docs == 2:
                # One removed.
                return "result", [10, 20], 30
            else:
                 return "result", [], 0

        mock_llm.query.side_effect = query_side_effect
        
        docs = [
            {"content": "Good Content 1", "id": "1"},
            {"content": "Bad Content", "id": "2"},
            {"content": "Good Content 2", "id": "3"}
        ]
        
        filtered_docs = defense.post_retrieval(docs, "query")
        
        # Expectation: Max score (50, corresponding to index 1 "Bad Content") should be removed?
        # Only if max_corruptions behavior is correctly understood as "remove highest attention".
        
        # Let's trace logic:
        # Pass 1 Sort: [10, 50, 20] -> Normalized [12, 62, 25] -> Sorted Order: Doc1(12), Doc3(25), Doc2(62)
        # Pass 2 Sort: Query on [Doc1, Doc3, Doc2]. Scores returned [10, 50, 20] (mock is dumb, returns by position)
        # Wait, if mock returns by position, then:
        # P1: Doc1 scores 10, Doc2 scores 50, Doc3 scores 20.
        # Sorted: Doc1, Doc3, Doc2.
        # P2: Query([Doc1, Doc3, Doc2]) -> returns [10, 50, 20].
        # Doc1=10, Doc3=50, Doc2=20.
        # Norm: 12, 62, 25.
        # Sorted again: Doc1(12), Doc2(25), Doc3(62).
        
        # Loop starts. Content list: [Doc1, Doc2, Doc3].
        # Query([Doc1, Doc2, Doc3]) -> returns [10, 50, 20].
        # Norm: 12, 62, 25.
        # Max is 62 (index 1).
        # Pop index 1 -> Doc2 is removed.
        # Remaining: [Doc1, Doc3].
        
        # Result: [Doc1, Doc3].
        
        self.assertEqual(len(filtered_docs), 2)
        self.assertEqual(filtered_docs[0]["content"], "Good Content 1") # Original Doc 1
        # self.assertEqual(filtered_docs[1]["content"], "Good Content 2")
        # Note: ordering might change due to sorting.
        # The defense returns `contents` which is the sorted list.
        # And `post_retrieval` re-orders original docs or filters them?
        # My implementation: `filtered_docs = [d for d in documents if d.get("content") in filtered_contents]`
        # This keeps properties of original documents and their original order if iterating `documents`.
        
        self.assertEqual(filtered_docs[0]["id"], "1")
        self.assertEqual(filtered_docs[1]["id"], "3")

if __name__ == '__main__':
    unittest.main()
