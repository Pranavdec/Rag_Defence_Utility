"""
Membership Inference Attack (MBA) Framework for RAG Systems.

Based on mask-based membership inference that strategically masks words in target
documents and evaluates the RAG system's ability to predict them.
"""

import re
import random
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class MBAFramework:
    """
    Mask-Based Attack (MBA) Framework for Membership Inference in RAG systems.
    
    The framework operates in two phases:
    1. Mask Generation: Strategically mask M words in target document
    2. Membership Inference: Query RAG and classify based on prediction accuracy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MBA Framework with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.mba_config = config.get('attack', {}).get('mba', {})
        self.data_config = config.get('data', {})
        self.retrieval_config = config.get('retrieval', {})
        
        # Attack parameters
        self.M = self.mba_config.get('M', 10)
        self.gamma = self.mba_config.get('gamma', 0.5)
        self.proxy_model_name = self.mba_config.get('proxy_model', "gpt2-xl")
        self.max_document_words = self.mba_config.get('max_document_words', 500)
        self.seed = self.mba_config.get('seed', 42)
        
        # Ingestion/Data parameters (to match pipeline)
        self.ingestion_size = self.data_config.get('ingestion_size', 1000)
        self.ingestion_seed = self.data_config.get('ingestion_seed', 42)
        self.dataset_name = self.data_config.get('dataset', 'nq')
        
        # Chunking parameters (to match pipeline)
        self.chunk_size = self.retrieval_config.get('chunk_size', 512)
        self.chunk_overlap = self.retrieval_config.get('chunk_overlap', 50)
        
        # Device settings
        self.device = "cpu" # Force CPU for GPT-2 to avoid conflict
        
        logger.info(f"Initializing MBA Framework: M={self.M}, gamma={self.gamma}")
        logger.info(f"Data Sync: dataset={self.dataset_name}, size={self.ingestion_size}, seed={self.ingestion_seed}")
        
        # Initialize proxy model
        logger.info(f"Loading proxy model ({self.proxy_model_name})...")
        self.proxy_tokenizer = GPT2Tokenizer.from_pretrained(self.proxy_model_name)
        self.proxy_model = GPT2LMHeadModel.from_pretrained(self.proxy_model_name)
        self.proxy_model.to(self.device)
        self.proxy_model.eval()
        
        # Spelling correction
        self.spelling_enabled = self.mba_config.get('enable_spelling_correction', False)
        if self.spelling_enabled:
            logger.info("Loading spelling correction model...")
            try:
                self.spell_tokenizer = AutoTokenizer.from_pretrained("oliverguhr/spelling-correction-english-base")
                self.spell_model = AutoModelForSeq2SeqLM.from_pretrained("oliverguhr/spelling-correction-english-base")
                self.spell_model.to(self.device)
                self.spell_model.eval()
            except Exception as e:
                logger.warning(f"Could not load spelling model: {e}. Disabling.")
                self.spelling_enabled = False
        else:
            logger.info("Spelling correction disabled")
            
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text exactly as the pipeline does.
        This ensures we are testing the actual units of information stored in the RAG.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks

    def load_and_prepare_data(self) -> Tuple[List[str], List[str]]:
        """
        Load dataset and split into members and non-members.
        Applies chunking to match RAG ingestion.
        
        Returns:
            Tuple[List[str], List[str]]: (member_chunks, non_member_chunks)
        """
        from ..data_loaders.nq_loader import NQLoader
        from ..data_loaders.trivia_loader import TriviaLoader
        from ..data_loaders.pubmed_loader import PubMedLoader

        # Loader Factory
        if self.dataset_name == 'nq':
            loader = NQLoader(cache_dir=self.config.get('paths', {}).get('cache', 'data/raw'))
        elif self.dataset_name == 'triviaqa':
            loader = TriviaLoader(cache_dir=self.config.get('paths', {}).get('cache', 'data/raw'))
        elif self.dataset_name == 'pubmedqa':
            loader = PubMedLoader(cache_dir=self.config.get('paths', {}).get('cache', 'data/raw'))
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        # Determine how many items to load
        # We need ingestion_size (Members) + num_non_members (Non-Members)
        # We add a buffer to ensure we have enough valid chunks
        num_non_members_target = self.mba_config.get('num_non_members', 50)
        total_needed = self.ingestion_size + num_non_members_target + 20 
        
        logger.info(f"Loading {total_needed} QA pairs from {self.dataset_name} (seed={self.ingestion_seed})...")
        qa_pairs = loader.load_qa_pairs(limit=total_needed, seed=self.ingestion_seed)
        
        if not qa_pairs:
            logger.error("No data loaded!")
            return [], []
            
        # Split into Member Candidates and Non-Member Candidates at the QA Pair level
        # The first 'ingestion_size' pairs correspond to what was ingested
        member_pairs = qa_pairs[:self.ingestion_size]
        non_member_pairs = qa_pairs[self.ingestion_size:]
        
        logger.info(f"Split: {len(member_pairs)} Member pairs, {len(non_member_pairs)} Non-Member pairs")
        
        # Process and Chunk Members
        member_chunks = []
        for pair in member_pairs:
            for passage in pair.gold_passages:
                chunks = self._chunk_text(passage)
                member_chunks.extend(chunks)
                
        # Process and Chunk Non-Members
        non_member_chunks = []
        for pair in non_member_pairs:
            for passage in pair.gold_passages:
                chunks = self._chunk_text(passage)
                non_member_chunks.extend(chunks)
                
        logger.info(f"Generated {len(member_chunks)} Member chunks, {len(non_member_chunks)} Non-Member chunks")
        return member_chunks, non_member_chunks

    def generate_attack_dataset(self) -> List[Dict]:
        """
        Generate the full attack dataset (both members and non-members).
        
        Returns:
            List of attack payloads
        """
        member_chunks, non_member_chunks = self.load_and_prepare_data()
        
        target_members = self.mba_config.get('num_members', 50)
        target_non_members = self.mba_config.get('num_non_members', 50)
        
        payloads = []
        
        # Sample Members
        if member_chunks:
            rng = random.Random(self.seed)
            selected_members = rng.sample(member_chunks, min(len(member_chunks), target_members))
            
            # Create IDs
            msg = f"Generating {len(selected_members)} MEMBER payloads"
            logger.info(msg)
            member_ids = [f"member_{i}_{self.seed}" for i in range(len(selected_members))]
            
            payloads.extend(self._create_payloads(selected_members, member_ids, is_member=True))
            
        # Sample Non-Members
        if non_member_chunks:
            rng = random.Random(self.seed + 1) # Different seed offset
            selected_non_members = rng.sample(non_member_chunks, min(len(non_member_chunks), target_non_members))
            
            msg = f"Generating {len(selected_non_members)} NON-MEMBER payloads"
            logger.info(msg)
            non_member_ids = [f"non_member_{i}_{self.seed}" for i in range(len(selected_non_members))]
            
            payloads.extend(self._create_payloads(selected_non_members, non_member_ids, is_member=False))
            
        logger.info(f"Total attack payloads generated: {len(payloads)}")
        return payloads

    # Deprecated/Legacy methods kept for compatibility or removed if not needed?
    # User asked to "edit" so I will replace the generation methods.
    # The old generate_member_payloads was called externally. Now we have a unified one.
    # I'll keep _create_payloads helper.
    
    def generate_member_payloads(self, *args, **kwargs):
        logger.warning("generate_member_payloads is deprecated. Use generate_attack_dataset()")
        return []

    def generate_non_member_payloads(self, *args, **kwargs):
        logger.warning("generate_non_member_payloads is deprecated. Use generate_attack_dataset()")
        return []

    def _create_payloads(self, documents: List[str], ids: List[str], is_member: bool) -> List[Dict]:
        """Helper to create payloads from docs."""
        payloads = []
        for i, doc in enumerate(documents):
            if not doc:
                continue
                
            doc_id = ids[i]
            
            # Generate masks
            masked_doc, answer_key = self.generate_masks(doc)
            
            if not answer_key:
                continue
                
            # Create query prompt
            prompt = f"Fill in the masked words in the following text. Provide answers in the format '[Mask_X]: answer'.\n\n{masked_doc}\n\nProvide the predicted words for each mask:"
            
            payload = {
                'id': doc_id,
                'query': prompt,
                'ground_truth': answer_key,
                'original_document': doc,
                'masked_document': masked_doc,
                'is_member': is_member
            }
            payloads.append(payload)
            
        return payloads

    def evaluate_attack_results(self, attack_payloads: List[Dict], responses: List[str]) -> Dict:
        """
        Evaluate attack results against ground truth.
        Handles both Member and Non-Member cases.
        
        Args:
            attack_payloads: List of payloads 
            responses: List of text responses from the RAG system
            
        Returns:
            Dictionary with evaluation metrics (accuracy, precision, recall, f1)
        """
        if len(attack_payloads) != len(responses):
            logger.warning(f"Number of payloads ({len(attack_payloads)}) mismatches responses ({len(responses)})")
        
        total_items = min(len(attack_payloads), len(responses))
        if total_items == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'details': []}
            
        true_positives = 0  # Member correctly predicted as Member
        false_positives = 0 # Non-Member predicted as Member
        true_negatives = 0  # Non-Member correctly predicted as Non-Member
        false_negatives = 0 # Member predicted as Non-Member
        
        results_details = []
        
        for i in range(total_items):
            payload = attack_payloads[i]
            response = responses[i]
            ground_truth = payload['ground_truth']
            is_member_actual = payload.get('is_member', True) # Default to True if missing
            
            num_masks = len(ground_truth)
            
            # Parse response
            predicted = self._parse_mask_answers(response, num_masks)
            
            # Check correctness of masks
            item_correct_masks = 0
            for mask_label, true_answer in ground_truth.items():
                pred_answer = predicted.get(mask_label, "").strip().lower()
                true_answer_clean = true_answer.strip().lower()
                
                # Check exact match or substring match
                if pred_answer == true_answer_clean or true_answer_clean in pred_answer:
                    item_correct_masks += 1
            
            item_accuracy = item_correct_masks / num_masks if num_masks > 0 else 0.0
            
            # Membership decision (Prediction: Is it a member?)
            # If accuracy > gamma, we predict it IS a member.
            is_member_prediction = item_accuracy > self.gamma
            
            # Update confusion matrix
            if is_member_actual:
                if is_member_prediction:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if is_member_prediction:
                    false_positives += 1
                else:
                    true_negatives += 1
                
            results_details.append({
                'id': payload['id'],
                'is_member_actual': is_member_actual,
                'is_member_prediction': is_member_prediction,
                'mask_accuracy': item_accuracy,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'response_snippet': response[:100] + "..." if len(response) > 100 else response
            })
            
        # Overall metrics
        total = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate average mask accuracy across all items
        mask_accuracies = [d['mask_accuracy'] for d in results_details]
        avg_mask_accuracy = sum(mask_accuracies) / len(mask_accuracies) if mask_accuracies else 0.0
        
        logger.info(f"Evaluation: Acc={accuracy:.2f}, Pre={precision:.2f}, Rec={recall:.2f}, F1={f1:.2f}")
        logger.info(f"Avg Mask Accuracy: {avg_mask_accuracy:.2f}")
        logger.info(f"Confusion Matrix: TP={true_positives}, TN={true_negatives}, FP={false_positives}, FN={false_negatives}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_mask_accuracy': avg_mask_accuracy,
            'confusion_matrix': {
                'tp': true_positives, 'tn': true_negatives, 
                'fp': false_positives, 'fn': false_negatives
            },
            'details': results_details
        }
    
    def generate_masks(self, document: str) -> Tuple[str, Dict[str, str]]:
        """
        Generate strategic masks for a document using the full MBA pipeline.
        
        Args:
            document: Original document text
            
        Returns:
            Tuple of (masked_document, answer_key)
        """
        # Truncate if too long
        words_in_doc = document.split()
        if len(words_in_doc) > self.max_document_words:
            document = ' '.join(words_in_doc[:self.max_document_words])
        
        # Extract candidate words
        candidate_words = self._extract_candidate_words(document)
        
        if not candidate_words:
            return document, {}
        
        # Calculate rank scores for candidates
        rank_scores = self._calculate_rank_scores(document, candidate_words)
        
        if not rank_scores:
            return document, {}
        
        # Select strategic masks
        masked_indices = self._select_strategic_masks(document, rank_scores)
        
        if not masked_indices:
            return document, {}
        
        # Integrate masks into document
        masked_doc, answer_key = self._integrate_masks(document, masked_indices)
        
        return masked_doc, answer_key
    
    def _extract_candidate_words(self, document: str) -> List[str]:
        """Extract candidate words from document (excluding stop words and punctuation)."""
        # Simple word tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', document)
        
        # Basic stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        candidates = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        return candidates
    
    def _extract_fragmented(self, words: List[str], document: str) -> List[str]:
        """
        Identify words that are fragmented by the tokenizer.
        
        A word is fragmented if it's split into multiple consecutive tokens
        without spaces or punctuation between them.
        """
        fragmented = []
        
        for word in words:
            tokens = self.proxy_tokenizer.encode(word, add_special_tokens=False)
            # If tokenized into multiple pieces, it's fragmented
            if len(tokens) > 1:
                fragmented.append(word)
            else:
                # Also include non-fragmented words
                fragmented.append(word)
        
        return list(set(fragmented))
    
    def _correct_spelling(self, words: List[str], document: str) -> Dict[str, str]:
        """
        Correct misspelled words using spelling correction model.
        
        Returns:
            Dictionary mapping original word to corrected word
        """
        corrected = {}
        
        for word in words:
            # Find word in document with context
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, document, re.IGNORECASE)
            
            if match:
                start_pos = match.start()
                # Get 2 preceding words for context
                context_start = max(0, start_pos - 50)
                context = document[context_start:match.end()]
                
                try:
                    inputs = self.spell_tokenizer(context, return_tensors="pt", max_length=128, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.spell_model.generate(**inputs, max_length=128)
                    
                    corrected_text = self.spell_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract the corrected version of the word
                    corrected_word_match = re.search(r'\b\w+\b', corrected_text[-len(word)*2:])
                    if corrected_word_match:
                        corrected[word] = corrected_word_match.group()
                    else:
                        corrected[word] = word
                except Exception as e:
                    logger.debug(f"Spelling correction failed for '{word}': {e}")
                    corrected[word] = word
            else:
                corrected[word] = word
        
        return corrected
    
    def _calculate_rank_scores(self, document: str, words) -> Dict[str, float]:
        """
        Calculate rank scores for words using proxy model.
        
        Higher rank = harder to predict = better candidate for masking
        """
        import time
        scores = {}
        
        # Handle both list and dict inputs
        word_list = list(words.keys()) if isinstance(words, dict) else words
        
        total_words = len(word_list)
        total_start = time.time()
        
        for idx, word in enumerate(word_list):
            word_start = time.time()
            
            try:
                # Find word position in document
                pattern = r'\b' + re.escape(word) + r'\b'
                match = re.search(pattern, document, re.IGNORECASE)
                
                if not match:
                    continue
                
                start_idx = match.start()
                context = document[:start_idx]
                
                # Tokenize context with truncation (GPT-2 max length is 1024)
                context_tokens = self.proxy_tokenizer.encode(
                    context, 
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                )
                context_tokens = context_tokens.to(self.device)
                
                # Skip if context is empty
                if context_tokens.shape[1] == 0:
                    continue
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.proxy_model(context_tokens)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                
                # Get rank of target word
                target_tokens = self.proxy_tokenizer.encode(word, add_special_tokens=False)
                
                if len(target_tokens) > 0:
                    # For multi-token words, use the first token's rank
                    target_token_id = target_tokens[0]
                    target_prob = probs[target_token_id].item()
                    
                    # Count how many tokens have higher probability
                    rank = (probs > target_prob).sum().item() + 1
                    scores[word] = float(rank)
                
                word_time = time.time() - word_start
                # logger.info(f"  Word {idx+1}/{total_words} '{word}': {word_time:.2f}s")
                
            except RuntimeError as e:
                # CUDA errors or other runtime issues
                logger.debug(f"Runtime error scoring '{word}': {e}")
                continue
            except Exception as e:
                logger.debug(f"Rank scoring failed for '{word}': {e}")
                continue
        
        total_time = time.time() - total_start
        logger.info(f"Total scoring time: {total_time:.2f}s for {len(scores)} words")
        
        return scores
    
    def _select_strategic_masks(self, document: str, rank_scores: Dict[str, float]) -> List[Tuple[str, int, int]]:
        """
        Select M words with highest rank scores, evenly distributed across document.
        
        Returns:
            List of (word, start_pos, end_pos) tuples
        """
        if len(rank_scores) == 0:
            return []
        
        # Sort words by rank score (descending)
        sorted_words = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Divide document into M sections
        doc_len = len(document)
        section_size = doc_len // self.M if self.M > 0 else doc_len
        
        selected = []
        used_positions = set()
        
        # Try to select one word per section
        for section_idx in range(self.M):
            section_start = section_idx * section_size
            section_end = section_start + section_size if section_idx < self.M - 1 else doc_len
            
            # Find best word in this section
            best_word = None
            best_score = -1
            best_pos = None
            
            for word, score in sorted_words:
                # Find word positions in this section
                pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(pattern, document, re.IGNORECASE):
                    pos = match.start()
                    if section_start <= pos < section_end:
                        # Check not adjacent to already selected masks
                        too_close = any(abs(pos - used_pos) < 20 for used_pos in used_positions)
                        
                        if not too_close and score > best_score:
                            best_word = word
                            best_score = score
                            best_pos = (match.start(), match.end())
            
            if best_word and best_pos:
                selected.append((best_word, best_pos[0], best_pos[1]))
                used_positions.add(best_pos[0])
        
        # If we didn't get enough masks, fill from remaining high-rank words
        max_iterations = len(sorted_words) * 2  # Safety limit
        iterations = 0
        while len(selected) < self.M and len(selected) < len(sorted_words) and iterations < max_iterations:
            iterations += 1
            found_one = False
            
            for word, score in sorted_words:
                if len(selected) >= self.M:
                    break
                
                # Check if already selected
                if any(w == word for w, _, _ in selected):
                    continue
                
                pattern = r'\b' + re.escape(word) + r'\b'
                match = re.search(pattern, document, re.IGNORECASE)
                
                if match:
                    pos = match.start()
                    too_close = any(abs(pos - used_pos) < 20 for used_pos in used_positions)
                    
                    if not too_close:
                        selected.append((word, match.start(), match.end()))
                        used_positions.add(match.start())
                        found_one = True
                        break
            
            # If we couldn't find any new words, break
            if not found_one:
                break
        
        # Sort by position
        selected.sort(key=lambda x: x[1])
        
        return selected
    
    def _integrate_masks(self, document: str, masked_indices: List[Tuple[str, int, int]]) -> Tuple[str, Dict[str, str]]:
        """
        Create masked document and answer key.
        
        Args:
            document: Original document
            masked_indices: List of (word, start_pos, end_pos) tuples
            
        Returns:
            Tuple of (masked_document, answer_key)
        """
        if len(masked_indices) == 0:
            return document, {}
        
        # Sort by position (descending) to replace from end to start
        masked_indices_sorted = sorted(masked_indices, key=lambda x: x[1], reverse=True)
        
        answer_key = {}
        masked_doc = document
        
        for idx, (word, start_pos, end_pos) in enumerate(reversed(masked_indices_sorted)):
            mask_label = f"[Mask_{idx + 1}]"
            answer_key[mask_label] = word
            
            # Replace word with mask
            masked_doc = masked_doc[:start_pos] + mask_label + masked_doc[end_pos:]
        
        return masked_doc, answer_key
    
    def _integrate_masks_in_full_doc(self, full_document: str, masked_indices: List[Tuple[str, int, int]], 
                                      truncated_document: str) -> Tuple[str, Dict[str, str]]:
        """
        Apply masks found in truncated document to the full original document.
        
        Args:
            full_document: Full original document
            masked_indices: List of (word, start_pos, end_pos) from truncated doc
            truncated_document: The truncated version where masks were found
            
        Returns:
            Tuple of (masked_document, answer_key)
        """
        if len(masked_indices) == 0:
            return full_document, {}
        
        answer_key = {}
        masked_doc = full_document
        
        # For each word to mask, find it in the full document and replace first occurrence
        for idx, (word, truncated_start_pos, truncated_end_pos) in enumerate(masked_indices):
            mask_label = f"[Mask_{idx + 1}]"
            answer_key[mask_label] = word
            
            # Find the word in full document (first occurrence)
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, masked_doc, re.IGNORECASE)
            
            if match:
                # Replace first occurrence
                masked_doc = masked_doc[:match.start()] + mask_label + masked_doc[match.end():]
        
        return masked_doc, answer_key
    
    def _parse_mask_answers(self, response: str, num_masks: int) -> Dict[str, str]:
        """
        Parse RAG response to extract predicted mask values.
        
        Expected format: "[Mask_X]: answer" or "Mask_X: answer"
        """
        predicted = {}
        
        # Try multiple patterns
        patterns = [
            r'\[Mask[_\s](\d+)\][:\s]+([a-zA-Z0-9\s\-]+)',
            r'Mask[_\s](\d+)[:\s]+([a-zA-Z0-9\s\-]+)',
            r'\[Mask[_\s](\d+)\][\s]*[:\-]?[\s]*([a-zA-Z0-9\s\-]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                mask_num = match.group(1)
                answer = match.group(2).strip()
                
                # Clean answer (remove trailing punctuation/newlines)
                answer = re.sub(r'[.,;:\n]+$', '', answer).strip()
                
                mask_label = f"[Mask_{mask_num}]"
                predicted[mask_label] = answer
        
        # If no matches, try to extract any words after "Mask"
        if len(predicted) == 0:
            simple_pattern = r'[Mm]ask.*?([a-zA-Z]+)'
            matches = re.findall(simple_pattern, response)
            for idx, word in enumerate(matches[:num_masks]):
                predicted[f"[Mask_{idx + 1}]"] = word
        
        return predicted
