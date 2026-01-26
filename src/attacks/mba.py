"""
Membership Inference Attack (MBA) Framework for RAG Systems.

Based on mask-based membership inference that strategically masks words in target
documents and evaluates the RAG system's ability to predict them.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
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
    
    def __init__(self, M: int = 10, gamma: float = 0.5, device: str = "auto", 
                 proxy_model: str = "gpt2-xl", enable_spelling: bool = False,
                 max_document_words: int = 500):
        """
        Initialize MBA Framework.
        
        Args:
            M: Number of masks to generate per document
            gamma: Prediction accuracy threshold for membership classification
            device: Device for model execution ('cuda', 'cpu', or 'auto')
            proxy_model: Proxy model name for difficulty scoring
            enable_spelling: Enable spelling correction (disabled by default)
            max_document_words: Maximum document length in words (default: 500)
        """
        self.M = M
        self.gamma = gamma
        self.proxy_model_name = proxy_model
        self.max_document_words = max_document_words
        
        # Force GPT-2 to CPU to avoid conflict with Llama on GPU
        # This is necessary because both models can't efficiently share GPU memory
        self.device = "cpu"
        
        logger.info(f"Initializing MBA Framework with M={M}, gamma={gamma}, device={self.device}")
        logger.info("Note: GPT-2 runs on CPU to avoid GPU memory conflict with Llama")
        
        # Initialize proxy model for mask generation
        logger.info(f"Loading proxy model ({proxy_model})...")
        self.proxy_tokenizer = GPT2Tokenizer.from_pretrained(proxy_model)
        self.proxy_model = GPT2LMHeadModel.from_pretrained(proxy_model)
        self.proxy_model.to(self.device)
        self.proxy_model.eval()
        
        # Spelling correction (disabled by default to save memory)
        self.spelling_enabled = enable_spelling
        if self.spelling_enabled:
            logger.info("Loading spelling correction model...")
            try:
                self.spell_tokenizer = AutoTokenizer.from_pretrained(
                    "oliverguhr/spelling-correction-english-base"
                )
                self.spell_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "oliverguhr/spelling-correction-english-base"
                )
                self.spell_model.to(self.device)
                self.spell_model.eval()
            except Exception as e:
                logger.warning(f"Could not load spelling model: {e}. Disabling spelling correction.")
                self.spelling_enabled = False
        else:
            logger.info("Spelling correction disabled")
        
        logger.info("MBA Framework initialized successfully")
    
    def generate_masks(self, document: str) -> Tuple[str, Dict[str, str]]:
        """
        Generate strategically placed masks in a document.
        
        Args:
            document: Target document text
            
        Returns:
            Tuple of (masked_document, answer_key) where answer_key maps mask labels to answers
        """
        # Store original document
        original_document = document
        
        # Truncate very long documents for mask generation (but we'll apply masks to full doc)
        words_in_doc = document.split()
        truncated = False
        if len(words_in_doc) > self.max_document_words:
            logger.warning(f"Document has {len(words_in_doc)} words, truncating to {self.max_document_words} for mask generation")
            document = " ".join(words_in_doc[:self.max_document_words])
            truncated = True
        
        # Step 1: Tokenize and extract candidate words
        words = self._extract_candidate_words(document)
        
        if len(words) == 0:
            logger.warning("No candidate words found in document")
            return document, {}
        
        # Pre-filter: Limit candidates to 3-5x the number of masks needed (for speed)
        # This significantly speeds up scoring while maintaining quality
        max_candidates = min(len(words), self.M * 3)  # Reduced from 5x to 3x for speed
        if len(words) > max_candidates:
            # Sample evenly across document to get diverse candidates
            import random
            step = len(words) // max_candidates
            words = [words[i] for i in range(0, len(words), max(1, step))][:max_candidates]
        
        logger.info(f"Pre-filtered to {len(words)} candidate words")
        
        # Step 2: Extract fragmented words (split by tokenizer)
        fragmented_words = self._extract_fragmented(words, document)
        logger.info(f"After deduplication: {len(fragmented_words)} unique words")
        
        # Step 3: Correct misspellings (if enabled)
        if self.spelling_enabled:
            corrected_words = self._correct_spelling(fragmented_words, document)
        else:
            corrected_words = fragmented_words
        
        # Step 4: Calculate rank scores using proxy model
        logger.info(f"Scoring {len(corrected_words)} words with GPT-2...")
        rank_scores = self._calculate_rank_scores(document, corrected_words)
        
        if len(rank_scores) == 0:
            logger.warning("No words could be scored")
            return document, {}
        
        # Step 5: Select M words with highest ranks (evenly distributed)
        masked_indices = self._select_strategic_masks(document, rank_scores)
        
        # Step 6: Create masked document and answer key
        # Apply masks to original document if truncated, otherwise use processed doc
        if truncated:
            # Apply masks to original full document by finding words in original
            masked_doc, answer_key = self._integrate_masks_in_full_doc(
                original_document, masked_indices, document
            )
        else:
            masked_doc, answer_key = self._integrate_masks(document, masked_indices)
        
        return masked_doc, answer_key
    
    def attack(self, target_document: str, rag_system) -> Dict:
        """
        Execute membership inference attack on a target document.
        
        Args:
            target_document: Document to test for membership
            rag_system: RAG system instance with run_single(question) method
            
        Returns:
            Dictionary with attack results:
                - is_member: Boolean membership prediction
                - accuracy: Prediction accuracy (0-1)
                - predicted: Dict of predicted mask values
                - ground_truth: Dict of actual mask values
                - num_masks: Number of masks used
        """
        logger.info("Generating masks for target document...")
        masked_query, answer_key = self.generate_masks(target_document)
        
        if len(answer_key) == 0:
            logger.warning("No masks generated, cannot perform attack")
            return {
                'is_member': False,
                'accuracy': 0.0,
                'predicted': {},
                'ground_truth': answer_key,
                'num_masks': 0
            }
        
        # Query RAG system
        prompt = f"Fill in the masked words in the following text. Provide answers in the format '[Mask_X]: answer'.\n\n{masked_query}\n\nProvide the predicted words for each mask:"
        
        logger.info(f"Querying RAG system with {len(answer_key)} masks...")
        try:
            response = rag_system.run_single(prompt)
            # Handle different response types
            if isinstance(response, dict):
                response_text = response.get('answer', '')
                # If answer is still a dict or non-string, try to extract or convert
                if isinstance(response_text, dict):
                    response_text = str(response_text)
                elif not isinstance(response_text, str):
                    response_text = str(response_text)
            else:
                response_text = str(response)
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                'is_member': False,
                'accuracy': 0.0,
                'predicted': {},
                'ground_truth': answer_key,
                'num_masks': len(answer_key),
                'error': str(e)
            }

        logger.info(f"RAG response: {response_text}, prompt: {prompt}")
        
        # Parse response to extract predicted mask values
        predicted = self._parse_mask_answers(response_text, len(answer_key))
        
        # Calculate accuracy
        correct = 0
        for mask_label, true_answer in answer_key.items():
            pred_answer = predicted.get(mask_label, "").strip().lower()
            true_answer_clean = true_answer.strip().lower()
            
            # Check exact match or substring match
            if pred_answer == true_answer_clean or true_answer_clean in pred_answer:
                correct += 1
        
        accuracy = correct / len(answer_key) if len(answer_key) > 0 else 0.0
        
        # Membership decision
        is_member = accuracy > self.gamma
        
        logger.info(f"Attack complete: accuracy={accuracy:.2f}, is_member={is_member}")
        
        return {
            'is_member': is_member,
            'accuracy': accuracy,
            'predicted': predicted,
            'ground_truth': answer_key,
            'num_masks': len(answer_key),
            'correct_predictions': correct
        }
    
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
