"""
Generation module for LLM-based answer generation.
Supports both Hugging Face Transformers (default) and Ollama.
Provides a simple interface for text generation with RAG context.
"""
from typing import List, Optional, Dict, Any
import logging
import torch
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceGenerator:
    """Wrapper for Hugging Face Transformers LLM generation."""
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        device: str = "auto",
        shared_model=None
    ):
        """
        Initialize HuggingFace generator.
        
        Args:
            model_path: HF model ID or path
            temperature: Generation temperature (0.0 = deterministic)
            device: "cuda", "cpu", or "auto"
            shared_model: Optional pre-loaded model instance (for sharing with AV defense)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.device = device
        
        if shared_model is not None:
            # Reuse shared model from AV defense
            self.llm = shared_model
            logger.info(f"HuggingFaceGenerator using shared model: {model_path}")
        else:
            # Load new model
            from .hf_model import create_model
            
            model_config = {
                "model_info": {"provider": "llama", "name": model_path},
                "params": {
                    # "max_output_tokens": 512,
                    "temperature": temperature
                }
            }
            
            # Determine actual device
            if device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
                
            logger.info(f"HuggingFaceGenerator loading model: {model_path} on {actual_device}")
            
            try:
                self.llm = create_model(model_config, device=actual_device)
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace model: {e}")
                logger.error("Ensure you have authenticated with HuggingFace: `huggingface-cli login`")
                logger.error("And have sufficient GPU memory (or set device='cpu')")
                raise e
    
    def generate(
        self,
        question: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Generate an answer given a question and retrieved contexts.
        
        Returns:
            dict with 'answer', 'model'
        """
        # Build the prompt in chat format
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the question based on the provided context. "
                "If the context doesn't contain enough information to answer, say so clearly. "
                "Be concise and accurate."
            )
        
        user_prompt = f"""Context:
{context_str}

Question: {question}

Answer:"""
        
        # Format as chat messages (Llama Instruct format)
        # Using apply_chat_template if available
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Apply chat template
            formatted_prompt = self.llm.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using simple concatenation: {e}")
            formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize
        inputs = self.llm.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True
        ).to(self.llm.model.device)
        
        # Generate with latency tracking
        start_time = time.time()
        with torch.no_grad():
            outputs = self.llm.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=self.llm.tokenizer.eos_token_id,
                # max_new_tokens=512,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                use_cache=True,
            )
        latency_ms = (time.time() - start_time) * 1000
        
        # Decode only the new tokens
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        answer = self.llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        
        return {
            "answer": answer.strip(),
            "model": self.model_path,
            "latency_ms": latency_ms
        }
    
    def generate_simple(self, prompt: str) -> str:
        """Simple generation without context (for testing)."""
        inputs = self.llm.tokenizer(prompt, return_tensors="pt").to(self.llm.model.device)
        
        with torch.no_grad():
            outputs = self.llm.model.generate(
                inputs['input_ids'],
                pad_token_id=self.llm.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        return self.llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


class OllamaGenerator:
    """Wrapper for Ollama LLM generation."""
    
    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        logger.info(f"OllamaGenerator initialized: model={model_name}, temp={temperature}")
    
    def generate(
        self,
        question: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Generate an answer given a question and retrieved contexts.
        
        Returns:
            dict with 'answer', 'latency_ms', 'model'
        """
        import ollama
        
        # Build the prompt
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the question based on the provided context. "
                "If the context doesn't contain enough information to answer, say so clearly. "
                "Be concise and accurate."
            )
        
        user_prompt = f"""Context:
{context_str}

Question: {question}

Answer:"""
        
        start_time = time.time()
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": self.temperature}
        )
        latency_ms = (time.time() - start_time) * 1000
        
        answer = response["message"]["content"]
        
        return {
            "answer": answer,
            "model": self.model_name,
            "latency_ms": latency_ms
        }
    
    def generate_simple(self, prompt: str) -> str:
        """Simple generation without context (for testing)."""
        import ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature}
        )
        return response["message"]["content"]


def create_generator(config: Dict[str, Any], defense_manager=None):
    """
    Factory function to create appropriate generator based on config.
    Auto-migrates legacy configs and supports model sharing with AV defense.
    
    Args:
        config: Full system config dict
        defense_manager: Optional DefenseManager instance for model sharing
    
    Returns:
        Generator instance (HuggingFaceGenerator or OllamaGenerator)
    """
    llm_config = config.get("system", {}).get("llm", {})
    

    provider = llm_config.get("provider")
    provider = provider.lower()
    temperature = llm_config.get("temperature", 0.0)
    
    if provider == "ollama":
        model_name = llm_config.get("model_name", "llama3")
        logger.info(f"Using Ollama generator with model: {model_name}")
        return OllamaGenerator(model_name=model_name, temperature=temperature)
    
    elif provider == "huggingface" or provider == "hf":
        model_path = llm_config.get("model_path", "meta-llama/Llama-3.1-8B-Instruct")
        device = llm_config.get("device", "auto")
        
        # Check if we can share model with AV defense
        shared_model = None
        if defense_manager is not None:
            shared_model = defense_manager.get_shared_av_model()
            if shared_model is not None:
                logger.info("Sharing HuggingFace model instance with Attention Filtering defense")
        
        return HuggingFaceGenerator(
            model_path=model_path,
            temperature=temperature,
            device=device,
            shared_model=shared_model
        )
    
    else:
        logger.warning(f"Unknown provider '{provider}', defaulting to HuggingFace")
        model_path = llm_config.get("model_path", "meta-llama/Llama-3.1-8B-Instruct")
        device = llm_config.get("device", "auto")
        return HuggingFaceGenerator(
            model_path=model_path,
            temperature=temperature,
            device=device
        )
