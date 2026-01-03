"""
Generation module for Ollama LLM.
Provides a simple interface for text generation with RAG context.
"""
import time
from typing import List, Optional
import ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


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
        
        # Time the generation
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
            "latency_ms": latency_ms,
            "model": self.model_name
        }
    
    def generate_simple(self, prompt: str) -> str:
        """Simple generation without context (for testing)."""
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature}
        )
        return response["message"]["content"]
