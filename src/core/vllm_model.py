import logging
import time
import os
from typing import List, Optional, Dict, Any

from .model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMGenerator:
    """Wrapper for vLLM generation backend."""
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None
    ):
        """
        Initialize the vLLM generator.
        
        Args:
            model_path: Model ID or path
            temperature: Generation temperature
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_model_len: Maximum context length
        """
        self.model_path = model_path
        self.temperature = temperature
        
        # vLLM (via FlashInfer) may JIT-compile CUDA kernels and requires `ninja`
        # to be discoverable in PATH. In Cursor/remote environments, the venv's
        # `env/bin` is not always present in PATH for subprocesses.
        venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "env", "bin"))
        if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

        # If the GPU is shared with other processes, the configured utilization can
        # exceed currently available free memory. Clamp utilization to what's safe
        # so vLLM can still allocate at least some KV cache blocks.
        effective_gpu_memory_utilization = gpu_memory_utilization
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if total_bytes > 0:
                    safe_util = max(0.06, (free_bytes / total_bytes) * 0.90)
                    if effective_gpu_memory_utilization > safe_util:
                        logger.warning(
                            "Clamping gpu_memory_utilization from %.3f to %.3f based on free GPU memory.",
                            effective_gpu_memory_utilization,
                            safe_util,
                        )
                        effective_gpu_memory_utilization = safe_util
        except Exception:
            pass

        import vllm
        logger.info(f"Initializing vLLM engine for {model_path}...")
        
        # Build vllm engine args
        engine_args = {
            "model": model_path,
            "gpu_memory_utilization": effective_gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
        }
        if max_model_len:
            engine_args["max_model_len"] = max_model_len
            
        self.llm = vllm.LLM(**engine_args)
        
        # Sampling parameters
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            max_tokens=512
        )
        logger.info("vLLM engine initialized successfully.")

    def format_prompt(self, question: str, contexts: List[str], system_prompt: Optional[str] = None) -> str:
        """Format the input text for the LLM."""
        if not contexts or len(contexts) == 0:
            context_str = "No relevant context was found in the knowledge base."
        else:
            context_str = "\\n\\n".join([f"Context {i+1}:\\n{ctx}" for i, ctx in enumerate(contexts)])
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the question based on the provided context. "
                "If the context doesn't contain enough information to answer, say so clearly. "
                "Be concise and accurate."
            )
        
        user_prompt = f"Context:\\n{context_str}\\n\\nQuestion: {question}\\n\\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use vllm tokenizer to apply chat template
        try:
            tokenizer = self.llm.get_tokenizer()
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception:
            return f"{system_prompt}\\n\\n{user_prompt}"

    def generate(self, question: str, contexts: List[str], system_prompt: Optional[str] = None) -> dict:
        """Sequential single generation to maintain compatibility."""
        prompt = self.format_prompt(question, contexts, system_prompt)
        
        start_time = time.time()
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        latency_ms = (time.time() - start_time) * 1000
        
        answer = outputs[0].outputs[0].text.strip()
        
        return {
            "answer": answer,
            "model": self.model_path,
            "latency_ms": latency_ms
        }
        
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate answers for a batch of fully formatted prompts."""
        logger.info(f"Running vLLM batch generation for {len(prompts)} prompts...")
        
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        
        results = []
        for output in outputs:
            results.append(output.outputs[0].text.strip())
            
        return results

    def close(self):
        """Best-effort cleanup for vLLM resources."""
        try:
            if hasattr(self, "llm") and self.llm is not None:
                # vLLM exposes shutdown in some versions; guard for compatibility.
                if hasattr(self.llm, "shutdown"):
                    self.llm.shutdown()
        except Exception:
            pass
        try:
            self.llm = None
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass