from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Optional, Tuple, Union, Any, Dict

from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)


class VLLMJudgeModel(DeepEvalBaseLLM):
    """
    DeepEval judge model backed by vLLM.

    DeepEval expects a model implementing:
      - load_model()
      - generate(prompt, schema=...) -> (str|BaseModel, cost)
      - a_generate(prompt, schema=...) -> (str|BaseModel, cost)
      - get_model_name()

    When DeepEval passes a Pydantic schema, we use vLLM's
    StructuredOutputsParams(json=...) for constrained JSON decoding so the
    model is forced to produce output that matches the schema.  This mirrors
    what OllamaModel does via `format=schema.model_json_schema()`.

    We return `(text, 0)` to match DeepEval's built-in models contract.
    """

    def __init__(
        self,
        model: str,
        *,
        shared_llm=None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = 4096,
        extra_engine_args: Optional[Dict[str, Any]] = None,
    ):
        self._model_path = model
        self._shared_llm = shared_llm
        self._lock = threading.Lock()
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._max_model_len = max_model_len
        self._extra_engine_args = dict(extra_engine_args or {})

        # Cache whether StructuredOutputsParams is available (vLLM >= 0.19)
        try:
            from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
            self._has_structured_outputs = True
        except ImportError:
            self._has_structured_outputs = False
            logger.warning(
                "vLLM StructuredOutputsParams not available — falling back to "
                "prompt-engineering JSON hints for schema compliance."
            )

        super().__init__(model=model)

    def load_model(self, *args, **kwargs):
        if self._shared_llm is not None:
            return self._shared_llm
        from vllm import LLM

        engine_args: Dict[str, Any] = {
            "model": self._model_path,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "tensor_parallel_size": self._tensor_parallel_size,
        }
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            engine_args["download_dir"] = os.path.join(hf_home, "hub")
        if self._max_model_len is not None:
            engine_args["max_model_len"] = int(self._max_model_len)
        engine_args.update(self._extra_engine_args)
        return LLM(**engine_args)

    def get_model_name(self, *args, **kwargs) -> str:
        return self._model_path

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return str(prompt)

    def _build_sampling_params(self, schema: Optional[Any] = None):
        """
        Build SamplingParams.  If a Pydantic schema is given and vLLM supports
        StructuredOutputsParams, use constrained JSON decoding so the output is
        guaranteed to match the schema.  This is the same effect as Ollama's
        `format=schema.model_json_schema()`.

        Returns (SamplingParams, structured_outputs_enabled).
        """
        from vllm import SamplingParams

        base_kwargs: Dict[str, Any] = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        structured_ok = False
        if schema is not None and self._has_structured_outputs:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                json_schema = (
                    schema.model_json_schema()
                    if hasattr(schema, "model_json_schema")
                    else schema.schema()
                )
                base_kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=json.dumps(json_schema)
                )
                structured_ok = True
                logger.debug(
                    "Using StructuredOutputsParams for schema "
                    f"{schema.__name__ if hasattr(schema, '__name__') else schema}"
                )
            except Exception as exc:
                logger.warning(
                    "Could not build StructuredOutputsParams: %s. Falling back to prompt hint.",
                    exc,
                )

        return SamplingParams(**base_kwargs), structured_ok

    def _inject_schema_hint(self, prompt_text: str, schema: Any) -> str:
        """
        Fallback: append a JSON format instruction to the prompt when
        constrained decoding is not available.
        """
        try:
            json_schema = (
                schema.model_json_schema()
                if hasattr(schema, "model_json_schema")
                else schema.schema()
            )
            fields = list(json_schema.get("properties", {}).keys())
            hint = (
                "\n\nRespond ONLY with a valid JSON object. "
                f"Required fields: {fields}. "
                "Do not include any text outside the JSON object."
            )
            return prompt_text + hint
        except Exception:
            return prompt_text + "\n\nRespond ONLY with a valid JSON object."

    def generate(
        self, prompt: str, schema: Optional[Any] = None, **kwargs
    ) -> Tuple[Union[str, Any], float]:
        from vllm import SamplingParams

        text_prompt = self._prompt_to_text(prompt)

        if schema is not None and self._has_structured_outputs:
            sampling, structured_ok = self._build_sampling_params(schema)
            if not structured_ok:
                text_prompt = self._inject_schema_hint(text_prompt, schema)
        elif schema is not None:
            # Prompt-engineering fallback when StructuredOutputsParams is unavailable
            text_prompt = self._inject_schema_hint(text_prompt, schema)
            sampling = SamplingParams(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        else:
            sampling = SamplingParams(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

        # vLLM's in-process LLM is not thread-safe across concurrent calls;
        # serialize with a lock so DeepEval's async gather doesn't corrupt state.
        with self._lock:
            outputs = self.model.generate([text_prompt], sampling, use_tqdm=False)
            out_text = outputs[0].outputs[0].text

        return out_text, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[Any] = None, **kwargs
    ) -> Tuple[Union[str, Any], float]:
        return await asyncio.to_thread(self.generate, prompt, schema=schema, **kwargs)

    # -----------------------------------------------------------------------
    # DeepEval routing fix
    # -----------------------------------------------------------------------
    # When using_native_model=False (all custom DeepEvalBaseLLM subclasses),
    # DeepEval's a_generate_with_schema_and_extract does:
    #   result = await model.a_generate_with_schema(...)
    # and passes `result` directly to trimAndLoadJson, which expects a str.
    # But the default a_generate_with_schema returns whatever a_generate
    # returns — which is (text, cost) — a tuple, causing:
    #   AttributeError: 'tuple' object has no attribute 'find'
    #
    # Fix: override both *_with_schema methods to return just the text string
    # so DeepEval's non-native path gets a str it can parse as JSON.

    def generate_with_schema(self, prompt: Any, schema: Optional[Any] = None, **kwargs) -> str:
        text, _cost = self.generate(prompt, schema=schema, **kwargs)
        return text

    async def a_generate_with_schema(self, prompt: Any, schema: Optional[Any] = None, **kwargs) -> str:
        text, _cost = await self.a_generate(prompt, schema=schema, **kwargs)
        return text
