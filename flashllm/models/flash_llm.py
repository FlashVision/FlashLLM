"""FlashLLM — Main model wrapper for HuggingFace causal language models."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from flashllm.utils.logger import get_logger
from flashllm.utils.model_utils import count_parameters, estimate_memory

logger = get_logger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


class FlashLLM:
    """Unified wrapper around HuggingFace causal language models.

    Provides a consistent interface for loading, configuring, and using
    any AutoModelForCausalLM-compatible model.

    Args:
        model_id: HuggingFace model ID or local path.
        torch_dtype: Data type ("float32", "float16", "bfloat16", "auto").
        device_map: Device placement strategy ("auto", "cuda", "cpu").
        trust_remote_code: Whether to trust remote model code.
        attn_implementation: Attention implementation ("flash_attention_2", "eager", "sdpa").
        load_in_4bit: Load with 4-bit quantization.
        load_in_8bit: Load with 8-bit quantization.

    Example:
        >>> model = FlashLLM("meta-llama/Llama-3.1-8B-Instruct")
        >>> response = model.generate("What is AI?")
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_id = model_id
        self._device_map = device_map

        dtype = DTYPE_MAP.get(torch_dtype, "auto")

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                    bnb_4bit_use_double_quant=True if load_in_4bit else False,
                )
                model_kwargs["quantization_config"] = quant_config
            except ImportError:
                raise ImportError("Quantized loading requires: pip install bitsandbytes")

        logger.info(f"Loading model: {model_id}")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        params = count_parameters(self.model)
        logger.info(f"Model loaded: {params['total']:,} params ({params['total_gb']:.2f} GB)")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            do_sample: Whether to use sampling.

        Returns:
            Generated text (excluding prompt).
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: Conversation history [{"role": ..., "content": ...}].
            max_new_tokens: Max tokens for response.
            temperature: Sampling temperature.

        Returns:
            Assistant response.
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)

    def save(self, path: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to: {path}")

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return next(self.model.parameters()).dtype

    def __repr__(self) -> str:
        params = count_parameters(self.model)
        return f"FlashLLM(model_id='{self.model_id}', params={params['total']:,}, dtype={self.dtype})"
