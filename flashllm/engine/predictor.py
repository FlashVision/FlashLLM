"""Predictor for text generation, chat, and completion."""

from typing import Dict, Generator, List, Optional, Union

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """High-level text generation interface.

    Wraps a FlashLLM model with convenient generation methods including
    streaming, chat templates, and configurable sampling.

    Args:
        model_id: HuggingFace model ID or path to local checkpoint.
        device: Device for inference.
        torch_dtype: Data type for model weights.
        max_tokens: Default maximum tokens to generate.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_tokens: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens

        from flashllm.models.flash_llm import FlashLLM
        flash = FlashLLM(model_id, torch_dtype=torch_dtype, device_map=device)
        self.model = flash.model
        self.tokenizer = flash.tokenizer
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to sample (False = greedy).
            num_beams: Number of beams for beam search.

        Returns:
            Generated text string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a response for a chat conversation.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            max_tokens: Maximum new tokens.
            temperature: Sampling temperature.

        Returns:
            Assistant response string.
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Yields:
            Individual token strings as they are generated.
        """
        from flashllm.generation.streaming import StreamingGenerator

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = StreamingGenerator(self.tokenizer)

        gen_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
            **kwargs,
        }

        import threading
        thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    @torch.inference_mode()
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum new tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of generated text strings.
        """
        results = []
        for prompt in prompts:
            text = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
            results.append(text)
        return results
