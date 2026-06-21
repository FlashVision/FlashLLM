"""Streaming token generation for real-time output."""

from queue import Queue
from typing import Generator

import torch


class StreamingGenerator:
    """Token-by-token streaming generator compatible with HuggingFace's TextIteratorStreamer.

    Args:
        tokenizer: Tokenizer for decoding tokens.
        skip_prompt: Whether to skip the prompt tokens in output.
        skip_special_tokens: Whether to skip special tokens in decoding.
    """

    def __init__(self, tokenizer, skip_prompt: bool = True, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self._queue: Queue = Queue()
        self._prompt_length = 0
        self._token_count = 0

    def put(self, value):
        """Called by model.generate() to add new tokens."""
        if isinstance(value, torch.Tensor):
            self._token_count += 1
            if self.skip_prompt and self._token_count <= self._prompt_length:
                return
            text = self.tokenizer.decode(value.squeeze(), skip_special_tokens=self.skip_special_tokens)
            self._queue.put(text)
        elif isinstance(value, str):
            self._queue.put(value)

    def end(self):
        """Signal that generation is complete."""
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        value = self._queue.get()
        if value is None:
            raise StopIteration
        return value


def stream_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> Generator[str, None, None]:
    """Stream generated tokens one at a time.

    Args:
        model: Language model.
        tokenizer: Tokenizer instance.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        device: Model device.

    Yields:
        Individual decoded token strings.
    """
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            if past_key_values is not None:
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            else:
                outputs = model(**inputs, use_cache=True)

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        logits = logits / max(temperature, 1e-7)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        yield token_text
