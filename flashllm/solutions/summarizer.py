"""Text summarization solution."""

from typing import Optional


class Summarizer:
    """Document summarization with configurable length and style.

    Args:
        model_id: HuggingFace model ID.
        device: Device for inference.
        max_tokens: Maximum tokens in the summary.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_tokens: int = 256,
        temperature: float = 0.3,
    ):
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from flashllm.engine.predictor import Predictor

            self._predictor = Predictor(model_id=self.model_id, device=self.device)
        return self._predictor

    def summarize(self, text: str, max_length: Optional[int] = None, style: str = "concise") -> str:
        """Summarize a document.

        Args:
            text: The document text to summarize.
            max_length: Maximum summary length in words.
            style: Summary style ("concise", "detailed", "bullet_points").
        """
        length_hint = f" in approximately {max_length} words" if max_length else ""
        prompt = f"Summarize the following text{length_hint}.\n\nText:\n{text}\n\nSummary:"
        return self.predictor.generate(prompt, max_tokens=max_length or self.max_tokens, temperature=self.temperature)
