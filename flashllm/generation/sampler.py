"""Sampling strategies for text generation."""

from typing import Optional

import torch
import torch.nn.functional as F


class Sampler:
    """Configurable token sampler with top-k, top-p, temperature, and repetition penalty.

    Args:
        temperature: Sampling temperature (1.0 = no change, <1 = more greedy, >1 = more random).
        top_k: Keep only top-k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        min_p: Minimum probability threshold.
        repetition_penalty: Penalty for previously generated tokens.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty

    def __call__(
        self,
        logits: torch.Tensor,
        generated_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample a token from logits.

        Args:
            logits: Raw logits of shape (batch, vocab_size) or (vocab_size,).
            generated_ids: Previously generated token IDs for repetition penalty.

        Returns:
            Sampled token ID(s).
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logits = self._apply_repetition_penalty(logits, generated_ids)
        logits = logits / max(self.temperature, 1e-7)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        logits = self._apply_min_p(logits)

        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        return token_id.squeeze(-1)

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if generated_ids is None or self.repetition_penalty == 1.0:
            return logits

        for i in range(logits.shape[0]):
            for token_id in generated_ids[i] if generated_ids.dim() > 1 else generated_ids:
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= self.repetition_penalty
                else:
                    logits[i, token_id] *= self.repetition_penalty
        return logits

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k <= 0 or self.top_k >= logits.shape[-1]:
            return logits

        values, _ = torch.topk(logits, self.top_k, dim=-1)
        min_values = values[..., -1:]
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)
        return logits

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > self.top_p
        sorted_logits[mask] = float("-inf")

        logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)
        return logits

    def _apply_min_p(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_p <= 0.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = max_prob * self.min_p
        logits = torch.where(probs < threshold, torch.full_like(logits, float("-inf")), logits)
        return logits

    def greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding — select the most likely token."""
        return logits.argmax(dim=-1)
