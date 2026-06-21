"""Language modeling head for causal LMs."""

import torch
import torch.nn as nn

from flashllm.nn import RMSNorm


class LanguageModelHead(nn.Module):
    """Language model head that projects hidden states to vocabulary logits.

    Args:
        hidden_size: Model hidden dimension.
        vocab_size: Vocabulary size.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(self, hidden_size: int = 4096, vocab_size: int = 32000, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits."""
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def tie_embedding_weights(self, embedding: nn.Embedding):
        """Tie the LM head weights with the token embedding."""
        self.lm_head.weight = embedding.weight
