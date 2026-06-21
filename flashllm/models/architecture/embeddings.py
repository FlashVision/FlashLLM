"""Token and positional embeddings for LLMs."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional scaling.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Embedding dimension.
        padding_idx: Padding token index.
    """

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Applies rotation-based positional encoding that encodes relative
    position information through rotation matrices.

    Args:
        dim: Head dimension (must be even).
        max_position: Maximum sequence length.
        base: Base frequency for the rotation (10000 in original paper).
    """

    def __init__(self, dim: int, max_position: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor (used only for device/dtype).
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim).
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding (for models that don't use RoPE).

    Args:
        max_position: Maximum sequence length.
        hidden_size: Embedding dimension.
    """

    def __init__(self, max_position: int = 2048, hidden_size: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(max_position, hidden_size)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(position_ids)
