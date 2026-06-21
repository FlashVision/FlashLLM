"""Neural network building blocks for LLMs — RMSNorm, SwiGLU, RotaryEmbedding."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it skips the mean computation,
    used in LLaMA, Mistral, Gemma models.

    Args:
        hidden_size: Dimension of the input.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern LLMs.

    Combines a gated linear unit with SiLU activation:
        SwiGLU(x) = SiLU(W_gate · x) * (W_up · x)

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.
        bias: Whether to use bias in linear layers.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Args:
        dim: Embedding dimension (per head).
        max_seq_len: Maximum sequence length.
        base: Base frequency.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> tuple:
        """Compute rotary embeddings.

        Args:
            seq_len: Current sequence length.
            device: Target device.

        Returns:
            Tuple of (cos, sin) tensors.
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


__all__ = ["RMSNorm", "SwiGLU", "RotaryEmbedding"]
