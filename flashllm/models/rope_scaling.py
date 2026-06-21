"""RoPE scaling methods for context window extension.

Implements YaRN, Dynamic NTK, and linear scaling strategies that allow
models to handle sequences longer than their training context length.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class LinearScaledRoPE(nn.Module):
    """Linear RoPE scaling — simply scales positions by a factor.

    Divides position indices by the scaling factor, effectively
    compressing the positional encoding space.

    Args:
        dim: Head dimension (must be even).
        max_position: Maximum position for precomputation.
        base: RoPE base frequency.
        scaling_factor: Linear scaling factor (e.g., 2.0 for 2x context).
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class DynamicNTKScaledRoPE(nn.Module):
    """Dynamic NTK-aware RoPE scaling.

    Adjusts the base frequency dynamically based on the sequence length,
    preserving low-frequency components while extending high-frequency ones.

    Args:
        dim: Head dimension (must be even).
        max_position: Original training max position.
        base: Original RoPE base frequency.
        scaling_factor: Target scaling factor.
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            if seq_len > self.max_position:
                new_base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            else:
                inv_freq = self.inv_freq

            t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class YaRNScaledRoPE(nn.Module):
    """YaRN (Yet another RoPE extensioN) scaling.

    Combines NTK-aware scaling with attention temperature correction
    and a smooth interpolation between original and scaled frequencies.

    Reference: https://arxiv.org/abs/2309.00071

    Args:
        dim: Head dimension (must be even).
        max_position: Original training max position.
        base: RoPE base frequency.
        scaling_factor: Target scaling factor.
        beta_fast: Fast frequency boundary.
        beta_slow: Slow frequency boundary.
        attn_factor: Attention scaling factor for length extrapolation.
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self._attn_scale = 1.0

        self._compute_yarn_freqs()

    def _compute_yarn_freqs(self):
        """Compute YaRN-modified inverse frequencies."""
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (torch.arange(0, dim, 2).float() / dim))

        low = _yarn_find_correction_dim(self.beta_fast, dim, self.base, self.max_position)
        high = _yarn_find_correction_dim(self.beta_slow, dim, self.base, self.max_position)

        low = max(low, 0)
        high = min(high, dim // 2 - 1)

        inv_freq_mask = torch.ones(dim // 2)
        if low != high:
            inv_freq_mask[int(low) : int(high) + 1] = torch.linspace(1, 0, int(high) - int(low) + 1)

        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._attn_scale = 0.1 * math.log(self.scaling_factor) + 1.0 if self.scaling_factor > 1 else 1.0

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos() * self._attn_scale
            self._sin_cached = emb.sin() * self._attn_scale
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    @property
    def attention_scale(self) -> float:
        return self._attn_scale


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position: int,
) -> float:
    """Find the RoPE dimension for a given number of rotations."""
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def get_rope_scaling(
    method: str,
    dim: int,
    max_position: int = 8192,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    **kwargs,
) -> nn.Module:
    """Factory function for RoPE scaling methods.

    Args:
        method: Scaling method ("linear", "dynamic_ntk", "yarn").
        dim: Head dimension.
        max_position: Original max position.
        base: RoPE base frequency.
        scaling_factor: Context extension factor.

    Returns:
        RoPE scaling module.
    """
    methods = {
        "linear": LinearScaledRoPE,
        "dynamic_ntk": DynamicNTKScaledRoPE,
        "dynamic": DynamicNTKScaledRoPE,
        "yarn": YaRNScaledRoPE,
    }

    if method not in methods:
        available = ", ".join(methods.keys())
        raise ValueError(f"Unknown RoPE scaling: '{method}'. Available: [{available}]")

    return methods[method](
        dim=dim,
        max_position=max_position,
        base=base,
        scaling_factor=scaling_factor,
        **kwargs,
    )
