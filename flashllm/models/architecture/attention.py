"""Attention mechanisms for LLMs — MHA, GQA, MQA."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.models.architecture.embeddings import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional Grouped-Query Attention (GQA).

    When num_kv_heads < num_heads, this becomes GQA.
    When num_kv_heads == 1, this becomes MQA.
    When num_kv_heads == num_heads, this is standard MHA.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key-value heads (for GQA/MQA).
        dropout: Attention dropout.
        max_position: Maximum sequence position for RoPE.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_position: int = 8192,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_position=max_position)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q = self._apply_rotary(q, cos, sin, position_ids)
        k = self._apply_rotary(k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, present_kv

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply rotary positional embeddings."""
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            seq_len = x.shape[2]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., : x.shape[-1] // 2]
        x_pass = x[..., x.shape[-1] // 2:]
        x_rotated = torch.cat((-x_pass, x_rot), dim=-1)

        return x * cos + x_rotated * sin


class GroupedQueryAttention(MultiHeadAttention):
    """Grouped-Query Attention — a convenience alias for MHA with fewer KV heads."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, **kwargs):
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, **kwargs)
