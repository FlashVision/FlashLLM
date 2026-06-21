"""Transformer building blocks for LLMs."""

from typing import Optional

import torch
import torch.nn as nn

from flashllm.models.architecture.attention import MultiHeadAttention
from flashllm.nn import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Standard transformer decoder block with pre-norm architecture.

    Components:
        - RMSNorm → Multi-Head Attention → Residual
        - RMSNorm → SwiGLU FFN → Residual

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA). Defaults to num_heads.
        intermediate_size: FFN intermediate dimension.
        dropout: Dropout probability.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: int = 11008,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = SwiGLU(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Position indices for rotary embeddings.
            past_key_value: Cached KV for incremental decoding.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of (output, present_key_value).
        """
        residual = x
        x = self.attention_norm(x)
        attn_output, present_kv = self.attention(
            x, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))

        return x, present_kv
