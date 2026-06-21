"""Mixture-of-Experts (MoE) layer for LLMs.

Implements a sparse MoE layer with top-k routing, expert parallelism support,
and auxiliary load-balancing loss to prevent expert collapse.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.nn import SwiGLU
from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class TopKRouter(nn.Module):
    """Learned top-k routing network for MoE.

    Computes gating scores for each token and selects the top-k experts.
    Includes auxiliary load-balancing loss to encourage uniform expert usage.

    Args:
        hidden_size: Input dimension.
        num_experts: Total number of experts.
        top_k: Number of experts activated per token.
        jitter_noise: Noise added during training for exploration.
        normalize_weights: Whether to normalize the gating weights to sum to 1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        normalize_weights: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.normalize_weights = normalize_weights
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Args:
            x: Input tensor of shape (batch * seq_len, hidden_size).

        Returns:
            Tuple of:
                - weights: Gating weights for selected experts (num_tokens, top_k).
                - indices: Selected expert indices (num_tokens, top_k).
                - load_balancing_loss: Auxiliary loss scalar.
        """
        if self.training and self.jitter_noise > 0:
            x = x + torch.randn_like(x) * self.jitter_noise

        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)

        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        if self.normalize_weights:
            top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-9)

        load_balancing_loss = self._compute_load_balancing_loss(scores, top_k_indices)

        return top_k_scores, top_k_indices, load_balancing_loss

    def _compute_load_balancing_loss(
        self,
        scores: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary load-balancing loss.

        Encourages uniform distribution of tokens across experts.
        Loss = num_experts * sum(fraction_i * probability_i) for all experts,
        where fraction_i is the fraction of tokens routed to expert i.
        """
        num_tokens = scores.shape[0]

        one_hot = F.one_hot(selected_indices, self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=0).sum(dim=0)
        fraction = tokens_per_expert / (num_tokens * self.top_k + 1e-9)

        avg_probs = scores.mean(dim=0)

        loss = self.num_experts * (fraction * avg_probs).sum()
        return loss


class Expert(nn.Module):
    """Single expert network — a SwiGLU FFN.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoELayer(nn.Module):
    """Mixture-of-Experts layer.

    Replaces the standard FFN in a transformer block with a sparse MoE,
    where only top-k experts are activated per token.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: Expert FFN intermediate dimension.
        num_experts: Total number of experts.
        top_k: Number of active experts per token.
        jitter_noise: Router jitter noise for training.
        load_balance_weight: Weight of the auxiliary load-balancing loss.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_experts: int = 8,
        top_k: int = 2,
        jitter_noise: float = 0.01,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        self.router = TopKRouter(
            hidden_size, num_experts, top_k, jitter_noise=jitter_noise,
        )
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of (output, load_balancing_loss).
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)

        weights, indices, lb_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            mask = (indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)

            expert_weights = torch.zeros(mask.sum(), device=x.device)
            for k in range(self.top_k):
                k_mask = indices[mask, k] == expert_idx
                expert_weights[k_mask] = weights[mask, k][k_mask]

            output[mask] += expert_output * expert_weights.unsqueeze(-1)

        output = output.reshape(batch_size, seq_len, hidden_size)
        return output, lb_loss * self.load_balance_weight


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN replacing dense FFN.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (GQA).
        intermediate_size: Expert intermediate dimension.
        num_experts: Number of experts.
        top_k: Active experts per token.
        dropout: Dropout rate.
        norm_eps: Normalization epsilon.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: int = 14336,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        from flashllm.models.architecture.attention import MultiHeadAttention
        from flashllm.nn import RMSNorm

        self.attention_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[tuple], torch.Tensor]:
        residual = x
        x = self.attention_norm(x)
        attn_output, present_kv = self.attention(
            x, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.ffn_norm(x)
        moe_output, lb_loss = self.moe(x)
        x = residual + self.dropout(moe_output)

        return x, present_kv, lb_loss
