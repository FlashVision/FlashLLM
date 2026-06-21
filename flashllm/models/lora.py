"""LoRA, QLoRA, and DoRA for parameter-efficient LLM fine-tuning."""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """LoRA-augmented linear layer.

    Adds low-rank matrices A and B such that:
        output = original_linear(x) + (x @ A @ B) * (alpha / rank)

    Args:
        original: The original nn.Linear layer to augment.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor.
        dropout: Dropout rate on the LoRA path.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original(x)
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_output + lora_output

    @property
    def weight(self):
        return self.original.weight

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original linear layer."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = self.original.weight.data + (self.lora_A @ self.lora_B).T * self.scaling
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
) -> nn.Module:
    """Apply LoRA to target modules in a model.

    Args:
        model: The base model.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        target_modules: List of module name patterns to apply LoRA to.
        dropout: Dropout for LoRA layers.

    Returns:
        Model with LoRA layers applied.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                lora_count += 1

    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    logger.info(f"LoRA applied to {lora_count} layers (rank={rank}, alpha={alpha})")
    return model


def apply_qlora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
) -> nn.Module:
    """Apply QLoRA (Quantized LoRA) to a model.

    The base model weights are kept in quantized form (4-bit NF4),
    while LoRA adapters are trained in full precision.

    Args:
        model: The base model (should already be loaded in 4-bit).
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        target_modules: Target module patterns.
        dropout: Dropout for LoRA layers.

    Returns:
        Model with QLoRA applied.
    """
    return apply_lora(model, rank=rank, alpha=alpha, target_modules=target_modules, dropout=dropout)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into the base model.

    After merging, the model can be saved/exported without LoRA overhead.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with merged weights (no LoRA layers).
    """
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            merged_linear = module.merge()
            setattr(parent, child_name, merged_linear)
            merged_count += 1

    logger.info(f"Merged {merged_count} LoRA layers into base model")
    return model
