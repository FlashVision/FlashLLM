"""Model utilities — parameter counting, memory estimation."""

from typing import Dict

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with total, trainable, frozen counts and size in GB.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    bytes_per_param = next(model.parameters()).element_size() if total > 0 else 2
    total_gb = (total * bytes_per_param) / (1024**3)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_gb": total_gb,
        "trainable_pct": 100 * trainable / max(total, 1),
    }


def estimate_memory(
    num_params: int,
    dtype: str = "float16",
    batch_size: int = 1,
    seq_length: int = 2048,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """Estimate GPU memory requirements.

    Args:
        num_params: Total number of model parameters.
        dtype: Weight data type.
        batch_size: Training batch size.
        seq_length: Sequence length.
        optimizer: Optimizer type (affects state memory).
        gradient_checkpointing: Whether gradient checkpointing is used.

    Returns:
        Dictionary with memory estimates in GB.
    """
    bytes_per_dtype = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
    param_bytes = bytes_per_dtype.get(dtype, 2)

    model_memory_gb = (num_params * param_bytes) / (1024**3)

    if optimizer == "adamw":
        optimizer_memory_gb = (num_params * 8) / (1024**3)
    elif optimizer == "sgd":
        optimizer_memory_gb = (num_params * 4) / (1024**3)
    else:
        optimizer_memory_gb = (num_params * 8) / (1024**3)

    gradient_memory_gb = (num_params * param_bytes) / (1024**3)
    if gradient_checkpointing:
        gradient_memory_gb *= 0.3

    activation_memory_gb = batch_size * seq_length * num_params * 0.00001

    total_training_gb = model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
    total_inference_gb = model_memory_gb * 1.2

    return {
        "model_gb": model_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "gradient_gb": gradient_memory_gb,
        "activation_gb": activation_memory_gb,
        "total_training_gb": total_training_gb,
        "total_inference_gb": total_inference_gb,
    }


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    return dtype_map.get(dtype_str, torch.float16)
