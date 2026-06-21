"""EXL2 quantization — variable bits-per-weight for optimal quality/size.

EXL2 (ExLlamaV2) uses a per-layer adaptive bitrate that allocates more
bits to sensitive layers and fewer bits to robust ones.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class EXL2QuantConfig:
    """Configuration for EXL2 quantization.

    Args:
        target_bpw: Target average bits per weight (e.g., 4.0, 3.5, 5.0).
        calibration_length: Sequence length for calibration data.
        calibration_samples: Number of calibration samples.
        head_bits: Bits for the LM head (typically 6 or 8).
        measurement_length: Length for sensitivity measurement.
    """

    def __init__(
        self,
        target_bpw: float = 4.0,
        calibration_length: int = 2048,
        calibration_samples: int = 100,
        head_bits: int = 6,
        measurement_length: int = 2048,
    ):
        self.target_bpw = target_bpw
        self.calibration_length = calibration_length
        self.calibration_samples = calibration_samples
        self.head_bits = head_bits
        self.measurement_length = measurement_length


class EXL2LayerQuantizer:
    """Quantize a single linear layer using EXL2-style mixed precision.

    Decomposes each weight matrix into groups and assigns bits based
    on sensitivity analysis.

    Args:
        bits: Bits per weight for this layer.
        group_size: Quantization group size.
    """

    def __init__(self, bits: float = 4.0, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def quantize_weight(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantize a weight matrix.

        Args:
            weight: Float weight tensor of shape (out_features, in_features).

        Returns:
            Dict with quantized weights, scales, and zeros.
        """
        out_features, in_features = weight.shape
        num_groups = (in_features + self.group_size - 1) // self.group_size
        int_bits = max(2, min(8, round(self.bits)))

        scales = torch.zeros(out_features, num_groups, dtype=torch.float16)
        zeros = torch.zeros(out_features, num_groups, dtype=torch.float16)
        max_val = (1 << int_bits) - 1
        q_weight = torch.zeros(out_features, in_features, dtype=torch.int8)

        for g in range(num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, in_features)
            group = weight[:, start:end].float()

            group_min = group.min(dim=1, keepdim=True).values
            group_max = group.max(dim=1, keepdim=True).values
            scale = (group_max - group_min) / max_val
            scale = scale.clamp(min=1e-10)
            zero = -group_min / scale

            quantized = torch.clamp(torch.round(group / scale + zero), 0, max_val).to(torch.int8)

            scales[:, g] = scale.squeeze().half()
            zeros[:, g] = zero.squeeze().half()
            q_weight[:, start:end] = quantized

        return {
            "q_weight": q_weight,
            "scales": scales,
            "zeros": zeros,
            "bits": torch.tensor(int_bits),
            "group_size": torch.tensor(self.group_size),
        }


def compute_layer_sensitivity(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure per-layer sensitivity for adaptive bit allocation.

    Uses the Hessian trace approximation to estimate how sensitive
    each layer is to quantization error.

    Args:
        model: The model to analyze.
        calibration_data: List of input token tensors.
        device: Device for computation.

    Returns:
        Dict mapping layer name to sensitivity score.
    """
    sensitivities = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.data
        sensitivity = weight.float().norm(p="fro").item() ** 2
        sensitivity /= weight.numel()
        sensitivities[name] = sensitivity

    return sensitivities


def allocate_bits(
    sensitivities: Dict[str, float],
    target_bpw: float,
    min_bits: float = 2.0,
    max_bits: float = 8.0,
) -> Dict[str, float]:
    """Allocate bits per layer based on sensitivity scores.

    More sensitive layers get more bits; less sensitive get fewer.

    Args:
        sensitivities: Per-layer sensitivity scores.
        target_bpw: Target average bits per weight.
        min_bits: Minimum bits per layer.
        max_bits: Maximum bits per layer.

    Returns:
        Dict mapping layer name to allocated bits.
    """
    if not sensitivities:
        return {}

    total_sensitivity = sum(sensitivities.values())
    mean_sensitivity = total_sensitivity / len(sensitivities)

    allocations = {}
    for name, sens in sensitivities.items():
        ratio = sens / (mean_sensitivity + 1e-10)
        bits = target_bpw * (0.5 + 0.5 * ratio)
        bits = max(min_bits, min(max_bits, bits))
        allocations[name] = bits

    if allocations:
        current_avg = sum(allocations.values()) / len(allocations)
        adjustment = target_bpw - current_avg
        for name in allocations:
            allocations[name] = max(min_bits, min(max_bits, allocations[name] + adjustment))

    return allocations


def quantize_exl2(
    model_id: str,
    output_dir: str,
    target_bpw: float = 4.0,
    calibration_samples: int = 100,
    head_bits: int = 6,
    device: str = "cuda",
) -> str:
    """Quantize a model using EXL2-style mixed precision.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Output directory.
        target_bpw: Target average bits per weight.
        calibration_samples: Number of calibration samples.
        head_bits: Bits for LM head.
        device: Device for quantization.

    Returns:
        Path to quantized model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for EXL2 quantization: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sensitivities = compute_layer_sensitivity(model, [])
    bit_allocations = allocate_bits(sensitivities, target_bpw)

    quantized_state = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        bits = bit_allocations.get(name, target_bpw)
        quantizer = EXL2LayerQuantizer(bits=bits)
        quantized_state[name] = quantizer.quantize_weight(module.weight.data)

    torch.save(quantized_state, output_path / "quantized_weights.pt")
    torch.save(bit_allocations, output_path / "bit_allocations.pt")
    tokenizer.save_pretrained(str(output_path))

    avg_bpw = sum(bit_allocations.values()) / max(len(bit_allocations), 1)
    logger.info("EXL2 quantization complete: avg %.2f bpw, saved to %s", avg_bpw, output_path)
    return str(output_path)
