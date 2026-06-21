"""HQQ — Half-Quadratic Quantization for LLMs.

Data-free quantization method that uses half-quadratic splitting to
optimize weight quantization without requiring calibration data.

Reference: https://arxiv.org/abs/2309.15531
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class HQQQuantizer:
    """Half-Quadratic Quantization for a single weight matrix.

    Optimizes quantization parameters by alternating between:
    1. Fixing the quantized weights and solving for optimal scales/zeros
    2. Fixing scales/zeros and finding optimal quantized weights

    Args:
        bits: Number of quantization bits (2, 3, 4, 8).
        group_size: Quantization group size.
        num_iters: Number of HQQ optimization iterations.
        axis: Quantization axis (0=per-channel, 1=per-group).
        lp_norm: Norm for the optimization objective (typically 0.7).
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        num_iters: int = 20,
        axis: int = 1,
        lp_norm: float = 0.7,
    ):
        self.bits = bits
        self.group_size = group_size
        self.num_iters = num_iters
        self.axis = axis
        self.lp_norm = lp_norm
        self.max_int = 2**bits - 1

    def quantize(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantize a weight matrix using HQQ.

        Args:
            weight: Float weight tensor of shape (out_features, in_features).

        Returns:
            Dict with quantized weights, scales, zeros, and shape info.
        """
        original_shape = weight.shape
        weight = weight.float()

        if self.axis == 1:
            weight = self._reshape_for_groups(weight)

        scale, zero = self._compute_initial_params(weight)
        q_weight = self._round_and_clamp(weight, scale, zero)

        for iteration in range(self.num_iters):
            scale, zero = self._optimize_params(weight, q_weight)
            q_weight = self._round_and_clamp(weight, scale, zero)

        return {
            "q_weight": q_weight.to(torch.uint8),
            "scale": scale.half(),
            "zero": zero.half(),
            "shape": torch.tensor(original_shape),
            "bits": torch.tensor(self.bits),
            "group_size": torch.tensor(self.group_size),
        }

    def dequantize(self, quantized: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize back to float weights.

        Args:
            quantized: Output from quantize().

        Returns:
            Reconstructed float weight tensor.
        """
        q_weight = quantized["q_weight"].float()
        scale = quantized["scale"].float()
        zero = quantized["zero"].float()
        shape = tuple(quantized["shape"].tolist())

        weight = (q_weight - zero) * scale

        if self.axis == 1:
            weight = weight.reshape(shape)

        return weight

    def _reshape_for_groups(self, weight: torch.Tensor) -> torch.Tensor:
        out_features, in_features = weight.shape
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))
        return weight.reshape(-1, self.group_size)

    def _compute_initial_params(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute initial scale and zero point via min-max."""
        w_min = weight.min(dim=-1, keepdim=True).values
        w_max = weight.max(dim=-1, keepdim=True).values

        scale = (w_max - w_min) / self.max_int
        scale = scale.clamp(min=1e-10)
        zero = -w_min / scale

        return scale, zero

    def _optimize_params(
        self,
        weight: torch.Tensor,
        q_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize scale and zero using least-squares fitting."""
        q_float = q_weight.float()

        q_mean = q_float.mean(dim=-1, keepdim=True)
        w_mean = weight.mean(dim=-1, keepdim=True)

        q_centered = q_float - q_mean
        w_centered = weight - w_mean

        numerator = (w_centered * q_centered).sum(dim=-1, keepdim=True)
        denominator = (q_centered * q_centered).sum(dim=-1, keepdim=True).clamp(min=1e-10)

        scale = numerator / denominator
        scale = scale.clamp(min=1e-10)
        zero = (w_mean / scale) - q_mean + (self.max_int / 2.0)

        return scale, zero

    def _round_and_clamp(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize weights given scale and zero point."""
        q = torch.round(weight / scale + zero)
        return torch.clamp(q, 0, self.max_int)


class HQQLinear(nn.Module):
    """Linear layer with HQQ quantized weights.

    Drop-in replacement for nn.Linear that stores weights in quantized form
    and dequantizes on-the-fly during forward pass.

    Args:
        in_features: Input feature size.
        out_features: Output feature size.
        bits: Quantization bits.
        group_size: Quantization group size.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 64,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.quantizer = HQQQuantizer(bits=bits, group_size=group_size)

        self._quantized: Optional[Dict[str, torch.Tensor]] = None
        self._weight_placeholder = nn.Parameter(
            torch.empty(0),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4, group_size: int = 64) -> "HQQLinear":
        """Create an HQQLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None
        hqq = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )
        hqq.quantize_weight(linear.weight.data)
        if has_bias:
            hqq.bias = nn.Parameter(linear.bias.data.clone())
        return hqq

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize and store a weight tensor."""
        self._quantized = self.quantizer.quantize(weight)
        for key, val in self._quantized.items():
            self.register_buffer(f"hqq_{key}", val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantized is None:
            self._quantized = {k.replace("hqq_", ""): getattr(self, k) for k in dir(self) if k.startswith("hqq_")}

        weight = self.quantizer.dequantize(self._quantized).to(x.dtype)
        output = torch.nn.functional.linear(x, weight, self.bias)
        return output


def quantize_hqq(
    model_id: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 64,
    device: str = "cuda",
) -> str:
    """Quantize a model using HQQ.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Output directory.
        bits: Quantization bits.
        group_size: Group size.
        device: Device for quantization.

    Returns:
        Path to quantized model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for HQQ quantization: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 1024:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            hqq_linear = HQQLinear.from_linear(module, bits=bits, group_size=group_size)
            setattr(parent, child_name, hqq_linear)
            quantized_count += 1

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("HQQ quantization complete: %d layers @ %d-bit, saved to %s", quantized_count, bits, output_path)
    return str(output_path)
