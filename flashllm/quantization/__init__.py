from flashllm.quantization.gptq import quantize_gptq
from flashllm.quantization.awq import quantize_awq
from flashllm.quantization.bitsandbytes import quantize_bitsandbytes
from flashllm.quantization.exl2 import quantize_exl2, EXL2QuantConfig
from flashllm.quantization.hqq import quantize_hqq, HQQQuantizer, HQQLinear

__all__ = [
    "quantize_gptq",
    "quantize_awq",
    "quantize_bitsandbytes",
    "quantize_exl2",
    "EXL2QuantConfig",
    "quantize_hqq",
    "HQQQuantizer",
    "HQQLinear",
]


def quantize_model(model_id: str, method: str = "gptq", bits: int = 4, output_dir: str = None, **kwargs) -> str:
    """Quantize a model with the specified method.

    Args:
        model_id: HuggingFace model ID.
        method: Quantization method ("gptq", "awq", "bitsandbytes", "exl2", "hqq").
        bits: Number of bits (4 or 8).
        output_dir: Output directory for quantized model.

    Returns:
        Path to quantized model.
    """
    if output_dir is None:
        output_dir = f"{model_id.split('/')[-1]}-{method}-{bits}bit"

    if method == "gptq":
        return quantize_gptq(model_id, bits=bits, output_dir=output_dir)
    elif method == "awq":
        return quantize_awq(model_id, bits=bits, output_dir=output_dir)
    elif method == "bitsandbytes":
        return quantize_bitsandbytes(model_id, bits=bits, output_dir=output_dir)
    elif method == "exl2":
        target_bpw = kwargs.get("target_bpw", float(bits))
        return quantize_exl2(model_id, output_dir=output_dir, target_bpw=target_bpw)
    elif method == "hqq":
        return quantize_hqq(model_id, output_dir=output_dir, bits=bits)
    else:
        raise ValueError(
            f"Unsupported quantization method: {method}. Use 'gptq', 'awq', 'bitsandbytes', 'exl2', or 'hqq'."
        )
