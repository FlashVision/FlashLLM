from flashllm.quantization.gptq import quantize_gptq
from flashllm.quantization.awq import quantize_awq
from flashllm.quantization.bitsandbytes import quantize_bitsandbytes

__all__ = ["quantize_gptq", "quantize_awq", "quantize_bitsandbytes"]


def quantize_model(model_id: str, method: str = "gptq", bits: int = 4, output_dir: str = None) -> str:
    """Quantize a model with the specified method.

    Args:
        model_id: HuggingFace model ID.
        method: Quantization method ("gptq", "awq", "bitsandbytes").
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
    else:
        raise ValueError(f"Unsupported quantization method: {method}. Use 'gptq', 'awq', or 'bitsandbytes'.")
