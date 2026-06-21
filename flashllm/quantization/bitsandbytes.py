"""bitsandbytes quantization (4-bit/8-bit) for LLMs."""

from pathlib import Path

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


def quantize_bitsandbytes(
    model_id: str,
    bits: int = 4,
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    double_quant: bool = True,
    output_dir: str = "quantized_model",
) -> str:
    """Load and save a model with bitsandbytes quantization.

    Args:
        model_id: HuggingFace model ID.
        bits: Quantization bits (4 or 8).
        compute_dtype: Compute dtype for 4-bit.
        quant_type: 4-bit quant type ("nf4" or "fp4").
        double_quant: Enable double quantization.
        output_dir: Output directory.

    Returns:
        Path to quantized model directory.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError:
        raise ImportError("bitsandbytes quantization requires: pip install 'flashllm[quantization]'")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dt = dtype_map.get(compute_dtype, torch.bfloat16)

    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute_dt,
            bnb_4bit_quant_type=quant_type, bnb_4bit_use_double_quant=double_quant,
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"bitsandbytes supports 4-bit or 8-bit, got {bits}")

    logger.info(f"Loading model with {bits}-bit quantization: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Saving to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"bitsandbytes {bits}-bit quantization complete: {output_path}")
    return str(output_path)
