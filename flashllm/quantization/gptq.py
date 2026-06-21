"""GPTQ quantization for LLMs."""

from pathlib import Path

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


def quantize_gptq(
    model_id: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    dataset: str = "c4",
    num_samples: int = 128,
    seq_length: int = 2048,
    output_dir: str = "quantized_model",
    device: str = "cuda",
) -> str:
    """Quantize a model using GPTQ.

    Args:
        model_id: HuggingFace model ID.
        bits: Number of quantization bits (typically 4).
        group_size: Group size for quantization.
        desc_act: Whether to use descending activation order.
        dataset: Calibration dataset name.
        num_samples: Number of calibration samples.
        seq_length: Sequence length for calibration.
        output_dir: Directory to save quantized model.
        device: Device for quantization.

    Returns:
        Path to quantized model directory.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        raise ImportError("GPTQ quantization requires: pip install 'flashllm[quantization]'")

    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantize_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=desc_act)
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config=quantize_config)

    logger.info(f"Preparing calibration data ({num_samples} samples)...")
    calibration_data = _get_calibration_data(tokenizer, num_samples, seq_length)

    logger.info(f"Quantizing to {bits}-bit with GPTQ...")
    model.quantize(calibration_data)

    logger.info(f"Saving to {output_path}...")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"GPTQ quantization complete: {output_path}")
    return str(output_path)


def _get_calibration_data(tokenizer, num_samples: int, seq_length: int) -> list:
    """Generate calibration data for GPTQ quantization."""
    texts = ["The quick brown fox jumps over the lazy dog. " * 50] * num_samples
    encodings = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=seq_length, return_tensors="pt")
        encodings.append(enc["input_ids"])
    return encodings
