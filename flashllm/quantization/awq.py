"""AWQ (Activation-aware Weight Quantization) for LLMs."""

from pathlib import Path

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


def quantize_awq(
    model_id: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    output_dir: str = "quantized_model",
    device: str = "cuda",
) -> str:
    """Quantize a model using AWQ.

    Args:
        model_id: HuggingFace model ID.
        bits: Number of quantization bits.
        group_size: Group size for quantization.
        zero_point: Whether to use zero-point quantization.
        output_dir: Output directory.
        device: Device for quantization.

    Returns:
        Path to quantized model directory.
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("AWQ quantization requires: pip install 'flashllm[quantization]'")

    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_id}")
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quant_config = {"zero_point": zero_point, "q_group_size": group_size, "w_bit": bits, "version": "GEMM"}

    logger.info(f"Quantizing to {bits}-bit with AWQ...")
    model.quantize(tokenizer, quant_config=quant_config)

    logger.info(f"Saving to {output_path}...")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"AWQ quantization complete: {output_path}")
    return str(output_path)
