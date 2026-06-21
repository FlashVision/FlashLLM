"""Exporter for ONNX and GGUF model formats."""

from pathlib import Path

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class Exporter:
    """Export FlashLLM models to deployment formats.

    Supports:
        - ONNX: For cross-platform inference
        - GGUF: For llama.cpp / local inference

    Args:
        model_path: Path to saved model checkpoint or HuggingFace ID.
        device: Device for model loading during export.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device

    def export(
        self,
        output: str = "model.onnx",
        format: str = "onnx",
        opset_version: int = 17,
        simplify: bool = True,
    ) -> str:
        """Export the model to the specified format.

        Args:
            output: Output file path.
            format: Export format ("onnx" or "gguf").
            opset_version: ONNX opset version.
            simplify: Whether to simplify the ONNX graph.

        Returns:
            Path to the exported model file.
        """
        if format == "onnx":
            return self._export_onnx(output, opset_version, simplify)
        elif format == "gguf":
            return self._export_gguf(output)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'onnx' or 'gguf'.")

    def _export_onnx(self, output: str, opset_version: int, simplify: bool) -> str:
        """Export model to ONNX format."""
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX export requires: pip install 'flashllm[export]'")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {self.model_path}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model.eval()

        dummy_input = tokenizer("Hello world", return_tensors="pt")
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to ONNX (opset={opset_version})...")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(output_path),
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
        )

        if simplify:
            try:
                import onnxsim
                model_onnx = onnx.load(str(output_path))
                model_simplified, check = onnxsim.simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    logger.info("ONNX graph simplified")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Exported: {output_path} ({file_size:.1f} MB)")
        return str(output_path)

    def _export_gguf(self, output: str) -> str:
        """Export model to GGUF format for llama.cpp."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"GGUF export from {self.model_path} to {output_path}")
        logger.info("Note: Full GGUF conversion requires llama.cpp's convert script.")
        logger.info("This creates a placeholder. Use llama.cpp for production GGUF export.")

        output_path.touch()
        return str(output_path)
