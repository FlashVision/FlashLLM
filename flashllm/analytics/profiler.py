"""Model profiling utilities — memory, compute, per-layer analysis."""

from typing import Dict, List

import torch
import torch.nn as nn

from flashllm.utils.logger import get_logger
from flashllm.utils.model_utils import count_parameters

logger = get_logger(__name__)


class Profiler:
    """Profile LLM memory usage and per-layer timing.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Device for profiling.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashllm.models.flash_llm import FlashLLM
            self._model = FlashLLM(self.model_id, device_map=self.device)

    def run(self, input_length: int = 128) -> Dict:
        """Profile the model.

        Args:
            input_length: Input sequence length for profiling.

        Returns:
            Dictionary with profiling results.
        """
        self._load_model()
        model = self._model.model
        tokenizer = self._model.tokenizer

        params = count_parameters(model)
        print(f"\n{'='*60}")
        print(f"  Model Profile: {self.model_id}")
        print(f"{'='*60}")
        print(f"  Total parameters:      {params['total']:>15,}")
        print(f"  Trainable parameters:  {params['trainable']:>15,}")
        print(f"  Model size (FP16):     {params['total_gb']:.2f} GB")
        print()

        layer_info = self._profile_layers(model)
        print(f"  {'Layer':<40} {'Params':>12} {'Size (MB)':>10}")
        print(f"  {'-'*40} {'-'*12} {'-'*10}")
        for name, info in layer_info[:20]:
            print(f"  {name:<40} {info['params']:>12,} {info['size_mb']:>10.2f}")

        if len(layer_info) > 20:
            print(f"  ... and {len(layer_info)-20} more layers")

        if self.device == "cuda":
            print()
            mem_info = self._profile_memory(model, tokenizer, input_length)
            print(f"  Peak memory (inference): {mem_info['peak_mb']:.0f} MB")
            print(f"  Model memory:            {mem_info['model_mb']:.0f} MB")

        print(f"\n{'='*60}\n")

        return {"params": params, "layers": layer_info}

    def _profile_layers(self, model: nn.Module) -> List[tuple]:
        """Get per-layer parameter counts."""
        layer_info = []
        for name, module in model.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                size_mb = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)) / (1024**2)
                layer_info.append((name, {"params": params, "size_mb": size_mb}))

        layer_info.sort(key=lambda x: x[1]["params"], reverse=True)
        return layer_info

    def _profile_memory(self, model: nn.Module, tokenizer, input_length: int) -> Dict:
        """Profile GPU memory usage."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        model_mem = torch.cuda.memory_allocated() / (1024**2)

        dummy_text = "hello " * input_length
        inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        return {"model_mb": model_mem, "peak_mb": peak_mem}
