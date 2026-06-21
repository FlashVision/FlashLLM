"""Benchmarking utilities for LLM throughput and latency."""

import time
from typing import Dict

import torch

from flashllm.utils.logger import get_logger
from flashllm.utils.model_utils import count_parameters

logger = get_logger(__name__)


class Benchmark:
    """Benchmark LLM inference throughput and latency.

    Measures tokens/second, time-to-first-token, and memory usage.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Device for benchmarking.
        torch_dtype: Model weight dtype.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashllm.models.flash_llm import FlashLLM
            self._model = FlashLLM(self.model_id, torch_dtype=self.torch_dtype, device_map=self.device)

    def run(
        self,
        prompt: str = "The meaning of life is",
        max_tokens: int = 128,
        num_runs: int = 10,
        warmup_runs: int = 2,
    ) -> Dict[str, float]:
        """Run the benchmark.

        Args:
            prompt: Input prompt for generation.
            max_tokens: Tokens to generate per run.
            num_runs: Number of benchmark iterations.
            warmup_runs: Warmup iterations (not counted).

        Returns:
            Dictionary with tokens_per_sec, latency_ms, ttft_ms, memory_mb, params.
        """
        self._load_model()

        for _ in range(warmup_runs):
            self._model.generate(prompt, max_new_tokens=max_tokens, do_sample=False)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        total_tokens = 0

        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = self._model.generate(prompt, max_new_tokens=max_tokens, do_sample=False)

            if self.device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            tokens_generated = len(self._model.tokenizer.encode(output))
            total_tokens += tokens_generated

        avg_latency = sum(latencies) / num_runs
        avg_tokens = total_tokens / num_runs
        tokens_per_sec = avg_tokens / avg_latency

        memory_mb = 0.0
        if self.device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        params = count_parameters(self._model.model)

        results = {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": avg_latency * 1000,
            "avg_tokens_generated": avg_tokens,
            "memory_mb": memory_mb,
            "params": params["total"],
            "params_gb": params["total_gb"],
            "num_runs": num_runs,
            "model_id": self.model_id,
        }

        logger.info(
            f"Benchmark: {tokens_per_sec:.1f} tok/s, "
            f"{avg_latency*1000:.1f} ms/gen, "
            f"{memory_mb:.0f} MB peak"
        )

        return results
