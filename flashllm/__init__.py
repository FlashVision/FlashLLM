"""FlashLLM — Lightweight LLM inference and fine-tuning."""

__version__ = "1.0.0"

from flashllm.models.flash_llm import FlashLLM
from flashllm.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashllm.engine.trainer import Trainer
from flashllm.engine.validator import Validator
from flashllm.engine.predictor import Predictor
from flashllm.engine.exporter import Exporter
from flashllm.cfg import get_config
from flashllm.solutions import Chatbot, Summarizer, CodeAssistant
from flashllm.analytics import Benchmark

__all__ = [
    "FlashLLM", "Trainer", "Predictor", "Validator", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "Chatbot", "Summarizer", "CodeAssistant",
    "Benchmark",
    "__version__",
]
