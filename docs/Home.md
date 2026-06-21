# FlashLLM Documentation

Welcome to the FlashLLM documentation — a lightweight LLM inference and fine-tuning framework.

## Contents

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Setup guide for all platforms |
| [Quick Start](Quick-Start.md) | Get running in 5 minutes |
| [Models](Models.md) | Supported models and loading |
| [Training](Training.md) | SFT, LoRA, QLoRA training |
| [LoRA Fine-Tuning](LoRA-Fine-Tuning.md) | Parameter-efficient fine-tuning |
| [Quantization](Quantization.md) | GPTQ, AWQ, bitsandbytes |
| [FAQ](FAQ.md) | Frequently asked questions |

## Quick Links

- **GitHub**: [FlashVision/FlashLLM](https://github.com/FlashVision/FlashLLM)
- **PyPI**: [flashllm](https://pypi.org/project/flashllm/)
- **Issues**: [Report a bug](https://github.com/FlashVision/FlashLLM/issues)

## Architecture

FlashLLM wraps HuggingFace Transformers with:
- A streamlined Python API (`FlashLLM`, `Trainer`, `Predictor`)
- CLI for common tasks (`flashllm chat`, `flashllm train`)
- Built-in solutions (Chatbot, Summarizer, CodeAssistant, RAG)
- Efficient generation (KV cache, streaming, sampling strategies)
- Parameter-efficient training (LoRA, QLoRA, DPO)
