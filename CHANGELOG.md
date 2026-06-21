# Changelog

All notable changes to FlashLLM will be documented in this file.

## [1.0.0] — 2026-06-21

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashllm train`, `chat`, `generate`, `export`, `quantize`, `benchmark`, `check`, `settings`, `version`
- **Python API** — `FlashLLM`, `Trainer`, `Predictor`, `Exporter`, `Validator`
- **Models** — LLaMA, Mistral, Phi, Gemma, Qwen (any HuggingFace AutoModelForCausalLM)
- **LoRA fine-tuning** — standard LoRA, QLoRA, DoRA support
- **Training methods** — SFT, DPO, RLHF
- **Quantization** — GPTQ, AWQ, bitsandbytes (4-bit/8-bit)
- **Generation** — top-k, top-p, temperature, beam search, KV cache, streaming
- **Solutions** — Chatbot, Summarizer, CodeAssistant, RAG
- **Analytics** — Benchmark, Profiler, training plots
- **Export** — ONNX, GGUF formats
- **Chat templates** — Alpaca, ChatML, Llama, Mistral formats
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12)
- **Examples** — 5 runnable example scripts

### Architecture
- HuggingFace Transformers backend (AutoModelForCausalLM)
- Modular generation pipeline (sampling, beam search, KV cache)
- Registry-based component system
- Config-driven training with YAML files
