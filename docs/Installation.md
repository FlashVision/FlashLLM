# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU inference/training)

## pip (recommended)

```bash
pip install flashllm
```

## From source (development)

```bash
git clone https://github.com/FlashVision/FlashLLM.git
cd FlashLLM
pip install -e ".[all]"
```

## Optional extras

```bash
pip install -e ".[quantization]"  # GPTQ, AWQ, bitsandbytes
pip install -e ".[analytics]"    # Benchmarking, plots
pip install -e ".[export]"       # ONNX export
pip install -e ".[rag]"          # RAG (FAISS + sentence-transformers)
pip install -e ".[dev]"          # pytest, ruff, pre-commit
pip install -e ".[all]"          # Everything
```

## One-command setup

```bash
bash setup_env.sh           # auto-detect GPU
bash setup_env.sh --cpu     # force CPU-only
bash setup_env.sh --cuda 12.4  # force CUDA 12.4
```

## Verify installation

```bash
flashllm check       # health check (imports, GPU)
flashllm settings    # Python, PyTorch, CUDA, GPU info
flashllm version     # version string
```

## Docker

```bash
docker build -t flashllm -f docker/Dockerfile .
docker run --gpus all flashllm check
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: transformers` | `pip install transformers` |
| CUDA out of memory | Use `--load-in-4bit` or a smaller model |
| Flash Attention not available | Install `flash-attn` or use `--attn-implementation eager` |
| Tokenizer warnings | Set `TOKENIZERS_PARALLELISM=false` |
