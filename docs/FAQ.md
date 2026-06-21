# FAQ

## General

### What models does FlashLLM support?

Any model compatible with HuggingFace's `AutoModelForCausalLM`. This includes LLaMA, Mistral, Phi, Gemma, Qwen, and many more. See [Models](Models.md) for tested models.

### How is FlashLLM different from HuggingFace Transformers?

FlashLLM wraps Transformers with:
- A simpler, unified API (one class for all models)
- Built-in LoRA/QLoRA without external libraries
- CLI for common tasks
- Pre-built solutions (chatbot, summarizer, RAG)
- Integrated benchmarking and profiling

### Can I use models from the HuggingFace Hub?

Yes! Just pass the model ID:
```python
from flashllm import FlashLLM
model = FlashLLM("your-org/your-model")
```

## Training

### How much VRAM do I need?

| Model Size | Full Fine-tune | LoRA | QLoRA |
|-----------|---------------|------|-------|
| 3B | ~24 GB | ~8 GB | ~4 GB |
| 7-8B | ~60 GB | ~18 GB | ~6 GB |
| 13B | ~100 GB | ~30 GB | ~10 GB |

### What dataset format should I use?

FlashLLM supports:
- **Alpaca**: `{"instruction": ..., "input": ..., "output": ...}`
- **Chat**: `{"messages": [{"role": ..., "content": ...}]}`
- **DPO**: `{"prompt": ..., "chosen": ..., "rejected": ...}`

### How do I resume training from a checkpoint?

```python
trainer = Trainer(model_id="workspace/checkpoint", ...)
```

## Inference

### How do I stream output?

```python
from flashllm import Predictor

predictor = Predictor(model_id="meta-llama/Llama-3.1-8B-Instruct")
for token in predictor.stream("Tell me a story"):
    print(token, end="", flush=True)
```

### How do I reduce inference latency?

1. Use quantization (4-bit GPTQ or AWQ)
2. Enable Flash Attention 2
3. Use KV cache (enabled by default)
4. Reduce `max_tokens` to what you actually need

### Can I use FlashLLM without a GPU?

Yes, but it will be slow for large models:
```python
model = FlashLLM("google/gemma-2-2b", device_map="cpu")
```

## Quantization

### Which quantization method should I use?

- **GPTQ**: Best quality, requires calibration data
- **AWQ**: Fastest inference speed
- **bitsandbytes**: Easiest setup, no calibration needed

### Can I fine-tune a quantized model?

Yes! Use QLoRA:
```python
model = FlashLLM("meta-llama/Llama-3.1-8B", load_in_4bit=True)
```

## Troubleshooting

### CUDA out of memory

- Use a smaller model (Phi-3 Mini, Gemma 2B)
- Enable 4-bit quantization: `load_in_4bit=True`
- Reduce batch size
- Enable gradient checkpointing

### Model generates repetitive text

Increase `repetition_penalty` (try 1.1-1.3) or lower `temperature`.

### Slow tokenizer warning

Set environment variable: `export TOKENIZERS_PARALLELISM=false`
