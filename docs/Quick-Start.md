# Quick Start

Get FlashLLM running in 5 minutes.

## 1. Install

```bash
pip install flashllm
```

## 2. Generate text

```python
from flashllm import FlashLLM

model = FlashLLM("microsoft/Phi-3-mini-4k-instruct")
print(model.generate("Explain machine learning in one sentence."))
```

## 3. Chat interactively

```bash
flashllm chat --model meta-llama/Llama-3.1-8B-Instruct --device cuda
```

Or via Python:

```python
from flashllm.solutions import Chatbot

bot = Chatbot(model_id="meta-llama/Llama-3.1-8B-Instruct")
print(bot.chat("What is Python?"))
print(bot.chat("How do I install it?"))
```

## 4. Fine-tune with LoRA

```python
from flashllm import Trainer

trainer = Trainer(
    model_id="meta-llama/Llama-3.1-8B",
    dataset="data/instructions.jsonl",
    method="lora",
    lora_rank=16,
    epochs=3,
    device="cuda",
)
trainer.train()
```

## 5. Quantize a model

```bash
flashllm quantize --model meta-llama/Llama-3.1-8B --method gptq --bits 4
```

## 6. Benchmark

```bash
flashllm benchmark --model meta-llama/Llama-3.1-8B-Instruct --device cuda
```

## Next steps

- [Models](Models.md) — all supported models
- [Training](Training.md) — SFT, DPO, LoRA details
- [Quantization](Quantization.md) — reduce model size
