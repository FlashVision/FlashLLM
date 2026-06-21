# LoRA Fine-Tuning

Parameter-efficient fine-tuning with Low-Rank Adaptation.

## What is LoRA?

LoRA freezes the base model weights and trains small rank-decomposition matrices (A, B) injected into attention layers. This enables fine-tuning large models with minimal GPU memory.

## Basic Usage

```python
from flashllm import FlashLLM, apply_lora, merge_lora_weights, Trainer

trainer = Trainer(
    model_id="meta-llama/Llama-3.1-8B",
    dataset="data/instructions.jsonl",
    method="lora",
    lora_rank=16,
    lora_alpha=32,
    epochs=3,
)
trainer.train()
```

## LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 16 | Rank of the low-rank matrices |
| `alpha` | 32 | Scaling factor (alpha/rank) |
| `dropout` | 0.05 | Dropout on LoRA path |
| `target_modules` | All projection layers | Which layers to apply LoRA to |

## Target Modules

```python
target_modules = [
    "q_proj",    # Query projection
    "k_proj",    # Key projection
    "v_proj",    # Value projection
    "o_proj",    # Output projection
    "gate_proj", # FFN gate
    "up_proj",   # FFN up
    "down_proj", # FFN down
]
```

## QLoRA

QLoRA loads the base model in 4-bit (NF4) and trains LoRA adapters in full precision:

```python
from flashllm import FlashLLM

model = FlashLLM("meta-llama/Llama-3.1-8B", load_in_4bit=True)
```

```bash
flashllm train --config configs/flashllm_mistral_7b_lora.yaml --device cuda
```

## Merging Weights

After training, merge LoRA back into the base model for deployment:

```python
from flashllm import merge_lora_weights

merge_lora_weights(model)
model.save_pretrained("merged_model")
```

## Memory Comparison

| Method | 8B Model VRAM | Trainable Params |
|--------|---------------|------------------|
| Full fine-tune | ~60 GB | 100% |
| LoRA (r=16) | ~18 GB | ~0.1% |
| QLoRA (r=16) | ~6 GB | ~0.1% |

## Tips

- Higher rank = more capacity but more memory
- `alpha = 2 * rank` is a good starting point
- Include `gate_proj`, `up_proj`, `down_proj` for better results
- Use QLoRA for models that don't fit in VRAM with standard LoRA
