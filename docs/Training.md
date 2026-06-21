# Training

FlashLLM supports multiple training methods for fine-tuning language models.

## Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | Instruction following |
| **LoRA** | Low-Rank Adaptation | Efficient fine-tuning |
| **QLoRA** | Quantized LoRA | Fine-tuning on limited VRAM |
| **DPO** | Direct Preference Optimization | Alignment / preference tuning |

## Supervised Fine-Tuning (SFT)

```python
from flashllm import Trainer

trainer = Trainer(
    model_id="meta-llama/Llama-3.1-8B",
    dataset="data/instructions.jsonl",
    method="sft",
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    device="cuda",
)
trainer.train()
```

## Config-driven Training

```bash
flashllm train --config configs/flashllm_llama_7b_sft.yaml --device cuda
```

## Dataset Formats

### Alpaca format
```json
{"instruction": "Summarize this text", "input": "Long text...", "output": "Summary..."}
```

### Chat format
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

### DPO format
```json
{"prompt": "Question?", "chosen": "Good answer", "rejected": "Bad answer"}
```

## Training Features

- **Gradient accumulation**: Effective larger batch sizes
- **Mixed precision (AMP)**: FP16/BF16 for speed and memory
- **Gradient checkpointing**: Trade compute for memory
- **Cosine annealing + warmup**: Standard LR schedule
- **Callbacks**: EarlyStopping, CSVLogger, WandB

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| learning_rate | 2e-5 (SFT), 2e-4 (LoRA) | LoRA uses higher LR |
| batch_size | 4 | Per device |
| gradient_accumulation | 8 | Effective batch = 32 |
| warmup_ratio | 0.03 | 3% warmup |
| weight_decay | 0.01 | AdamW |
| max_grad_norm | 1.0 | Gradient clipping |
