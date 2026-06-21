# Quantization

Reduce model size and inference latency with post-training quantization.

## Supported Methods

| Method | Bits | Speed | Quality | Library |
|--------|------|-------|---------|---------|
| **GPTQ** | 4 | Fast | High | auto-gptq |
| **AWQ** | 4 | Fastest | High | autoawq |
| **bitsandbytes** | 4/8 | Moderate | Good | bitsandbytes |

## GPTQ

```python
from flashllm.quantization import quantize_gptq

quantize_gptq(
    "meta-llama/Llama-3.1-8B",
    bits=4,
    group_size=128,
    output_dir="llama-8b-gptq-4bit",
)
```

## AWQ

```python
from flashllm.quantization import quantize_awq

quantize_awq(
    "mistralai/Mistral-7B-v0.3",
    bits=4,
    group_size=128,
    output_dir="mistral-7b-awq-4bit",
)
```

## bitsandbytes

```python
from flashllm.quantization import quantize_bitsandbytes

quantize_bitsandbytes(
    "meta-llama/Llama-3.1-8B",
    bits=4,
    quant_type="nf4",
    output_dir="llama-8b-bnb-4bit",
)
```

## CLI

```bash
flashllm quantize --model meta-llama/Llama-3.1-8B --method gptq --bits 4
flashllm quantize --model meta-llama/Llama-3.1-8B --method awq --bits 4
flashllm quantize --model meta-llama/Llama-3.1-8B --method bitsandbytes --bits 8
```

## Loading Quantized Models

```python
from flashllm import FlashLLM

# Load on-the-fly with bitsandbytes
model = FlashLLM("meta-llama/Llama-3.1-8B", load_in_4bit=True)

# Load pre-quantized model
model = FlashLLM("llama-8b-gptq-4bit")
```

## Memory Savings

| Model | FP16 | GPTQ-4bit | AWQ-4bit |
|-------|------|-----------|----------|
| LLaMA 8B | 16 GB | ~4.5 GB | ~4.5 GB |
| Mistral 7B | 14 GB | ~4 GB | ~4 GB |
| Phi-3 Mini | 7.6 GB | ~2.5 GB | ~2.5 GB |

## Tips

- GPTQ: Best for offline quantization with calibration data
- AWQ: Fastest inference, good for deployment
- bitsandbytes: Easiest to use, no calibration needed
- Use NF4 for 4-bit bitsandbytes (better than FP4)
