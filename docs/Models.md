# Supported Models

FlashLLM supports any HuggingFace `AutoModelForCausalLM`-compatible model.

## Tested Models

| Model | ID | Parameters | Context |
|-------|-----|-----------|---------|
| LLaMA 3.1 8B | `meta-llama/Llama-3.1-8B` | 8B | 128K |
| LLaMA 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 128K |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | 7B | 32K |
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 32K |
| Phi-3 Mini 4K | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 4K |
| Phi-3 Mini 128K | `microsoft/Phi-3-mini-128k-instruct` | 3.8B | 128K |
| Gemma 2 2B | `google/gemma-2-2b` | 2B | 8K |
| Gemma 2 9B | `google/gemma-2-9b` | 9B | 8K |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | 7B | 128K |
| Qwen 2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | 128K |

## Loading a Model

```python
from flashllm import FlashLLM

# Standard loading
model = FlashLLM("meta-llama/Llama-3.1-8B-Instruct")

# With 4-bit quantization (saves ~75% VRAM)
model = FlashLLM("meta-llama/Llama-3.1-8B-Instruct", load_in_4bit=True)

# Specific dtype
model = FlashLLM("mistralai/Mistral-7B-v0.3", torch_dtype="bfloat16")

# CPU inference
model = FlashLLM("google/gemma-2-2b", device_map="cpu")
```

## LoRA Target Modules

| Family | Default Targets |
|--------|----------------|
| LLaMA | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Mistral | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Phi | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| Gemma | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Qwen | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## Custom / Local Models

```python
model = FlashLLM("/path/to/local/model")
model = FlashLLM("workspace/lora_finetune/best")
```
