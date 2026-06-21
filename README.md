<p align="center">
  <img src="assets/logo.png" width="200" alt="FlashLLM Logo">
</p>

<h1 align="center">FlashLLM</h1>

<p align="center">
  <a href="https://pypi.org/project/flashllm/"><img src="https://img.shields.io/pypi/v/flashllm?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/FlashVision/FlashLLM/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashLLM/ci.yml?logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/LoRA-Fine_Tuning-ff6b6b" alt="LoRA">
  <img src="https://img.shields.io/badge/RLHF-DPO%2FPPO-9b59b6" alt="RLHF">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Lightweight LLM fine-tuning, inference, quantization, RLHF, and serving</b>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#models">Models</a> •
  <a href="#fine-tuning">Fine-Tuning</a> •
  <a href="#rlhf">RLHF</a> •
  <a href="#serving">Serving</a> •
  <a href="#examples">Examples</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is FlashLLM?

FlashLLM is an end-to-end framework for working with Large Language Models — from fine-tuning to deployment. It provides a `pip`-installable Python package with a CLI, a high-level Python API, and built-in solutions for chat, summarization, and more.

```bash
pip install -e .
flashllm train --config configs/flashllm_finetune_lora.yaml
flashllm generate --model gpt2 --prompt "The future of AI is"
flashllm serve --model ./my_model --port 8000
```

---

## Installation

### pip (recommended)

```bash
pip install flashllm

# With all extras (RLHF, serving, quantization, analytics)
pip install "flashllm[all]"
```

### From source (for development)

```bash
git clone https://github.com/FlashVision/FlashLLM.git
cd FlashLLM
pip install -e ".[all]"
```

### Optional extras

```bash
pip install -e ".[rlhf]"           # TRL for RLHF
pip install -e ".[serving]"        # FastAPI server
pip install -e ".[export]"         # ONNX export
pip install -e ".[quantization]"   # GPTQ + AWQ
pip install -e ".[analytics]"      # Benchmarking, plots
pip install -e ".[all]"            # Everything
```

### Verify installation

```bash
flashllm check       # runs full health check
flashllm settings    # shows Python, PyTorch, CUDA, GPU info
flashllm version     # prints version
```

---

## Usage

### Python API

```python
from flashllm import FlashLLM, Trainer, Predictor, Exporter

# Generate text
llm = FlashLLM.from_pretrained("gpt2", device="cuda")
output = llm.generate("The meaning of life is", max_new_tokens=100)
print(output)

# Fine-tune with LoRA
trainer = Trainer(
    model_name="gpt2",
    epochs=3,
    use_lora=True,
    device="cuda",
)
trainer.train()

# Export to ONNX
exporter = Exporter(model_path="workspace/train/best_model")
exporter.export(output="model.onnx")
```

### CLI

```bash
# Train
flashllm train --config configs/flashllm_finetune_lora.yaml

# Generate
flashllm generate --model gpt2 --prompt "Hello, world!" --temperature 0.7

# Serve
flashllm serve --model ./my_model --port 8000

# Interactive chat
flashllm chat --model ./my_model

# Export
flashllm export --model ./my_model --output model.onnx

# Benchmark
flashllm benchmark --model gpt2 --num-tokens 128
```

---

## Models

FlashLLM supports any HuggingFace causal language model:

| Model Family | Examples | Parameters | Notes |
|-------------|----------|-----------|-------|
| **GPT-2** | gpt2, gpt2-medium | 117M-774M | Good for testing |
| **LLaMA** | Llama-2-7b, Llama-3-8b | 7B-70B | Meta's open models |
| **Mistral** | Mistral-7B-v0.1 | 7B | Efficient GQA |
| **Phi** | phi-2, phi-3-mini | 2.7B-3.8B | Compact & capable |
| **Gemma** | gemma-2b, gemma-7b | 2B-7B | Google's lightweight |
| **Qwen** | Qwen2-7B | 7B | Multilingual |

```python
from flashllm import FlashLLM

# Any HuggingFace model works
llm = FlashLLM.from_pretrained("mistralai/Mistral-7B-v0.1", device="cuda")
llm = FlashLLM.from_pretrained("microsoft/phi-2", load_in_4bit=True)
```

---

## Fine-Tuning

### LoRA Fine-Tuning (recommended)

Parameter-efficient — freeze base, train only low-rank adapters:

```python
from flashllm import Trainer
from flashllm.cfg import get_config

config = get_config()
config.train.use_lora = True
config.train.lora_rank = 16
config.train.lora_alpha = 32.0
config.train.learning_rate = 2e-4
config.data.train_path = "data/train.jsonl"

trainer = Trainer(config=config, model_name="gpt2", device="cuda")
trainer.train()
```

### QLoRA (4-bit quantized base + LoRA)

```python
config.train.use_qlora = True  # 4-bit base weights + full-precision LoRA
```

### Full Fine-Tuning

```bash
flashllm train --model gpt2 --epochs 3 --batch-size 2 --lr 2e-5
```

### Config-driven Training

```bash
flashllm train --config configs/flashllm_finetune_lora.yaml
flashllm train --config configs/flashllm_finetune_full.yaml
```

| Config | Description |
|--------|-------------|
| `flashllm_finetune_lora.yaml` | LoRA fine-tuning on instruction data |
| `flashllm_finetune_full.yaml` | Full SFT with all parameters |
| `flashllm_rlhf.yaml` | DPO preference alignment |
| `flashllm_quantize.yaml` | Post-training quantization |

---

## RLHF

### DPO (Direct Preference Optimization)

```python
from flashllm.rlhf import DPOTrainer

dpo = DPOTrainer(model=policy, ref_model=ref, tokenizer=tok, beta=0.1)
metrics = dpo.train_step(
    prompt="What is AI?",
    chosen_response="AI is a field of computer science...",
    rejected_response="idk",
)
```

### PPO with Reward Model

```python
from flashllm.rlhf import PPOTrainer, RewardModel

reward_model = RewardModel(base_model_name="gpt2")
ppo = PPOTrainer(policy_model=model, reward_model=reward_model, tokenizer=tok)
metrics = ppo.train_step(prompts=["Explain gravity"])
```

---

## Serving

```python
from flashllm.serving import InferenceServer

server = InferenceServer(model_path="./my_model", port=8000)
server.run()
```

Endpoints: `POST /generate`, `POST /chat`, `GET /health`, `GET /docs`

```bash
curl http://localhost:8000/generate -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_new_tokens": 100, "temperature": 0.7}'
```

---

## Generation

```python
from flashllm import FlashLLM

llm = FlashLLM.from_pretrained("gpt2", device="cuda")

# Greedy
output = llm.generate("Hello", do_sample=False)

# Sampling with temperature
output = llm.generate("Hello", temperature=0.8, top_k=50, top_p=0.9)

# With repetition penalty
output = llm.generate("Hello", repetition_penalty=1.2)
```

---

## Quantization

```python
from flashllm.models.quantization import GPTQQuantizer, AWQQuantizer

# GPTQ 4-bit quantization
gptq = GPTQQuantizer(bits=4, group_size=128)
quantized = gptq.quantize_model(model, calibration_data)

# AWQ activation-aware quantization
awq = AWQQuantizer(bits=4)
quantized = awq.quantize_model(model, calibration_data)
```

---

## Analytics

```python
from flashllm.analytics import Benchmark, Profiler

# Benchmark throughput
bench = Benchmark(model_path="gpt2", device="cuda")
results = bench.run()  # {'tokens_per_second': 142.3, 'ttft_ms': 12.5, ...}

# Profile layers
profiler = Profiler(model_path="gpt2")
profiler.run()  # prints per-layer timing breakdown
```

---

## Examples

Ready-to-run scripts in the [`examples/`](examples/) folder:

| Script | What it does |
|--------|--------------|
| `finetune_lora.py` | LoRA fine-tuning on instruction data |
| `finetune_full.py` | Full SFT with all parameters |
| `generate_text.py` | Text generation with various strategies |
| `quantize_model.py` | GPTQ and AWQ quantization |
| `serve_model.py` | Start FastAPI inference server |
| `rlhf_training.py` | DPO preference alignment |
| `benchmark_inference.py` | Measure throughput and latency |

```bash
cd examples
python generate_text.py
python benchmark_inference.py
```

---

## Project Structure

```
FlashLLM/
├── flashllm/                  # Main package (pip install -e .)
│   ├── __init__.py            # Public API
│   ├── cli.py                 # CLI entry point (flashllm command)
│   ├── registry.py            # Pluggable component registry
│   ├── cfg/                   # Configuration + YAML loading
│   ├── data/                  # Datasets, tokenizers, templates
│   ├── engine/                # Trainer, Validator, Predictor, Exporter
│   ├── models/                # FlashLLM wrapper, LoRA, architecture, quantization
│   ├── losses/                # Cross-entropy, RLHF losses
│   ├── nn/                    # RMSNorm, SwiGLU, RotaryEmbedding
│   ├── rlhf/                  # RewardModel, PPOTrainer, DPOTrainer
│   ├── serving/               # FastAPI server, inference engine, chat
│   ├── generation/            # Sampler, BeamSearch
│   ├── solutions/             # ChatAssistant, TextSummarizer
│   ├── analytics/             # Benchmark, Profiler, plots
│   └── utils/                 # Checkpoint, logger, metrics
├── configs/                   # YAML configs (pick & train)
├── examples/                  # Ready-to-run example scripts
├── tests/                     # Unit tests (pytest)
├── docs/                      # Documentation
├── docker/                    # Dockerfile + docker-compose
├── pyproject.toml             # Package configuration
├── CONTRIBUTING.md            # How to contribute
├── CHANGELOG.md               # Version history
└── LICENSE                    # MIT
```

---

## Docker

```bash
# Build
docker build -t flashllm -f docker/Dockerfile .

# Run inference
docker run --gpus all flashllm generate --model gpt2 --prompt "Hello!"

# Start server
docker run --gpus all -p 8000:8000 flashllm serve --model gpt2

# docker-compose
cd docker && docker compose up
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FlashVision/FlashLLM.git
cd FlashLLM
pip install -e ".[dev,all]"
ruff check flashllm/
pytest tests/ -v
flashllm check
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/FlashVision/FlashLLM">
    <b>FlashVision</b>
  </a>
  — Open-source lightweight AI
</p>
