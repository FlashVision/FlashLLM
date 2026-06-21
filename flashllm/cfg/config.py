"""Configuration for FlashLLM."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_id: str = "meta-llama/Llama-3.1-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    max_seq_length: int = 2048
    trust_remote_code: bool = False
    device_map: str = "auto"


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = "data/instructions.jsonl"
    dataset_format: str = "alpaca"
    template: str = "llama"
    val_split: float = 0.05
    num_workers: int = 4
    max_samples: Optional[int] = None


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    method: str = "sft"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    save_dir: str = "workspace/train"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10
    amp: bool = True
    gradient_checkpointing: bool = True
    resume: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DPOConfig:
    """DPO training configuration."""
    beta: float = 0.1
    loss_type: str = "sigmoid"
    reference_free: bool = False
    label_smoothing: float = 0.0


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class Config:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def get_config(
    model_id: str = "meta-llama/Llama-3.1-8B",
    method: str = "sft",
    **overrides,
) -> Config:
    """Return configuration for a given model and training method.

    Args:
        model_id: HuggingFace model identifier.
        method: Training method (sft, lora, qlora, dpo).
        **overrides: Additional overrides applied to the Config.
    """
    cfg = Config()
    cfg.model.model_id = model_id
    cfg.train.method = method

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    YAML structure mirrors the Config dataclass hierarchy:
        model:
          model_id: "meta-llama/Llama-3.1-8B"
          max_seq_length: 2048
        data:
          dataset_path: data/instructions.jsonl
        train:
          method: sft
          epochs: 3
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    section_map: Dict[str, object] = {
        "model": cfg.model,
        "data": cfg.data,
        "train": cfg.train,
        "lora": cfg.lora,
        "dpo": cfg.dpo,
        "generation": cfg.generation,
    }

    for section_name, section_obj in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    return cfg
