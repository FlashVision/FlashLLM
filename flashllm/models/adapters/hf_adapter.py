"""HuggingFace model adapter — unified loading for any AutoModelForCausalLM."""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceAdapter:
    """Adapter for loading and configuring HuggingFace causal LM models.

    Args:
        model_id: HuggingFace model identifier.
        torch_dtype: Weight data type.
        device_map: Device placement.
        trust_remote_code: Trust remote code in model repos.
    """

    SUPPORTED_FAMILIES = {
        "llama": ["meta-llama"],
        "mistral": ["mistralai"],
        "phi": ["microsoft"],
        "gemma": ["google"],
        "qwen": ["Qwen"],
    }

    def __init__(self, model_id: str, torch_dtype: str = "auto", device_map: str = "auto", trust_remote_code: bool = False):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

    def detect_family(self) -> str:
        """Detect the model family from the model ID."""
        for family, prefixes in self.SUPPORTED_FAMILIES.items():
            for prefix in prefixes:
                if prefix.lower() in self.model_id.lower():
                    return family
        return "unknown"

    def load_model(self, **kwargs) -> AutoModelForCausalLM:
        """Load the model."""
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
        dtype = dtype_map.get(self.torch_dtype, "auto")
        return AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=self.trust_remote_code, **kwargs
        )

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_target_modules(self) -> list:
        """Get recommended LoRA target modules for this model family."""
        family = self.detect_family()
        targets = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        return targets.get(family, ["q_proj", "k_proj", "v_proj", "o_proj"])
