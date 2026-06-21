"""GGUF export for llama.cpp inference.

Converts HuggingFace / PyTorch models to GGUF format for efficient
CPU inference with llama.cpp and related runtimes.
"""

import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3


class GGMLType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14


GGML_TYPE_SIZE = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.Q4_0: 18,  # block_size=32, 18 bytes per block
    GGMLType.Q4_1: 20,
    GGMLType.Q5_0: 22,
    GGMLType.Q5_1: 24,
    GGMLType.Q8_0: 34,
    GGMLType.Q8_1: 36,
    GGMLType.Q2_K: 256,
    GGMLType.Q3_K: 256,
    GGMLType.Q4_K: 144,
    GGMLType.Q5_K: 176,
    GGMLType.Q6_K: 210,
}

GGML_BLOCK_SIZE = {
    GGMLType.F32: 1,
    GGMLType.F16: 1,
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.Q2_K: 256,
    GGMLType.Q3_K: 256,
    GGMLType.Q4_K: 256,
    GGMLType.Q5_K: 256,
    GGMLType.Q6_K: 256,
}


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


def _quantize_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantize a float tensor to Q4_0 format.

    Each block of 32 values is stored as a scale (fp16) + 16 bytes of 4-bit values.
    """
    flat = tensor.float().flatten().numpy()
    n = len(flat)
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size

    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:n] = flat

    result = bytearray()
    for i in range(n_blocks):
        block = padded[i * block_size:(i + 1) * block_size]
        abs_max = np.max(np.abs(block))
        scale = abs_max / 7.0 if abs_max > 0 else 0.0

        result.extend(struct.pack("<e", np.float16(scale)))

        if scale > 0:
            quantized = np.clip(np.round(block / scale) + 8, 0, 15).astype(np.uint8)
        else:
            quantized = np.full(block_size, 8, dtype=np.uint8)

        for j in range(0, block_size, 2):
            byte = (quantized[j] & 0x0F) | ((quantized[j + 1] & 0x0F) << 4)
            result.append(byte)

    return bytes(result)


def _quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """Quantize a float tensor to Q8_0 format.

    Each block of 32 values is stored as a scale (fp16) + 32 bytes of int8 values.
    """
    flat = tensor.float().flatten().numpy()
    n = len(flat)
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size

    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:n] = flat

    result = bytearray()
    for i in range(n_blocks):
        block = padded[i * block_size:(i + 1) * block_size]
        abs_max = np.max(np.abs(block))
        scale = abs_max / 127.0 if abs_max > 0 else 0.0

        result.extend(struct.pack("<e", np.float16(scale)))

        if scale > 0:
            quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        else:
            quantized = np.zeros(block_size, dtype=np.int8)

        result.extend(quantized.tobytes())

    return bytes(result)


class GGUFWriter:
    """Write tensors and metadata to GGUF format.

    Args:
        output_path: Path for the output .gguf file.
        arch: Model architecture name (e.g., "llama").
    """

    def __init__(self, output_path: str, arch: str = "llama"):
        self.output_path = Path(output_path)
        self.arch = arch
        self._metadata: List[Tuple[str, Any, GGUFValueType]] = []
        self._tensors: List[Tuple[str, torch.Tensor, GGMLType]] = []

        self.add_string("general.architecture", arch)
        self.add_string("general.file_type", "gguf")

    def add_string(self, key: str, value: str):
        self._metadata.append((key, value, GGUFValueType.STRING))

    def add_uint32(self, key: str, value: int):
        self._metadata.append((key, value, GGUFValueType.UINT32))

    def add_uint64(self, key: str, value: int):
        self._metadata.append((key, value, GGUFValueType.UINT64))

    def add_float32(self, key: str, value: float):
        self._metadata.append((key, value, GGUFValueType.FLOAT32))

    def add_bool(self, key: str, value: bool):
        self._metadata.append((key, value, GGUFValueType.BOOL))

    def add_tensor(self, name: str, tensor: torch.Tensor, quant_type: GGMLType = GGMLType.F16):
        self._tensors.append((name, tensor, quant_type))

    def write(self):
        """Write the GGUF file to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self._tensors)))
            f.write(struct.pack("<Q", len(self._metadata)))

            for key, value, vtype in self._metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype.value))
                if vtype == GGUFValueType.STRING:
                    self._write_string(f, value)
                elif vtype == GGUFValueType.UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUFValueType.UINT64:
                    f.write(struct.pack("<Q", value))
                elif vtype == GGUFValueType.FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUFValueType.BOOL:
                    f.write(struct.pack("<B", int(value)))

            tensor_data_list = []
            offset = 0
            for name, tensor, quant_type in self._tensors:
                self._write_string(f, name)
                shape = list(tensor.shape)
                f.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", quant_type.value))
                f.write(struct.pack("<Q", offset))

                data = self._quantize_tensor(tensor, quant_type)
                tensor_data_list.append(data)
                offset += len(data)

            alignment = 32
            current = f.tell()
            padding = (alignment - (current % alignment)) % alignment
            f.write(b"\x00" * padding)

            for data in tensor_data_list:
                f.write(data)

        total_size = self.output_path.stat().st_size / (1024 * 1024)
        logger.info("GGUF written: %s (%.1f MB, %d tensors)", self.output_path, total_size, len(self._tensors))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _quantize_tensor(self, tensor: torch.Tensor, quant_type: GGMLType) -> bytes:
        if quant_type == GGMLType.F32:
            return tensor.float().cpu().numpy().tobytes()
        elif quant_type == GGMLType.F16:
            return tensor.half().cpu().numpy().tobytes()
        elif quant_type == GGMLType.Q4_0:
            return _quantize_q4_0(tensor.cpu())
        elif quant_type == GGMLType.Q8_0:
            return _quantize_q8_0(tensor.cpu())
        else:
            logger.warning("Unsupported quant type %s, falling back to F16", quant_type.name)
            return tensor.half().cpu().numpy().tobytes()


HF_TO_GGUF_TENSOR_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{}.self_attn.q_proj.weight": "blk.{}.attn_q.weight",
    "model.layers.{}.self_attn.k_proj.weight": "blk.{}.attn_k.weight",
    "model.layers.{}.self_attn.v_proj.weight": "blk.{}.attn_v.weight",
    "model.layers.{}.self_attn.o_proj.weight": "blk.{}.attn_output.weight",
    "model.layers.{}.mlp.gate_proj.weight": "blk.{}.ffn_gate.weight",
    "model.layers.{}.mlp.up_proj.weight": "blk.{}.ffn_up.weight",
    "model.layers.{}.mlp.down_proj.weight": "blk.{}.ffn_down.weight",
    "model.layers.{}.input_layernorm.weight": "blk.{}.attn_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "blk.{}.ffn_norm.weight",
}


def export_to_gguf(
    model_id: str,
    output_path: str,
    quantization: str = "q4_0",
    model=None,
) -> str:
    """Export a HuggingFace model to GGUF format.

    Args:
        model_id: HuggingFace model ID or local path.
        output_path: Output .gguf file path.
        quantization: Quantization type ("f16", "q4_0", "q8_0").
        model: Pre-loaded model (optional, will load from model_id if None).

    Returns:
        Path to the written GGUF file.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    quant_map = {
        "f32": GGMLType.F32,
        "f16": GGMLType.F16,
        "q4_0": GGMLType.Q4_0,
        "q8_0": GGMLType.Q8_0,
    }
    quant_type = quant_map.get(quantization.lower(), GGMLType.Q4_0)

    config = AutoConfig.from_pretrained(model_id)
    arch = getattr(config, "model_type", "llama")

    if model is None:
        logger.info("Loading model: %s", model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    writer = GGUFWriter(output_path, arch=arch)

    writer.add_uint32(f"{arch}.embedding_length", getattr(config, "hidden_size", 4096))
    writer.add_uint32(f"{arch}.block_count", getattr(config, "num_hidden_layers", 32))
    writer.add_uint32(f"{arch}.attention.head_count", getattr(config, "num_attention_heads", 32))
    writer.add_uint32(f"{arch}.attention.head_count_kv",
                      getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)))
    writer.add_uint32(f"{arch}.feed_forward_length", getattr(config, "intermediate_size", 11008))
    writer.add_uint32(f"{arch}.context_length", getattr(config, "max_position_embeddings", 4096))
    writer.add_uint32(f"{arch}.vocab_size", getattr(config, "vocab_size", 32000))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", getattr(config, "rms_norm_eps", 1e-5))
    writer.add_float32(f"{arch}.rope.freq_base", getattr(config, "rope_theta", 10000.0))

    state_dict = model.state_dict()
    num_layers = getattr(config, "num_hidden_layers", 32)

    for hf_pattern, gguf_pattern in HF_TO_GGUF_TENSOR_MAP.items():
        if "{}" in hf_pattern:
            for layer_idx in range(num_layers):
                hf_name = hf_pattern.format(layer_idx)
                gguf_name = gguf_pattern.format(layer_idx)
                if hf_name in state_dict:
                    tensor = state_dict[hf_name]
                    use_type = quant_type if tensor.dim() == 2 else GGMLType.F32
                    writer.add_tensor(gguf_name, tensor, use_type)
        else:
            if hf_pattern in state_dict:
                tensor = state_dict[hf_pattern]
                use_type = quant_type if tensor.dim() == 2 else GGMLType.F32
                writer.add_tensor(gguf_pattern, tensor, use_type)

    writer.write()
    return str(Path(output_path).resolve())
