"""Unit tests for model components."""

import pytest
import torch

from flashllm.nn import RMSNorm, SwiGLU, RotaryEmbedding
from flashllm.models.lora import LoRALinear
from flashllm.generation.sampler import Sampler
from flashllm.generation.kv_cache import KVCache


def test_rms_norm():
    norm = RMSNorm(hidden_size=64)
    x = torch.randn(2, 10, 64)
    output = norm(x)
    assert output.shape == (2, 10, 64)


def test_swiglu():
    ffn = SwiGLU(hidden_size=64, intermediate_size=128)
    x = torch.randn(2, 10, 64)
    output = ffn(x)
    assert output.shape == (2, 10, 64)


def test_rotary_embedding():
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    cos, sin = rope(seq_len=128, device=torch.device("cpu"))
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_lora_linear():
    original = torch.nn.Linear(64, 128, bias=False)
    lora = LoRALinear(original, rank=8, alpha=16.0)

    x = torch.randn(2, 10, 64)
    output = lora(x)
    assert output.shape == (2, 10, 128)

    assert not lora.original.weight.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_merge():
    original = torch.nn.Linear(32, 64, bias=False)
    lora = LoRALinear(original, rank=4, alpha=8.0)

    merged = lora.merge()
    assert isinstance(merged, torch.nn.Linear)
    assert merged.weight.shape == (64, 32)


def test_kv_cache():
    cache = KVCache(
        num_layers=2,
        max_batch_size=1,
        max_seq_length=128,
        num_kv_heads=4,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )

    key = torch.randn(1, 4, 5, 16)
    value = torch.randn(1, 4, 5, 16)

    cached_k, cached_v = cache.update(layer_idx=0, key=key, value=value)
    assert cached_k.shape == (1, 4, 5, 16)

    cache.advance(num_tokens=5)
    assert cache.get_seq_length() == 5

    cache.reset()
    assert cache.get_seq_length() == 0
