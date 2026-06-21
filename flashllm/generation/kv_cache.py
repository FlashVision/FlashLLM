"""KV cache management for efficient autoregressive generation."""

from typing import List, Optional, Tuple

import torch


class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Stores past key and value tensors for each layer to avoid recomputation
    during sequential token generation.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        max_seq_length: Maximum sequence length to cache.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        dtype: Data type for cache tensors.
        device: Device for cache tensors.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_length: int = 4096,
        num_kv_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seq_lengths: torch.Tensor = torch.zeros(max_batch_size, dtype=torch.long)

        self._allocate()

    def _allocate(self):
        """Pre-allocate cache tensors."""
        for _ in range(self.num_layers):
            k = torch.zeros(
                self.max_batch_size,
                self.num_kv_heads,
                self.max_seq_length,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            v = torch.zeros(
                self.max_batch_size,
                self.num_kv_heads,
                self.max_seq_length,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.key_cache.append(k)
            self.value_cache.append(v)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer and return the full cached KV."""
        seq_len = self.seq_lengths[batch_idx].item()
        new_len = key.shape[2]

        self.key_cache[layer_idx][batch_idx, :, seq_len : seq_len + new_len, :] = key[0]
        self.value_cache[layer_idx][batch_idx, :, seq_len : seq_len + new_len, :] = value[0]

        total_len = seq_len + new_len
        cached_keys = self.key_cache[layer_idx][batch_idx : batch_idx + 1, :, :total_len, :]
        cached_values = self.value_cache[layer_idx][batch_idx : batch_idx + 1, :, :total_len, :]

        return cached_keys, cached_values

    def advance(self, batch_idx: int = 0, num_tokens: int = 1):
        """Advance the sequence length counter after generation."""
        self.seq_lengths[batch_idx] += num_tokens

    def get_seq_length(self, batch_idx: int = 0) -> int:
        """Get current cached sequence length."""
        return self.seq_lengths[batch_idx].item()

    def reset(self, batch_idx: Optional[int] = None):
        """Reset cache for a specific batch index or all."""
        if batch_idx is not None:
            self.seq_lengths[batch_idx] = 0
            for layer_idx in range(self.num_layers):
                self.key_cache[layer_idx][batch_idx].zero_()
                self.value_cache[layer_idx][batch_idx].zero_()
        else:
            self.seq_lengths.zero_()
            for layer_idx in range(self.num_layers):
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()

    @property
    def memory_mb(self) -> float:
        """Estimate cache memory usage in MB."""
        element_size = self.key_cache[0].element_size() if self.key_cache else 2
        total_elements = (
            2 * self.num_layers * self.max_batch_size * self.num_kv_heads * self.max_seq_length * self.head_dim
        )
        return total_elements * element_size / (1024 * 1024)
