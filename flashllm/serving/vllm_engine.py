"""PagedAttention-style KV cache management for high-throughput serving.

Implements block-level memory management inspired by vLLM's PagedAttention,
allowing non-contiguous KV cache storage and efficient memory reuse.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BlockTable:
    """Maps logical blocks of a sequence to physical blocks in the memory pool.

    Each sequence maintains its own block table that tracks which physical
    blocks hold its KV cache data.
    """
    block_size: int
    physical_blocks: List[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.physical_blocks)

    @property
    def num_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def append_block(self, physical_block_id: int):
        self.physical_blocks.append(physical_block_id)

    def get_physical_block(self, logical_idx: int) -> int:
        if logical_idx >= len(self.physical_blocks):
            raise IndexError(
                f"Logical block {logical_idx} out of range "
                f"(have {len(self.physical_blocks)} blocks)"
            )
        return self.physical_blocks[logical_idx]

    def release_last_block(self) -> int:
        return self.physical_blocks.pop()


class MemoryPool:
    """Fixed-size pool of physical KV cache blocks.

    Pre-allocates GPU memory as a contiguous tensor, then manages
    allocation and deallocation of fixed-size blocks.

    Args:
        num_blocks: Total number of physical blocks.
        block_size: Number of tokens per block.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per attention head.
        dtype: Data type for cache tensors.
        device: Device for cache tensors.
    """

    def __init__(
        self,
        num_blocks: int = 256,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_pool = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.value_pool = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        self._free_blocks: List[int] = list(range(num_blocks))
        self._allocated: Dict[int, int] = {}  # block_id -> ref_count

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return 1.0 - (self.num_free_blocks / self.num_blocks)

    def allocate_block(self) -> int:
        """Allocate a single physical block. Returns block ID."""
        if not self._free_blocks:
            raise RuntimeError("Memory pool exhausted — no free blocks")
        block_id = self._free_blocks.pop()
        self._allocated[block_id] = 1
        return block_id

    def free_block(self, block_id: int):
        """Return a physical block to the free pool."""
        if block_id in self._allocated:
            self._allocated[block_id] -= 1
            if self._allocated[block_id] <= 0:
                del self._allocated[block_id]
                self.key_pool[:, block_id].zero_()
                self.value_pool[:, block_id].zero_()
                self._free_blocks.append(block_id)

    def ref_block(self, block_id: int):
        """Increment reference count for copy-on-write sharing."""
        if block_id in self._allocated:
            self._allocated[block_id] += 1

    def write_kv(
        self,
        layer_idx: int,
        block_id: int,
        slot_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Write key-value vectors into a specific block slot.

        Args:
            layer_idx: Transformer layer index.
            block_id: Physical block ID.
            slot_offset: Token offset within the block.
            key: Key tensor of shape (num_tokens, num_kv_heads, head_dim).
            value: Value tensor of shape (num_tokens, num_kv_heads, head_dim).
        """
        num_tokens = key.shape[0]
        self.key_pool[layer_idx, block_id, slot_offset:slot_offset + num_tokens] = key
        self.value_pool[layer_idx, block_id, slot_offset:slot_offset + num_tokens] = value

    def read_kv(
        self,
        layer_idx: int,
        block_ids: List[int],
        max_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read key-value vectors from a list of blocks.

        Args:
            layer_idx: Transformer layer index.
            block_ids: Ordered list of physical block IDs.
            max_tokens: Limit total tokens returned (for partial last block).

        Returns:
            Tuple of (keys, values), each of shape (total_tokens, num_kv_heads, head_dim).
        """
        keys = self.key_pool[layer_idx, block_ids].reshape(-1, self.num_kv_heads, self.head_dim)
        values = self.value_pool[layer_idx, block_ids].reshape(-1, self.num_kv_heads, self.head_dim)
        if max_tokens is not None:
            keys = keys[:max_tokens]
            values = values[:max_tokens]
        return keys, values

    @property
    def memory_mb(self) -> float:
        element_size = self.key_pool.element_size()
        total_bytes = 2 * self.key_pool.numel() * element_size
        return total_bytes / (1024 * 1024)


class PagedKVCache:
    """Paged KV cache that manages block tables per sequence.

    Automatically allocates new blocks as sequences grow, and
    supports prefix sharing via reference-counted blocks.

    Args:
        memory_pool: Underlying physical memory pool.
    """

    def __init__(self, memory_pool: MemoryPool):
        self.pool = memory_pool
        self.block_tables: Dict[int, BlockTable] = {}
        self.seq_lengths: Dict[int, int] = {}

    def register_sequence(self, seq_id: int):
        """Register a new sequence for KV caching."""
        self.block_tables[seq_id] = BlockTable(block_size=self.pool.block_size)
        self.seq_lengths[seq_id] = 0

    def append_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Append new KV entries for a sequence at a given layer.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            keys: Key tensor of shape (num_new_tokens, num_kv_heads, head_dim).
            values: Value tensor of shape (num_new_tokens, num_kv_heads, head_dim).
        """
        table = self.block_tables[seq_id]
        num_tokens = keys.shape[0]
        written = 0

        while written < num_tokens:
            current_len = self.seq_lengths[seq_id] + written if layer_idx == 0 else (
                self.seq_lengths[seq_id] + written
            )
            block_idx = current_len // self.pool.block_size
            slot_offset = current_len % self.pool.block_size

            if block_idx >= table.num_blocks:
                new_block = self.pool.allocate_block()
                table.append_block(new_block)

            physical_block = table.get_physical_block(block_idx)
            chunk = min(num_tokens - written, self.pool.block_size - slot_offset)

            self.pool.write_kv(
                layer_idx, physical_block, slot_offset,
                keys[written:written + chunk],
                values[written:written + chunk],
            )
            written += chunk

        if layer_idx == self.pool.num_layers - 1:
            self.seq_lengths[seq_id] += num_tokens

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve full KV cache for a sequence at a given layer."""
        table = self.block_tables[seq_id]
        seq_len = self.seq_lengths[seq_id]
        return self.pool.read_kv(layer_idx, table.physical_blocks, max_tokens=seq_len)

    def free_sequence(self, seq_id: int):
        """Release all blocks held by a sequence."""
        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id].physical_blocks:
                self.pool.free_block(block_id)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def fork_sequence(self, src_id: int, dst_id: int):
        """Copy-on-write fork: new sequence shares blocks with source."""
        src_table = self.block_tables[src_id]
        dst_table = BlockTable(block_size=self.pool.block_size)
        for block_id in src_table.physical_blocks:
            dst_table.append_block(block_id)
            self.pool.ref_block(block_id)
        self.block_tables[dst_id] = dst_table
        self.seq_lengths[dst_id] = self.seq_lengths[src_id]


class VLLMEngine:
    """High-throughput serving engine with PagedAttention.

    Manages multiple sequences with paged KV caching and provides
    a step-based interface for iteration-level scheduling.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Associated tokenizer.
        max_num_seqs: Maximum concurrent sequences.
        block_size: Tokens per KV cache block.
        gpu_memory_fraction: Fraction of GPU memory for KV cache.
        dtype: Data type for KV cache.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_num_seqs: int = 256,
        block_size: int = 16,
        gpu_memory_fraction: float = 0.9,
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        self.dtype = dtype

        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_kv_heads = getattr(config, "num_key_value_heads",
                               getattr(config, "num_attention_heads", 32))
        head_dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)

        device = next(model.parameters()).device
        num_blocks = self._estimate_num_blocks(
            gpu_memory_fraction, num_layers, num_kv_heads, head_dim, block_size, dtype, device,
        )

        self.memory_pool = MemoryPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=str(device),
        )
        self.kv_cache = PagedKVCache(self.memory_pool)
        self._next_seq_id = 0

        logger.info(
            "VLLMEngine initialized: %d blocks, %.1f MB KV cache",
            num_blocks, self.memory_pool.memory_mb,
        )

    def _estimate_num_blocks(
        self, fraction, num_layers, num_kv_heads, head_dim, block_size, dtype, device,
    ) -> int:
        if device.type != "cuda":
            return 512

        total_mem = torch.cuda.get_device_properties(device).total_mem
        available = int(total_mem * fraction)

        element_size = torch.tensor([], dtype=dtype).element_size()
        bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * element_size
        num_blocks = max(available // bytes_per_block, 64)
        return min(num_blocks, 65536)

    def add_request(self, prompt: str, max_tokens: int = 256) -> int:
        """Add a new generation request.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.

        Returns:
            Sequence ID for tracking.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.kv_cache.register_sequence(seq_id)
        return seq_id

    def remove_request(self, seq_id: int):
        """Remove a completed or cancelled request."""
        self.kv_cache.free_sequence(seq_id)

    @property
    def num_active_sequences(self) -> int:
        return len(self.kv_cache.block_tables)

    @property
    def cache_utilization(self) -> float:
        return self.memory_pool.utilization
