"""Prefix caching for repeated prompt prefixes in serving.

Caches KV states for common prompt prefixes (system prompts, few-shot examples)
so they don't need to be recomputed for each request.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PrefixCacheEntry:
    """A cached KV state for a token prefix."""
    prefix_hash: str
    token_ids: List[int]
    kv_states: List[Tuple[torch.Tensor, torch.Tensor]]  # per-layer (key, value)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_layers(self) -> int:
        return len(self.kv_states)

    @property
    def memory_bytes(self) -> int:
        total = 0
        for k, v in self.kv_states:
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total


class PrefixCache:
    """LRU cache for prompt prefix KV states.

    Stores computed KV caches for token prefixes, enabling instant
    reuse when multiple requests share the same prefix (e.g., system prompt).

    Args:
        max_entries: Maximum number of cached prefixes.
        max_memory_mb: Maximum memory usage in MB.
        min_prefix_length: Minimum prefix length to cache (tokens).
    """

    def __init__(
        self,
        max_entries: int = 256,
        max_memory_mb: float = 1024.0,
        min_prefix_length: int = 16,
    ):
        self.max_entries = max_entries
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.min_prefix_length = min_prefix_length

        self._cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self._current_memory = 0

        self._total_hits = 0
        self._total_misses = 0

    @staticmethod
    def compute_hash(token_ids: List[int]) -> str:
        """Compute a hash for a token prefix."""
        data = ",".join(str(t) for t in token_ids)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def lookup(self, token_ids: List[int]) -> Optional[PrefixCacheEntry]:
        """Look up the longest cached prefix that matches.

        Searches for progressively shorter prefixes until a match is found.

        Args:
            token_ids: Full input token sequence.

        Returns:
            Cached entry for the longest matching prefix, or None.
        """
        for length in range(len(token_ids), self.min_prefix_length - 1, -1):
            prefix = token_ids[:length]
            prefix_hash = self.compute_hash(prefix)

            if prefix_hash in self._cache:
                entry = self._cache[prefix_hash]
                if entry.token_ids == prefix:
                    entry.last_accessed = time.time()
                    entry.hit_count += 1
                    self._total_hits += 1
                    self._cache.move_to_end(prefix_hash)
                    return entry

        self._total_misses += 1
        return None

    def insert(
        self,
        token_ids: List[int],
        kv_states: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[PrefixCacheEntry]:
        """Insert a prefix and its KV states into the cache.

        Args:
            token_ids: Token prefix to cache.
            kv_states: Per-layer (key, value) tensors.

        Returns:
            The cached entry, or None if too small to cache.
        """
        if len(token_ids) < self.min_prefix_length:
            return None

        prefix_hash = self.compute_hash(token_ids)

        if prefix_hash in self._cache:
            self._cache.move_to_end(prefix_hash)
            return self._cache[prefix_hash]

        kv_cloned = [(k.clone(), v.clone()) for k, v in kv_states]

        entry = PrefixCacheEntry(
            prefix_hash=prefix_hash,
            token_ids=list(token_ids),
            kv_states=kv_cloned,
        )

        while (self._current_memory + entry.memory_bytes > self.max_memory_bytes
               or len(self._cache) >= self.max_entries):
            if not self._cache:
                break
            self._evict_lru()

        self._cache[prefix_hash] = entry
        self._current_memory += entry.memory_bytes

        return entry

    def _evict_lru(self):
        """Evict the least recently used entry."""
        if not self._cache:
            return
        _, entry = self._cache.popitem(last=False)
        self._current_memory -= entry.memory_bytes

    def remove(self, token_ids: List[int]) -> bool:
        """Remove a specific prefix from the cache."""
        prefix_hash = self.compute_hash(token_ids)
        if prefix_hash in self._cache:
            entry = self._cache.pop(prefix_hash)
            self._current_memory -= entry.memory_bytes
            return True
        return False

    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()
        self._current_memory = 0

    @property
    def num_entries(self) -> int:
        return len(self._cache)

    @property
    def memory_mb(self) -> float:
        return self._current_memory / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        total = self._total_hits + self._total_misses
        return self._total_hits / max(total, 1)

    def stats(self) -> Dict[str, float]:
        return {
            "num_entries": self.num_entries,
            "memory_mb": self.memory_mb,
            "hit_rate": self.hit_rate,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
        }


class PrefixAwareEngine:
    """Serving engine extension that uses prefix caching.

    Wraps a model to automatically cache and reuse KV states
    for shared prompt prefixes.

    Args:
        model: The language model.
        tokenizer: Associated tokenizer.
        cache_config: Prefix cache configuration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        max_cache_entries: int = 256,
        max_cache_memory_mb: float = 1024.0,
        min_prefix_length: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_cache = PrefixCache(
            max_entries=max_cache_entries,
            max_memory_mb=max_cache_memory_mb,
            min_prefix_length=min_prefix_length,
        )

    @torch.inference_mode()
    def generate_with_prefix_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with automatic prefix caching.

        Args:
            input_ids: Input token IDs, shape (1, seq_len).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated token IDs.
        """
        token_list = input_ids[0].tolist()
        cached = self.prefix_cache.lookup(token_list)

        if cached is not None:
            prefix_len = cached.num_tokens
            past_key_values = tuple(
                (k.to(input_ids.device), v.to(input_ids.device))
                for k, v in cached.kv_states
            )
            remaining_ids = input_ids[:, prefix_len:]

            outputs = self.model(
                input_ids=remaining_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            outputs = self.model(input_ids=input_ids, use_cache=True)
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                kv_states = [
                    (k.cpu(), v.cpu()) for k, v in outputs.past_key_values
                ]
                self.prefix_cache.insert(token_list, kv_states)

        return self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, **kwargs,
        )
