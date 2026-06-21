"""Tests for serving infrastructure — paged KV cache, continuous batching, speculative decoding."""

import pytest
import torch

from flashllm.serving.vllm_engine import MemoryPool, BlockTable, PagedKVCache
from flashllm.serving.continuous_batching import (
    ContinuousBatcher, SequenceRequest, SequenceStatus,
)
from flashllm.serving.prefix_cache import PrefixCache


class TestMemoryPool:
    def test_allocate_and_free(self):
        pool = MemoryPool(num_blocks=16, block_size=4, num_layers=2,
                          num_kv_heads=4, head_dim=8, dtype=torch.float32, device="cpu")
        assert pool.num_free_blocks == 16

        block_id = pool.allocate_block()
        assert pool.num_free_blocks == 15
        assert 0 <= block_id < 16

        pool.free_block(block_id)
        assert pool.num_free_blocks == 16

    def test_write_and_read(self):
        pool = MemoryPool(num_blocks=4, block_size=8, num_layers=2,
                          num_kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu")
        block_id = pool.allocate_block()

        key = torch.randn(3, 2, 4)
        value = torch.randn(3, 2, 4)
        pool.write_kv(0, block_id, 0, key, value)

        keys, values = pool.read_kv(0, [block_id], max_tokens=3)
        assert keys.shape == (3, 2, 4)
        assert torch.allclose(keys, key, atol=1e-6)

    def test_exhaustion_raises(self):
        pool = MemoryPool(num_blocks=2, block_size=4, num_layers=1,
                          num_kv_heads=1, head_dim=4, dtype=torch.float32, device="cpu")
        pool.allocate_block()
        pool.allocate_block()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate_block()

    def test_memory_mb(self):
        pool = MemoryPool(num_blocks=8, block_size=16, num_layers=4,
                          num_kv_heads=8, head_dim=64, dtype=torch.float16, device="cpu")
        assert pool.memory_mb > 0


class TestBlockTable:
    def test_append_and_get(self):
        table = BlockTable(block_size=16)
        table.append_block(5)
        table.append_block(10)
        assert table.num_blocks == 2
        assert table.get_physical_block(0) == 5
        assert table.get_physical_block(1) == 10

    def test_release_last(self):
        table = BlockTable(block_size=16)
        table.append_block(3)
        table.append_block(7)
        released = table.release_last_block()
        assert released == 7
        assert table.num_blocks == 1


class TestPagedKVCache:
    def test_register_and_free(self):
        pool = MemoryPool(num_blocks=8, block_size=4, num_layers=1,
                          num_kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu")
        cache = PagedKVCache(pool)

        cache.register_sequence(0)
        assert 0 in cache.block_tables

        cache.free_sequence(0)
        assert 0 not in cache.block_tables

    def test_append_and_get(self):
        pool = MemoryPool(num_blocks=8, block_size=4, num_layers=1,
                          num_kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu")
        cache = PagedKVCache(pool)
        cache.register_sequence(0)

        keys = torch.randn(3, 2, 4)
        values = torch.randn(3, 2, 4)
        cache.append_tokens(0, 0, keys, values)

        k, v = cache.get_kv(0, 0)
        assert k.shape[0] == 3

    def test_fork_sequence(self):
        pool = MemoryPool(num_blocks=8, block_size=4, num_layers=1,
                          num_kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu")
        cache = PagedKVCache(pool)
        cache.register_sequence(0)

        keys = torch.randn(2, 2, 4)
        values = torch.randn(2, 2, 4)
        cache.append_tokens(0, 0, keys, values)

        cache.fork_sequence(0, 1)
        assert cache.seq_lengths[1] == cache.seq_lengths[0]


class TestContinuousBatcher:
    def test_add_and_schedule(self):
        batcher = ContinuousBatcher(max_num_seqs=4, max_num_batched_tokens=1024)
        req_id = batcher.add_request([1, 2, 3, 4, 5], max_tokens=10)
        assert batcher.num_waiting == 1

        output = batcher.schedule(num_free_blocks=100, block_size=16)
        assert len(output.prefill_requests) == 1
        assert output.prefill_requests[0].request_id == req_id

    def test_process_outputs_eos(self):
        batcher = ContinuousBatcher()
        batcher.add_request([1, 2, 3], max_tokens=10)
        batcher.schedule(num_free_blocks=100, block_size=16)

        finished = batcher.process_outputs({0: 99}, eos_token_id=99)
        assert len(finished) == 1
        assert finished[0].status == SequenceStatus.FINISHED_EOS

    def test_process_outputs_max_length(self):
        batcher = ContinuousBatcher()
        batcher.add_request([1, 2], max_tokens=2)
        batcher.schedule(num_free_blocks=100, block_size=16)

        batcher.process_outputs({0: 10}, eos_token_id=99)
        finished = batcher.process_outputs({0: 11}, eos_token_id=99)
        assert len(finished) == 1
        assert finished[0].status == SequenceStatus.FINISHED_LENGTH

    def test_has_pending(self):
        batcher = ContinuousBatcher()
        assert not batcher.has_pending_requests
        batcher.add_request([1, 2], max_tokens=5)
        assert batcher.has_pending_requests


class TestPrefixCache:
    def test_insert_and_lookup(self):
        cache = PrefixCache(max_entries=10, min_prefix_length=2)
        tokens = [1, 2, 3, 4, 5]
        kv_states = [(torch.randn(4, 8), torch.randn(4, 8))]

        cache.insert(tokens, kv_states)
        assert cache.num_entries == 1

        result = cache.lookup(tokens)
        assert result is not None
        assert result.token_ids == tokens

    def test_miss_returns_none(self):
        cache = PrefixCache(min_prefix_length=2)
        result = cache.lookup([10, 20, 30])
        assert result is None

    def test_hit_rate(self):
        cache = PrefixCache(min_prefix_length=2)
        tokens = [1, 2, 3]
        cache.insert(tokens, [(torch.randn(2, 4), torch.randn(2, 4))])

        cache.lookup(tokens)
        cache.lookup([99, 98])
        assert cache.hit_rate > 0

    def test_eviction(self):
        cache = PrefixCache(max_entries=2, min_prefix_length=2)
        cache.insert([1, 2], [(torch.randn(2, 4), torch.randn(2, 4))])
        cache.insert([3, 4], [(torch.randn(2, 4), torch.randn(2, 4))])
        cache.insert([5, 6], [(torch.randn(2, 4), torch.randn(2, 4))])
        assert cache.num_entries <= 2
