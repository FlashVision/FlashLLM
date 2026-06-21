"""Continuous batching with iteration-level scheduling.

Implements dynamic batching where new requests can join a running batch
at each decoding iteration, maximizing GPU utilization.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_EOS = auto()
    PREEMPTED = auto()


@dataclass
class SequenceRequest:
    """A single generation request tracked through the batching system.

    Args:
        request_id: Unique request identifier.
        prompt_token_ids: Tokenized prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling bound.
        stop_token_ids: Token IDs that stop generation.
    """
    request_id: int
    prompt_token_ids: List[int]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: Optional[List[int]] = None

    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_finished(self) -> bool:
        return self.status in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_EOS,
        )

    def append_token(self, token_id: int):
        self.output_token_ids.append(token_id)


@dataclass
class SchedulerOutput:
    """Result of a scheduling decision for one iteration."""
    prefill_requests: List[SequenceRequest]
    decode_requests: List[SequenceRequest]
    preempted_requests: List[SequenceRequest]
    num_batched_tokens: int


class ContinuousBatcher:
    """Iteration-level scheduler for continuous batching.

    At each step, decides which sequences to prefill, decode, or preempt
    based on available memory and scheduling policy.

    Args:
        max_num_seqs: Maximum sequences in a batch.
        max_num_batched_tokens: Maximum total tokens per iteration.
        max_model_len: Maximum supported sequence length.
        preemption_mode: Strategy for preemption ("recompute" or "swap").
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        max_model_len: int = 8192,
        preemption_mode: str = "recompute",
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.preemption_mode = preemption_mode

        self._waiting: List[SequenceRequest] = []
        self._running: List[SequenceRequest] = []
        self._finished: List[SequenceRequest] = []
        self._next_id = 0

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> int:
        """Add a new request to the waiting queue.

        Returns:
            Request ID.
        """
        request = SequenceRequest(
            request_id=self._next_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=stop_token_ids or [],
        )
        self._next_id += 1
        self._waiting.append(request)
        return request.request_id

    def schedule(self, num_free_blocks: int, block_size: int) -> SchedulerOutput:
        """Run one scheduling iteration.

        Args:
            num_free_blocks: Available KV cache blocks.
            block_size: Tokens per block.

        Returns:
            SchedulerOutput describing what to run this iteration.
        """
        prefill_requests: List[SequenceRequest] = []
        decode_requests: List[SequenceRequest] = []
        preempted_requests: List[SequenceRequest] = []

        available_tokens = self.max_num_batched_tokens
        available_slots = self.max_num_seqs

        running_tokens = 0
        still_running = []
        for seq in self._running:
            blocks_needed = (seq.total_len + block_size - 1) // block_size
            if blocks_needed <= num_free_blocks and available_slots > 0:
                decode_requests.append(seq)
                running_tokens += 1
                available_slots -= 1
                available_tokens -= 1
            else:
                seq.status = SequenceStatus.PREEMPTED
                preempted_requests.append(seq)

        self._running = [s for s in self._running if s not in preempted_requests]

        for preempted in preempted_requests:
            preempted.status = SequenceStatus.WAITING
            self._waiting.insert(0, preempted)

        new_waiting = []
        for seq in self._waiting:
            if available_slots <= 0 or available_tokens <= 0:
                new_waiting.append(seq)
                continue

            blocks_needed = (seq.prompt_len + block_size - 1) // block_size
            if blocks_needed > num_free_blocks:
                new_waiting.append(seq)
                continue

            if seq.prompt_len > available_tokens:
                new_waiting.append(seq)
                continue

            seq.status = SequenceStatus.RUNNING
            prefill_requests.append(seq)
            self._running.append(seq)
            available_tokens -= seq.prompt_len
            available_slots -= 1
            num_free_blocks -= blocks_needed

        self._waiting = new_waiting

        total_tokens = sum(s.prompt_len for s in prefill_requests) + len(decode_requests)
        return SchedulerOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=preempted_requests,
            num_batched_tokens=total_tokens,
        )

    def process_outputs(
        self,
        outputs: Dict[int, int],
        eos_token_id: int,
    ) -> List[SequenceRequest]:
        """Process model outputs and update sequence states.

        Args:
            outputs: Mapping of request_id -> generated token_id.
            eos_token_id: End-of-sequence token ID.

        Returns:
            List of newly finished sequences.
        """
        newly_finished = []
        still_running = []

        for seq in self._running:
            if seq.request_id not in outputs:
                still_running.append(seq)
                continue

            token_id = outputs[seq.request_id]
            seq.append_token(token_id)

            if token_id == eos_token_id:
                seq.status = SequenceStatus.FINISHED_EOS
                newly_finished.append(seq)
            elif seq.output_len >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH
                newly_finished.append(seq)
            elif seq.stop_token_ids and token_id in seq.stop_token_ids:
                seq.status = SequenceStatus.FINISHED_STOPPED
                newly_finished.append(seq)
            else:
                still_running.append(seq)

        self._running = still_running
        self._finished.extend(newly_finished)
        return newly_finished

    def get_finished(self) -> List[SequenceRequest]:
        """Retrieve and clear all finished requests."""
        finished = self._finished
        self._finished = []
        return finished

    @property
    def has_pending_requests(self) -> bool:
        return len(self._waiting) > 0 or len(self._running) > 0

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_finished(self) -> int:
        return len(self._finished)
