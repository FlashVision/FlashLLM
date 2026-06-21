"""Distributed training with FSDP and DeepSpeed ZeRO.

Provides wrappers for scaling LLM training across multiple GPUs
with Fully Sharded Data Parallel (FSDP) and DeepSpeed ZeRO stages.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel.

    Args:
        sharding_strategy: FSDP sharding strategy ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD").
        cpu_offload: Whether to offload parameters to CPU.
        mixed_precision: Mixed precision policy ("fp16", "bf16", None).
        backward_prefetch: Backward prefetch mode ("BACKWARD_PRE", "BACKWARD_POST").
        activation_checkpointing: Enable gradient checkpointing.
        auto_wrap_policy: Auto-wrapping policy ("transformer", "size_based").
        min_num_params: Minimum params for size-based wrapping.
        sync_module_states: Sync module states across ranks at init.
        forward_prefetch: Enable forward prefetch for pipelining.
        limit_all_gathers: Rate-limit all-gather communication.
    """

    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    mixed_precision: Optional[str] = "bf16"
    backward_prefetch: str = "BACKWARD_PRE"
    activation_checkpointing: bool = True
    auto_wrap_policy: str = "transformer"
    min_num_params: int = 1_000_000
    sync_module_states: bool = True
    forward_prefetch: bool = False
    limit_all_gathers: bool = True


def wrap_with_fsdp(
    model: nn.Module,
    config: Optional[FSDPConfig] = None,
    transformer_layer_cls: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    """Wrap a model with FSDP for distributed training.

    Args:
        model: Model to wrap.
        config: FSDP configuration. Uses defaults if None.
        transformer_layer_cls: Transformer layer class for auto-wrapping.

    Returns:
        FSDP-wrapped model.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        CPUOffload,
        MixedPrecision,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )
    from functools import partial

    config = config or FSDPConfig()

    sharding_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_map.get(config.sharding_strategy, ShardingStrategy.FULL_SHARD)

    mixed_precision = None
    if config.mixed_precision == "bf16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.mixed_precision == "fp16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

    auto_wrap_policy = None
    if config.auto_wrap_policy == "transformer" and transformer_layer_cls is not None:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    elif config.auto_wrap_policy == "size_based":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )

    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = backward_prefetch_map.get(config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)

    wrapped = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=backward_prefetch,
        sync_module_states=config.sync_module_states,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
    )

    if config.activation_checkpointing and transformer_layer_cls is not None:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, transformer_layer_cls),
        )

    rank = int(os.environ.get("RANK", 0))
    logger.info(
        "[Rank %d] FSDP wrapped: strategy=%s, mixed_precision=%s, checkpointing=%s",
        rank,
        config.sharding_strategy,
        config.mixed_precision,
        config.activation_checkpointing,
    )
    return wrapped


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed ZeRO optimization.

    Args:
        stage: ZeRO optimization stage (0, 1, 2, 3).
        offload_optimizer: Offload optimizer states to CPU.
        offload_params: Offload parameters to CPU (stage 3 only).
        overlap_comm: Overlap communication with computation.
        contiguous_gradients: Use contiguous gradient buffers.
        reduce_bucket_size: Size of gradient reduction buckets.
        stage3_prefetch_bucket_size: Prefetch bucket size for stage 3.
        stage3_param_persistence_threshold: Param persistence threshold.
        fp16_enabled: Enable FP16 training.
        bf16_enabled: Enable BF16 training.
        gradient_accumulation_steps: Steps between gradient updates.
        train_micro_batch_size_per_gpu: Micro-batch size.
        gradient_clipping: Maximum gradient norm.
    """

    stage: int = 2
    offload_optimizer: bool = False
    offload_params: bool = False
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    fp16_enabled: bool = False
    bf16_enabled: bool = True
    gradient_accumulation_steps: int = 1
    train_micro_batch_size_per_gpu: int = 1
    gradient_clipping: float = 1.0


def get_deepspeed_config(config: Optional[DeepSpeedConfig] = None) -> Dict[str, Any]:
    """Build a DeepSpeed configuration dictionary.

    Args:
        config: DeepSpeed configuration. Uses defaults if None.

    Returns:
        Dictionary suitable for deepspeed.initialize().
    """
    config = config or DeepSpeedConfig()

    zero_config: Dict[str, Any] = {
        "stage": config.stage,
        "overlap_comm": config.overlap_comm,
        "contiguous_gradients": config.contiguous_gradients,
        "reduce_bucket_size": config.reduce_bucket_size,
    }

    if config.stage == 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": config.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": config.stage3_param_persistence_threshold,
            }
        )

    if config.offload_optimizer:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if config.offload_params and config.stage == 3:
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    ds_config: Dict[str, Any] = {
        "zero_optimization": zero_config,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "gradient_clipping": config.gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    if config.fp16_enabled:
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0, "loss_scale_window": 1000}
    if config.bf16_enabled:
        ds_config["bf16"] = {"enabled": True}

    return ds_config


def init_deepspeed(
    model: nn.Module,
    config: Optional[DeepSpeedConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    model_parameters: Optional[Any] = None,
):
    """Initialize a model with DeepSpeed.

    Args:
        model: Model to wrap.
        config: DeepSpeed configuration.
        optimizer: Optional optimizer (DeepSpeed can create its own).
        lr_scheduler: Optional learning rate scheduler.
        model_parameters: Parameters to optimize (defaults to model.parameters()).

    Returns:
        Tuple of (engine, optimizer, _, lr_scheduler).
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError("DeepSpeed support requires: pip install deepspeed")

    ds_config = get_deepspeed_config(config)

    if model_parameters is None:
        model_parameters = model.parameters()

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=ds_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    logger.info(
        "DeepSpeed initialized: ZeRO stage %d, offload_optimizer=%s",
        (config or DeepSpeedConfig()).stage,
        (config or DeepSpeedConfig()).offload_optimizer,
    )
    return engine, optimizer, lr_scheduler


def setup_distributed(backend: str = "nccl"):
    """Initialize the distributed process group.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU).
    """
    if not torch.distributed.is_initialized():
        init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        logger.info(
            "Distributed initialized: rank=%s, world_size=%s",
            os.environ.get("RANK", 0),
            os.environ.get("WORLD_SIZE", 1),
        )


def cleanup_distributed():
    """Clean up the distributed process group."""
    if torch.distributed.is_initialized():
        destroy_process_group()
