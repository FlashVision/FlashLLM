"""Checkpoint save/load utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = 0.0,
    path: str = "checkpoint",
    tokenizer=None,
    config: Optional[Dict] = None,
):
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        epoch: Current epoch.
        step: Current global step.
        loss: Current loss value.
        path: Directory to save checkpoint.
        tokenizer: Optional tokenizer to save.
        config: Optional config dict to save.
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_dir))
    else:
        torch.save(model.state_dict(), str(save_dir / "model.pt"))

    if tokenizer and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(save_dir))

    training_state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    if optimizer:
        training_state["optimizer"] = optimizer.state_dict()
    if scheduler:
        training_state["scheduler"] = scheduler.state_dict()

    torch.save(training_state, str(save_dir / "training_state.pt"))

    if config:
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Checkpoint saved: {save_dir} (epoch={epoch}, step={step}, loss={loss:.4f})")


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict:
    """Load a training checkpoint.

    Args:
        path: Checkpoint directory.
        model: Model to load weights into.
        optimizer: Optimizer to restore state.
        scheduler: Scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Dictionary with epoch, step, loss metadata.
    """
    load_dir = Path(path)

    if model is not None:
        if hasattr(model, "from_pretrained"):
            pass
        else:
            model_path = load_dir / "model.pt"
            if model_path.exists():
                state_dict = torch.load(str(model_path), map_location=device)
                model.load_state_dict(state_dict)

    training_state_path = load_dir / "training_state.pt"
    metadata = {"epoch": 0, "step": 0, "loss": 0.0}

    if training_state_path.exists():
        training_state = torch.load(str(training_state_path), map_location=device)
        metadata["epoch"] = training_state.get("epoch", 0)
        metadata["step"] = training_state.get("step", 0)
        metadata["loss"] = training_state.get("loss", 0.0)

        if optimizer and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])

    logger.info(f"Checkpoint loaded: {load_dir} (epoch={metadata['epoch']}, step={metadata['step']})")
    return metadata
