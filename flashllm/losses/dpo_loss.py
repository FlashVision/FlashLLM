"""DPO, IPO, and KTO loss functions."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dpo_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    ref_model: Optional[nn.Module] = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute DPO loss for a batch of chosen/rejected pairs."""
    chosen_logps = _compute_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rejected_logps = _compute_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    if ref_model is not None:
        with torch.no_grad():
            ref_chosen_logps = _compute_logps(ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            ref_rejected_logps = _compute_logps(
                ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
            )
    else:
        ref_chosen_logps = torch.zeros_like(chosen_logps)
        ref_rejected_logps = torch.zeros_like(rejected_logps)

    logits = beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
    loss = -F.logsigmoid(logits).mean()

    if label_smoothing > 0:
        reverse = -F.logsigmoid(-logits).mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * reverse
    return loss


def ipo_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    ref_model: Optional[nn.Module] = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Identity Preference Optimization loss."""
    chosen_logps = _compute_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rejected_logps = _compute_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    if ref_model is not None:
        with torch.no_grad():
            ref_chosen = _compute_logps(ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            ref_rejected = _compute_logps(ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
        chosen_logps = chosen_logps - ref_chosen
        rejected_logps = rejected_logps - ref_rejected

    diff = chosen_logps - rejected_logps
    return ((diff - 1 / (2 * beta)) ** 2).mean()


def kto_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    ref_model: Optional[nn.Module] = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Kahneman-Tversky Optimization loss."""
    chosen_logps = _compute_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rejected_logps = _compute_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    if ref_model is not None:
        with torch.no_grad():
            ref_chosen = _compute_logps(ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            ref_rejected = _compute_logps(ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
        chosen_rewards = beta * (chosen_logps - ref_chosen)
        rejected_rewards = beta * (rejected_logps - ref_rejected)
    else:
        chosen_rewards = beta * chosen_logps
        rejected_rewards = beta * rejected_logps

    return -F.logsigmoid(chosen_rewards).mean() + -F.logsigmoid(-rejected_rewards).mean()


def _compute_logps(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute sequence-level log probabilities."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_mask).sum(dim=-1)
