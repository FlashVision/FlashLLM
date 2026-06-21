"""Cross-entropy losses for language modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for causal language modeling.

    Handles the shift between logits and labels automatically.

    Args:
        ignore_index: Token ID to ignore in loss computation (typically -100).
        reduction: Loss reduction method ("mean", "sum", "none").
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with causal shift.

        Args:
            logits: Model output (batch, seq_len, vocab_size).
            labels: Target labels (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class LabelSmoothedCrossEntropy(nn.Module):
    """Label-smoothed cross-entropy loss.

    Distributes a small probability mass across all tokens to prevent
    overconfident predictions and improve generalization.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 typical).
        ignore_index: Token ID to ignore.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy.

        Args:
            logits: Model output (batch, seq_len, vocab_size).
            labels: Target labels (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        logits_flat.size(-1)

        nll_loss = F.nll_loss(log_probs, labels_flat, ignore_index=self.ignore_index, reduction="mean")
        smooth_loss = -log_probs.mean(dim=-1)

        mask = labels_flat != self.ignore_index
        smooth_loss = smooth_loss[mask].mean() if mask.any() else torch.tensor(0.0)

        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss
