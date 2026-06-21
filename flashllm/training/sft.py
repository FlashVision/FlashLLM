"""Supervised Fine-Tuning (SFT) implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class SFTTrainer:
    """Supervised Fine-Tuning trainer.

    Args:
        model: The language model to fine-tune.
        tokenizer: Tokenizer for the model.
        train_dataloader: Training DataLoader.
        val_dataloader: Optional validation DataLoader.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW.
        max_grad_norm: Maximum gradient norm for clipping.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_perplexity": torch.exp(torch.tensor(avg_loss)).item()}
