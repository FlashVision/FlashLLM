"""Supervised Fine-Tuning (SFT) implementation with NEFTune support."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class NEFTuneHook:
    """NEFTune — Noisy Embedding Fine-Tuning.

    Adds uniform noise to embedding vectors during training to act as
    a regularizer, improving instruction-following performance.

    Reference: https://arxiv.org/abs/2310.05914

    Args:
        noise_alpha: Noise scaling factor. Higher = more noise.
            Recommended: 5-15 for most models.
    """

    def __init__(self, noise_alpha: float = 5.0):
        self.noise_alpha = noise_alpha
        self._handle = None

    def _noise_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> torch.Tensor:
        if module.training:
            dims = torch.tensor(output.shape[1] * output.shape[2], dtype=output.dtype, device=output.device)
            mag = self.noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(output).uniform_(-mag.item(), mag.item())
            return output + noise
        return output

    def register(self, embedding_layer: nn.Module):
        """Register the noise hook on an embedding layer."""
        self._handle = embedding_layer.register_forward_hook(self._noise_hook)
        logger.info("NEFTune registered with alpha=%.1f", self.noise_alpha)

    def remove(self):
        """Remove the noise hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        neftune_alpha: NEFTune noise alpha. Set > 0 to enable. Recommended: 5-15.
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
        neftune_alpha: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

        self.neftune_hook: Optional[NEFTuneHook] = None
        if neftune_alpha > 0:
            self.neftune_hook = NEFTuneHook(noise_alpha=neftune_alpha)
            embedding_layer = self._find_embedding_layer(model)
            if embedding_layer is not None:
                self.neftune_hook.register(embedding_layer)
            else:
                logger.warning("Could not find embedding layer for NEFTune")

    @staticmethod
    def _find_embedding_layer(model: nn.Module) -> Optional[nn.Module]:
        """Find the token embedding layer in the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and "embed" in name.lower():
                return module
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return module
        return None

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
            outputs = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
            )
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_perplexity": torch.exp(torch.tensor(avg_loss)).item()}
