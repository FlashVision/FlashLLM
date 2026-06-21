"""Trainer for SFT, LoRA, and DPO training."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from flashllm.cfg.config import Config
from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Unified trainer for SFT, LoRA/QLoRA, and DPO.

    Args:
        model_id: HuggingFace model identifier.
        dataset: Path to training data.
        method: Training method ("sft", "lora", "qlora", "dpo").
        config: Pre-built Config object (overrides other args).
        device: Device to train on.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        save_dir: Directory to save checkpoints.
        lora_rank: LoRA rank (if method is "lora" or "qlora").
        lora_alpha: LoRA alpha scaling.
        lora_targets: Target modules for LoRA.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        dataset: Optional[str] = None,
        method: str = "sft",
        config: Optional[Config] = None,
        device: str = "cuda",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_dir: str = "workspace/train",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: Optional[List[str]] = None,
        **kwargs,
    ):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = Config()
            if model_id:
                self.cfg.model.model_id = model_id
            if dataset:
                self.cfg.data.dataset_path = dataset
            self.cfg.train.method = method
            self.cfg.train.epochs = epochs
            self.cfg.train.batch_size = batch_size
            self.cfg.train.learning_rate = learning_rate
            self.cfg.train.save_dir = save_dir
            self.cfg.lora.rank = lora_rank
            self.cfg.lora.alpha = lora_alpha
            if lora_targets:
                self.cfg.lora.target_modules = lora_targets

        self.device = device
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.callbacks: List = []
        self.global_step = 0
        self.best_loss = float("inf")

    def _setup_model(self):
        """Load model and tokenizer, apply LoRA if needed."""
        from flashllm.models.flash_llm import FlashLLM
        from flashllm.models.lora import apply_lora, apply_qlora

        flash_model = FlashLLM(
            self.cfg.model.model_id,
            torch_dtype=self.cfg.model.torch_dtype,
            device_map=self.device,
        )
        self.model = flash_model.model
        self.tokenizer = flash_model.tokenizer

        if self.cfg.train.method in ("lora", "qlora"):
            if self.cfg.train.method == "qlora":
                self.model = apply_qlora(
                    self.model,
                    rank=self.cfg.lora.rank,
                    alpha=self.cfg.lora.alpha,
                    target_modules=self.cfg.lora.target_modules,
                    dropout=self.cfg.lora.dropout,
                )
            else:
                self.model = apply_lora(
                    self.model,
                    rank=self.cfg.lora.rank,
                    alpha=self.cfg.lora.alpha,
                    target_modules=self.cfg.lora.target_modules,
                    dropout=self.cfg.lora.dropout,
                )
            logger.info(f"Applied {self.cfg.train.method.upper()} (rank={self.cfg.lora.rank})")

        if self.cfg.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _setup_data(self):
        """Create train and validation dataloaders."""
        from flashllm.data.dataset import SFTDataset, InstructDataset
        from flashllm.data.dataloader import create_train_val_loaders

        if self.cfg.train.method == "dpo":
            dataset = InstructDataset(
                self.cfg.data.dataset_path,
                self.tokenizer,
                max_seq_length=self.cfg.model.max_seq_length,
                max_samples=self.cfg.data.max_samples,
            )
        else:
            dataset = SFTDataset(
                self.cfg.data.dataset_path,
                self.tokenizer,
                max_seq_length=self.cfg.model.max_seq_length,
                template=self.cfg.data.template,
                max_samples=self.cfg.data.max_samples,
            )

        self.train_loader, self.val_loader = create_train_val_loaders(
            dataset,
            batch_size=self.cfg.train.batch_size,
            val_split=self.cfg.data.val_split,
            num_workers=self.cfg.data.num_workers,
        )

    def _setup_optimizer(self):
        """Configure optimizer and learning rate scheduler."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.cfg.train.learning_rate, weight_decay=self.cfg.train.weight_decay)

        total_steps = len(self.train_loader) * self.cfg.train.epochs
        warmup_steps = int(total_steps * self.cfg.train.warmup_ratio)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    def add_callback(self, callback):
        """Add a training callback."""
        self.callbacks.append(callback)

    def _run_callbacks(self, event: str, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event):
                getattr(cb, event)(**kwargs)

    def train(self):
        """Run the full training loop."""
        logger.info(f"Starting training: method={self.cfg.train.method}, model={self.cfg.model.model_id}")

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        save_dir = Path(self.cfg.train.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        scaler = torch.amp.GradScaler("cuda") if self.cfg.train.amp and self.device == "cuda" else None
        self.model.train()

        self._run_callbacks("on_train_start", trainer=self)

        for epoch in range(self.cfg.train.epochs):
            epoch_loss = 0.0
            self._run_callbacks("on_epoch_start", epoch=epoch)

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.train.epochs}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss = self._compute_loss(batch)
                    scaler.scale(loss).backward()

                    if (step + 1) % self.cfg.train.gradient_accumulation_steps == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.global_step += 1
                else:
                    loss = self._compute_loss(batch)
                    loss.backward()

                    if (step + 1) % self.cfg.train.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.global_step += 1

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

                if self.global_step > 0 and self.global_step % self.cfg.train.save_steps == 0:
                    self._save_checkpoint(save_dir / f"step_{self.global_step}")

            avg_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")
            self._run_callbacks("on_epoch_end", epoch=epoch, loss=avg_loss)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint(save_dir / "best")

        self._save_checkpoint(save_dir / "final")
        self._run_callbacks("on_train_end", trainer=self)
        logger.info(f"Training complete. Checkpoints saved to {save_dir}")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on training method."""
        if self.cfg.train.method == "dpo":
            from flashllm.losses.dpo_loss import dpo_loss
            return dpo_loss(self.model, batch, beta=self.cfg.dpo.beta)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(path))
        logger.info(f"Checkpoint saved: {path}")
