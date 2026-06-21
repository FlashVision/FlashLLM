"""Validator for evaluating LLM models."""

from typing import Dict, Optional

import torch
from tqdm import tqdm

from flashllm.utils.logger import get_logger
from flashllm.utils.metrics import compute_perplexity

logger = get_logger(__name__)


class Validator:
    """Evaluate a trained model on a validation set.

    Args:
        model_path: Path to saved model checkpoint.
        model_id: HuggingFace model ID (if model_path not provided).
        val_data: Path to validation data file.
        device: Device for evaluation.
        batch_size: Evaluation batch size.
        max_seq_length: Maximum sequence length.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        val_data: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_seq_length: int = 2048,
    ):
        self.model_path = model_path
        self.model_id = model_id
        self.val_data = val_data
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load model for evaluation."""
        from flashllm.models.flash_llm import FlashLLM

        source = self.model_path or self.model_id
        if source is None:
            raise ValueError("Either model_path or model_id must be provided")

        flash = FlashLLM(source, device_map=self.device)
        self.model = flash.model
        self.tokenizer = flash.tokenizer
        self.model.eval()

    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics.

        Returns:
            Dictionary containing perplexity, loss, and other metrics.
        """
        if self.model is None:
            self._load_model()

        from flashllm.data.dataset import SFTDataset
        from flashllm.data.dataloader import create_dataloader

        dataset = SFTDataset(
            self.val_data,
            self.tokenizer,
            max_seq_length=self.max_seq_length,
        )
        dataloader = create_dataloader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss
                num_tokens = (batch["labels"] != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = compute_perplexity(avg_loss)

        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
        }

        logger.info(f"Validation — loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
        return results
