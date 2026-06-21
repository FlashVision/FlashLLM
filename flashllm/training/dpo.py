"""Direct Preference Optimization (DPO) implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Args:
        model: The policy model to train.
        ref_model: The frozen reference model.
        tokenizer: Tokenizer instance.
        beta: DPO temperature parameter.
        loss_type: Loss variant ("sigmoid", "hinge", "ipo", "kto").
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        tokenizer=None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.loss_type = loss_type

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single DPO training step."""
        chosen_logps = self._get_log_probs(self.model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected_logps = self._get_log_probs(self.model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen_logps = self._get_log_probs(self.ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                ref_rejected_logps = self._get_log_probs(self.ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
        else:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)

        logits = self.beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
        return -F.logsigmoid(logits).mean()

    def _get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute sequence-level log probabilities."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return (token_logps * shift_mask).sum(dim=-1)
