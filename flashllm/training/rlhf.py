"""RLHF (Reinforcement Learning from Human Feedback) implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class RLHFTrainer:
    """RLHF trainer with PPO-style policy optimization.

    Args:
        policy_model: The model being trained.
        reward_model: Frozen reward model.
        ref_model: Frozen reference model for KL penalty.
        tokenizer: Tokenizer instance.
        kl_coef: KL divergence penalty coefficient.
        clip_range: PPO clip range.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        tokenizer=None,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.clip_range = clip_range

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def compute_rewards(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute rewards for generated sequences."""
        if self.reward_model is None:
            return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

        with torch.no_grad():
            outputs = self.reward_model(input_ids=generated_ids)
            rewards = outputs.logits[:, -1] if hasattr(outputs, "logits") else outputs[0][:, -1]
        return rewards.squeeze(-1)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Execute a single PPO training step."""
        outputs = self.policy_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logps = log_probs.gather(-1, batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        new_logps = (token_logps * batch["attention_mask"][:, 1:]).sum(dim=-1)

        ratio = torch.exp(new_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        return {
            "policy_loss": policy_loss,
            "approx_kl": ((ratio - 1) - torch.log(ratio)).mean(),
        }
