"""Speculative decoding for faster autoregressive generation.

Uses a small draft model to propose multiple tokens, then verifies
them in parallel with the target model, accepting correct predictions
and rejecting incorrect ones using a modified rejection sampling scheme.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenNode:
    """Node in a speculative token tree."""
    token_id: int
    log_prob: float
    depth: int
    parent: Optional["TokenNode"] = None
    children: List["TokenNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, token_id: int, log_prob: float) -> "TokenNode":
        child = TokenNode(
            token_id=token_id,
            log_prob=log_prob,
            depth=self.depth + 1,
            parent=self,
        )
        self.children.append(child)
        return child


class TokenTree:
    """Tree of speculative token proposals for tree-based verification.

    Supports multiple candidate continuations at each position,
    allowing the verifier to accept any valid path through the tree.
    """

    def __init__(self):
        self.root: Optional[TokenNode] = None
        self._nodes: List[TokenNode] = []

    def build_from_draft(
        self,
        draft_token_ids: torch.Tensor,
        draft_log_probs: torch.Tensor,
    ):
        """Build a linear chain from draft model predictions.

        Args:
            draft_token_ids: Token IDs from draft model, shape (num_speculative_tokens,).
            draft_log_probs: Log probabilities from draft model, shape (num_speculative_tokens,).
        """
        self.root = TokenNode(
            token_id=draft_token_ids[0].item(),
            log_prob=draft_log_probs[0].item(),
            depth=0,
        )
        self._nodes = [self.root]

        current = self.root
        for i in range(1, len(draft_token_ids)):
            child = current.add_child(
                token_id=draft_token_ids[i].item(),
                log_prob=draft_log_probs[i].item(),
            )
            self._nodes.append(child)
            current = child

    def get_token_ids(self) -> List[int]:
        """Get all token IDs in tree traversal order."""
        return [node.token_id for node in self._nodes]

    def get_verification_positions(self) -> List[int]:
        """Get position indices for parallel verification."""
        return list(range(len(self._nodes)))

    @property
    def num_tokens(self) -> int:
        return len(self._nodes)


class SpeculativeDecoder:
    """Speculative decoding with draft model + target model verification.

    The draft model generates K candidate tokens cheaply, then the
    target model verifies all K tokens in a single forward pass.
    Accepted tokens are kept; the first rejected token is resampled
    from the target model's distribution.

    Args:
        target_model: Large target language model.
        draft_model: Small draft language model.
        tokenizer: Shared tokenizer.
        num_speculative_tokens: Number of tokens to speculate per step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.total_draft_tokens = 0
        self.accepted_tokens = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs, shape (1, seq_len).
            max_new_tokens: Maximum tokens to generate.
            eos_token_id: End-of-sequence token ID.

        Returns:
            Generated token IDs including input, shape (1, seq_len + num_generated).
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        generated = input_ids.clone()
        num_generated = 0

        while num_generated < max_new_tokens:
            remaining = max_new_tokens - num_generated
            k = min(self.num_speculative_tokens, remaining)

            draft_ids, draft_log_probs = self._draft_step(generated, k)

            num_accepted, next_token = self._verify_step(
                generated, draft_ids, draft_log_probs,
            )

            self.total_draft_tokens += k
            self.accepted_tokens += num_accepted

            if num_accepted > 0:
                generated = torch.cat(
                    [generated, draft_ids[:, :num_accepted]], dim=-1,
                )
                num_generated += num_accepted

            if next_token is not None:
                generated = torch.cat(
                    [generated, next_token.unsqueeze(0).unsqueeze(0)], dim=-1,
                )
                num_generated += 1

                if next_token.item() == eos_token_id:
                    break

            if any(t == eos_token_id for t in draft_ids[0, :num_accepted].tolist()):
                break

        return generated

    @torch.inference_mode()
    def _draft_step(
        self,
        context: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate k draft tokens autoregressively.

        Returns:
            Tuple of (draft_token_ids, draft_log_probs), each shape (1, k).
        """
        draft_ids = []
        draft_log_probs = []
        current = context

        for _ in range(k):
            outputs = self.draft_model(current)
            logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
            logits = logits / max(self.temperature, 1e-7)

            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            if self.top_p < 1.0:
                probs = self._apply_top_p(probs)

            token_id = torch.multinomial(probs, num_samples=1)
            draft_ids.append(token_id)
            draft_log_probs.append(log_probs.gather(-1, token_id))

            current = torch.cat([current, token_id], dim=-1)

        return (
            torch.cat(draft_ids, dim=-1),
            torch.cat(draft_log_probs, dim=-1),
        )

    @torch.inference_mode()
    def _verify_step(
        self,
        context: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_log_probs: torch.Tensor,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Verify draft tokens against the target model.

        Uses modified rejection sampling: for each position, accept the
        draft token with probability min(1, p_target / p_draft).

        Returns:
            Tuple of (num_accepted, resampled_token_or_None).
        """
        k = draft_ids.shape[1]
        verification_input = torch.cat([context, draft_ids], dim=-1)

        outputs = self.target_model(verification_input)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        target_logits = logits[:, -(k + 1):, :] / max(self.temperature, 1e-7)
        target_log_probs = F.log_softmax(target_logits, dim=-1)

        num_accepted = 0
        for i in range(k):
            draft_token = draft_ids[0, i]
            p_target = target_log_probs[0, i, draft_token].exp()
            p_draft = draft_log_probs[0, i].exp()

            acceptance_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
            r = torch.rand(1, device=draft_ids.device)

            if r < acceptance_prob:
                num_accepted += 1
            else:
                adjusted_probs = torch.clamp(
                    target_log_probs[0, i].exp() - draft_log_probs[0, i].exp().unsqueeze(-1).expand_as(target_log_probs[0, i].exp()),
                    min=0,
                )
                adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
                next_token = torch.multinomial(adjusted_probs.unsqueeze(0), num_samples=1).squeeze()
                return num_accepted, next_token

        next_probs = target_log_probs[0, k].exp()
        if self.top_p < 1.0:
            next_probs = self._apply_top_p(next_probs.unsqueeze(0)).squeeze(0)
        next_token = torch.multinomial(next_probs.unsqueeze(0), num_samples=1).squeeze()
        return num_accepted, next_token

    def _apply_top_p(self, probs: torch.Tensor) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumulative_probs - sorted_probs) > self.top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        probs = sorted_probs.scatter(-1, sorted_indices.argsort(-1), sorted_probs)
        return probs

    def reset_stats(self):
        """Reset acceptance rate statistics."""
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
