"""Beam search decoding for text generation."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BeamHypothesis:
    """A single beam search hypothesis."""

    tokens: List[int] = field(default_factory=list)
    score: float = 0.0
    is_finished: bool = False


class BeamSearch:
    """Beam search decoding with length penalty.

    Args:
        num_beams: Number of beams to maintain.
        max_length: Maximum generation length.
        length_penalty: Exponential penalty for length (>1 = longer, <1 = shorter).
        early_stopping: Whether to stop when all beams are finished.
        eos_token_id: End-of-sequence token ID.
    """

    def __init__(
        self,
        num_beams: int = 4,
        max_length: int = 512,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id

    def search(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[BeamHypothesis]:
        """Run beam search on the model.

        Args:
            model: Language model with a forward method returning logits.
            input_ids: Initial input token IDs (batch_size=1, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            List of beam hypotheses sorted by score.
        """
        device = input_ids.device

        beam_input_ids = input_ids.repeat(self.num_beams, 1)
        beam_scores = torch.zeros(self.num_beams, device=device)
        beam_scores[1:] = float("-inf")

        finished_beams: List[BeamHypothesis] = []

        for step in range(self.max_length):
            with torch.no_grad():
                outputs = model(input_ids=beam_input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            vocab_size = log_probs.shape[-1]
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(-1)

            top_scores, top_indices = torch.topk(next_scores, 2 * self.num_beams)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            new_beam_input_ids = []
            new_beam_scores = []
            active_beams = 0

            for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
                if active_beams >= self.num_beams:
                    break

                token_id = token_idx.item()
                new_ids = torch.cat([beam_input_ids[beam_idx], token_idx.unsqueeze(0)])

                if token_id == self.eos_token_id:
                    length = new_ids.shape[0] - input_ids.shape[1]
                    adjusted_score = score.item() / (length**self.length_penalty)
                    finished_beams.append(
                        BeamHypothesis(
                            tokens=new_ids[input_ids.shape[1] :].tolist(),
                            score=adjusted_score,
                            is_finished=True,
                        )
                    )
                else:
                    new_beam_input_ids.append(new_ids)
                    new_beam_scores.append(score)
                    active_beams += 1

            if not new_beam_input_ids:
                break

            beam_input_ids = torch.stack(new_beam_input_ids)
            beam_scores = torch.stack(new_beam_scores)

            if self.early_stopping and len(finished_beams) >= self.num_beams:
                break

        for i in range(beam_input_ids.shape[0]):
            length = beam_input_ids.shape[1] - input_ids.shape[1]
            adjusted_score = beam_scores[i].item() / (length**self.length_penalty)
            finished_beams.append(
                BeamHypothesis(
                    tokens=beam_input_ids[i, input_ids.shape[1] :].tolist(),
                    score=adjusted_score,
                    is_finished=True,
                )
            )

        finished_beams.sort(key=lambda h: h.score, reverse=True)
        return finished_beams[: self.num_beams]
