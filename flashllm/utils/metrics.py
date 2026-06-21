"""Evaluation metrics for language models."""

import math
from typing import Dict, List, Optional


def compute_perplexity(avg_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss.

    Args:
        avg_loss: Average cross-entropy loss value.

    Returns:
        Perplexity score.
    """
    return math.exp(min(avg_loss, 100))


def compute_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU score for text generation.

    Simple implementation of corpus-level BLEU.

    Args:
        predictions: Generated text strings.
        references: Reference text strings.
        max_n: Maximum n-gram order.

    Returns:
        Dictionary with bleu-1 through bleu-n scores.
    """
    from collections import Counter

    scores = {}
    for n in range(1, max_n + 1):
        precision_sum = 0
        total_count = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(_get_ngrams(ref_tokens, n))

            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items())
            total = sum(pred_ngrams.values())

            precision_sum += clipped
            total_count += total

        scores[f"bleu-{n}"] = precision_sum / max(total_count, 1)

    return scores


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L score (longest common subsequence).

    Args:
        prediction: Generated text.
        reference: Reference text.

    Returns:
        Dictionary with precision, recall, f1 scores.
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    lcs_length = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs_length / max(len(pred_tokens), 1)
    recall = lcs_length / max(len(ref_tokens), 1)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_f1": f1}


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
