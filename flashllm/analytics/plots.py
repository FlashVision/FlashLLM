"""Plotting utilities for training metrics and benchmarks."""

from pathlib import Path
from typing import Dict, List, Optional


def plot_training_loss(
    losses: List[float],
    output_path: str = "training_loss.png",
    title: str = "Training Loss",
):
    """Plot training loss curve.

    Args:
        losses: List of loss values per step/epoch.
        output_path: Path to save the plot.
        title: Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires: pip install 'flashllm[analytics]'")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=1.5, color="#2196F3")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_comparison(
    results: List[Dict],
    output_path: str = "benchmark_comparison.png",
):
    """Plot benchmark comparison across models.

    Args:
        results: List of benchmark result dicts (from Benchmark.run()).
        output_path: Path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires: pip install 'flashllm[analytics]'")

    model_names = [r.get("model_id", "unknown").split("/")[-1] for r in results]
    tokens_per_sec = [r.get("tokens_per_sec", 0) for r in results]
    memory_mb = [r.get("memory_mb", 0) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.barh(model_names, tokens_per_sec, color="#4CAF50")
    ax1.set_xlabel("Tokens/sec")
    ax1.set_title("Throughput")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars2 = ax2.barh(model_names, memory_mb, color="#FF9800")
    ax2.set_xlabel("Memory (MB)")
    ax2.set_title("Peak Memory Usage")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity_curve(
    perplexities: List[float],
    output_path: str = "perplexity.png",
):
    """Plot perplexity over training steps.

    Args:
        perplexities: Perplexity values.
        output_path: Path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires: pip install 'flashllm[analytics]'")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(perplexities, linewidth=1.5, color="#9C27B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
