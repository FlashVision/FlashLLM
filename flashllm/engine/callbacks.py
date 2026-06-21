"""Training callbacks for FlashLLM."""

import csv
import time
from pathlib import Path
from typing import Optional

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class Callback:
    """Base callback class. Override methods to hook into training."""

    def on_train_start(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_start(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass


class EarlyStopping(Callback):
    """Stop training when a metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        metric: Metric to monitor ("loss" or "perplexity").
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 5, metric: str = "loss", min_delta: float = 0.001):
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.counter = 0

    def on_epoch_end(self, **kwargs):
        value = kwargs.get(self.metric, kwargs.get("loss", float("inf")))

        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping: No improvement for {self.patience} epochs. Stopping.")
                raise StopIteration("Early stopping triggered")


class CSVLogger(Callback):
    """Log training metrics to a CSV file.

    Args:
        filepath: Path to the CSV log file.
    """

    def __init__(self, filepath: str = "training_log.csv"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def on_epoch_end(self, **kwargs):
        row = {
            "epoch": kwargs.get("epoch", 0),
            "loss": kwargs.get("loss", 0.0),
            "timestamp": time.time(),
        }

        write_header = not self._initialized and not self.filepath.exists()

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._initialized = True


class WandBCallback(Callback):
    """Log metrics to Weights & Biases.

    Args:
        project: W&B project name.
        run_name: Optional run name.
    """

    def __init__(self, project: str = "flashllm", run_name: Optional[str] = None):
        self.project = project
        self.run_name = run_name
        self._run = None

    def on_train_start(self, **kwargs):
        try:
            import wandb

            self._run = wandb.init(project=self.project, name=self.run_name)
        except ImportError:
            logger.warning("wandb not installed. Skipping WandB logging.")

    def on_epoch_end(self, **kwargs):
        if self._run:
            import wandb

            wandb.log(
                {
                    "epoch": kwargs.get("epoch", 0),
                    "loss": kwargs.get("loss", 0.0),
                }
            )

    def on_train_end(self, **kwargs):
        if self._run:
            import wandb

            wandb.finish()
