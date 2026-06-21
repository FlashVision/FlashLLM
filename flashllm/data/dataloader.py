"""DataLoader utilities for LLM training."""

from typing import Optional

from torch.utils.data import DataLoader, Dataset, random_split


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for LLM training.

    Args:
        dataset: PyTorch dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop incomplete last batch.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def split_dataset(
    dataset: Dataset,
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Split a dataset into train and validation sets.

    Args:
        dataset: Full dataset.
        val_split: Fraction for validation (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    import torch

    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset


def create_train_val_loaders(
    dataset: Dataset,
    batch_size: int = 4,
    val_split: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """Create train and validation DataLoaders from a single dataset.

    Args:
        dataset: Full dataset.
        batch_size: Batch size for both loaders.
        val_split: Fraction for validation.
        num_workers: Number of workers.
        seed: Random seed.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset, val_dataset = split_dataset(dataset, val_split, seed)

    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    return train_loader, val_loader
