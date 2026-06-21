"""Dataset classes for LLM training."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from flashllm.data.templates import get_template


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset.

    Supports formats:
        - Alpaca: {"instruction": ..., "input": ..., "output": ...}
        - ShareGPT: {"conversations": [{"from": "human", "value": ...}, ...]}
        - Simple: {"prompt": ..., "completion": ...}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        template: str = "alpaca",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.template = get_template(template)
        self.samples = self._load_data(data_path, max_samples)

    def _load_data(self, data_path: str, max_samples: Optional[int]) -> List[Dict]:
        path = Path(data_path)
        samples = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("data", [])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if max_samples:
            samples = samples[:max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = self.template.format_sample(sample)

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ChatDataset(Dataset):
    """Multi-turn chat conversation dataset.

    Expected format:
        {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 4096,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = self._load_data(data_path, max_samples)

    def _load_data(self, data_path: str, max_samples: Optional[int]) -> List[Dict]:
        path = Path(data_path)
        samples = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                samples = json.load(f)

        if max_samples:
            samples = samples[:max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = sample.get("messages", sample.get("conversations", []))

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InstructDataset(Dataset):
    """Instruction-following dataset with DPO pairs.

    Expected format:
        {"prompt": ..., "chosen": ..., "rejected": ...}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = self._load_data(data_path, max_samples)

    def _load_data(self, data_path: str, max_samples: Optional[int]) -> List[Dict]:
        path = Path(data_path)
        samples = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                samples = json.load(f)

        if max_samples:
            samples = samples[:max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_seq_length, padding="max_length", return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_seq_length, padding="max_length", return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
