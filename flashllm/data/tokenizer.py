"""Tokenizer wrapper for HuggingFace AutoTokenizer."""

from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerWrapper:
    """Wrapper around HuggingFace AutoTokenizer with LLM-specific defaults.

    Handles padding token setup, special tokens, and provides a unified
    interface for encoding/decoding across different model families.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 2048,
        padding_side: str = "right",
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.max_length = max_length

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )

        self.tokenizer.padding_side = padding_side

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict:
        """Encode text to token IDs.

        Args:
            text: Input text or list of texts.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate.
            return_tensors: Return format ("pt", "np", None).

        Returns:
            Dictionary with input_ids, attention_mask.
        """
        return self.tokenizer(
            text,
            max_length=max_length or self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, **kwargs) -> Union[str, List[int]]:
        """Apply chat template to messages."""
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)

    def __call__(self, *args, **kwargs):
        """Delegate to underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.tokenizer)
