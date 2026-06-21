"""Structured output generation — JSON mode, regex-constrained, grammar-guided.

Constrains LLM generation to produce outputs matching a specific format,
using logit masking at each decoding step.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JSONSchema:
    """Simplified JSON schema for constraining generation.

    Args:
        schema: JSON Schema dict (subset of JSON Schema spec).
        strict: If True, require exact match; if False, allow extra fields.
    """
    schema: Dict[str, Any]
    strict: bool = True

    @property
    def required_keys(self) -> List[str]:
        return self.schema.get("required", [])

    @property
    def properties(self) -> Dict[str, Any]:
        return self.schema.get("properties", {})


class JSONModeConstraint:
    """Constrains generation to produce valid JSON matching a schema.

    At each decoding step, masks logits to only allow tokens that
    keep the output on a valid JSON path.

    Args:
        tokenizer: Tokenizer for encoding/decoding tokens.
        schema: Optional JSON schema to constrain output.
        max_depth: Maximum nesting depth.
    """

    JSON_STRUCTURAL_CHARS = set('{}[],:"\\ \n\t')
    JSON_VALUE_START_CHARS = set('"0123456789-tfn[{')

    def __init__(self, tokenizer, schema: Optional[JSONSchema] = None, max_depth: int = 32):
        self.tokenizer = tokenizer
        self.schema = schema
        self.max_depth = max_depth
        self._build_token_maps()

    def _build_token_maps(self):
        """Pre-compute token categories for efficient masking."""
        vocab_size = len(self.tokenizer)
        self._open_brace_tokens: Set[int] = set()
        self._close_brace_tokens: Set[int] = set()
        self._open_bracket_tokens: Set[int] = set()
        self._close_bracket_tokens: Set[int] = set()
        self._quote_tokens: Set[int] = set()
        self._colon_tokens: Set[int] = set()
        self._comma_tokens: Set[int] = set()
        self._string_content_tokens: Set[int] = set()
        self._number_tokens: Set[int] = set()
        self._whitespace_tokens: Set[int] = set()
        self._true_false_null_tokens: Set[int] = set()

        for token_id in range(vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                continue

            stripped = token_str.strip()
            if not stripped:
                self._whitespace_tokens.add(token_id)
                continue

            if stripped == "{":
                self._open_brace_tokens.add(token_id)
            elif stripped == "}":
                self._close_brace_tokens.add(token_id)
            elif stripped == "[":
                self._open_bracket_tokens.add(token_id)
            elif stripped == "]":
                self._close_bracket_tokens.add(token_id)
            elif stripped == ":":
                self._colon_tokens.add(token_id)
            elif stripped == ",":
                self._comma_tokens.add(token_id)
            elif stripped.startswith('"'):
                self._quote_tokens.add(token_id)

            if stripped and stripped[0].isdigit() or stripped.startswith("-"):
                self._number_tokens.add(token_id)

            if stripped in ("true", "false", "null"):
                self._true_false_null_tokens.add(token_id)

            if all(c.isprintable() or c in ' \t\n' for c in token_str):
                self._string_content_tokens.add(token_id)

    def __call__(
        self,
        logits: torch.Tensor,
        generated_text: str,
    ) -> torch.Tensor:
        """Apply JSON-mode constraint to logits.

        Args:
            logits: Raw logits of shape (vocab_size,) or (1, vocab_size).
            generated_text: Text generated so far.

        Returns:
            Masked logits.
        """
        squeezed = logits.dim() == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        state = self._analyze_state(generated_text)

        allowed_tokens = self._get_allowed_tokens(state)

        if allowed_tokens:
            mask = torch.full_like(logits, float("-inf"))
            for token_id in allowed_tokens:
                if token_id < logits.shape[-1]:
                    mask[0, token_id] = 0.0
            logits = logits + mask

        return logits.squeeze(0) if squeezed else logits

    def _analyze_state(self, text: str) -> str:
        """Determine the current JSON parsing state."""
        stripped = text.strip()
        if not stripped:
            return "START"

        depth = 0
        in_string = False
        escape_next = False
        last_structural = ""

        for char in stripped:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
                last_structural = "OBJECT_START"
            elif char == "}":
                depth -= 1
                last_structural = "OBJECT_END"
            elif char == "[":
                depth += 1
                last_structural = "ARRAY_START"
            elif char == "]":
                depth -= 1
                last_structural = "ARRAY_END"
            elif char == ":":
                last_structural = "COLON"
            elif char == ",":
                last_structural = "COMMA"

        if in_string:
            return "IN_STRING"
        if depth == 0 and last_structural in ("OBJECT_END", "ARRAY_END"):
            return "COMPLETE"
        return last_structural or "START"

    def _get_allowed_tokens(self, state: str) -> Set[int]:
        """Get allowed token IDs based on current state."""
        if state == "START":
            return self._open_brace_tokens | self._open_bracket_tokens | self._whitespace_tokens
        elif state == "OBJECT_START":
            return self._quote_tokens | self._close_brace_tokens | self._whitespace_tokens
        elif state == "COLON":
            return (self._quote_tokens | self._number_tokens |
                    self._open_brace_tokens | self._open_bracket_tokens |
                    self._true_false_null_tokens | self._whitespace_tokens)
        elif state == "COMMA":
            return self._quote_tokens | self._whitespace_tokens
        elif state == "OBJECT_END":
            return self._comma_tokens | self._close_brace_tokens | self._close_bracket_tokens | self._whitespace_tokens
        elif state == "ARRAY_START":
            return (self._quote_tokens | self._number_tokens |
                    self._open_brace_tokens | self._open_bracket_tokens |
                    self._true_false_null_tokens | self._close_bracket_tokens |
                    self._whitespace_tokens)
        elif state == "ARRAY_END":
            return self._comma_tokens | self._close_bracket_tokens | self._close_brace_tokens | self._whitespace_tokens
        elif state == "IN_STRING":
            return self._string_content_tokens
        elif state == "COMPLETE":
            eos = self.tokenizer.eos_token_id
            return {eos} if eos is not None else set()
        return set()


class RegexConstraint:
    """Constrains generation to match a regular expression pattern.

    Uses the regex pattern to determine valid next characters and
    masks tokens accordingly at each step.

    Args:
        tokenizer: Tokenizer for encoding/decoding tokens.
        pattern: Regular expression pattern the output must match.
    """

    def __init__(self, tokenizer, pattern: str):
        self.tokenizer = tokenizer
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def __call__(
        self,
        logits: torch.Tensor,
        generated_text: str,
    ) -> torch.Tensor:
        """Apply regex constraint to logits.

        Args:
            logits: Raw logits.
            generated_text: Text generated so far.

        Returns:
            Masked logits allowing only tokens that keep partial match valid.
        """
        squeezed = logits.dim() == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        vocab_size = logits.shape[-1]
        mask = torch.full_like(logits, float("-inf"))

        for token_id in range(min(vocab_size, len(self.tokenizer))):
            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                continue
            candidate = generated_text + token_str
            if self._is_valid_prefix(candidate):
                mask[0, token_id] = 0.0

        if self._compiled.fullmatch(generated_text):
            eos = self.tokenizer.eos_token_id
            if eos is not None and eos < vocab_size:
                mask[0, eos] = 0.0

        logits = logits + mask
        return logits.squeeze(0) if squeezed else logits

    def _is_valid_prefix(self, text: str) -> bool:
        """Check if text could be a prefix of a valid match."""
        if self._compiled.fullmatch(text):
            return True
        partial = re.compile(self.pattern[:])
        try:
            return partial.match(text) is not None
        except re.error:
            return True


@dataclass
class GrammarRule:
    """A single production rule in a context-free grammar."""
    name: str
    alternatives: List[List[str]]  # each alternative is a list of symbol names or literals


class GrammarConstraint:
    """Grammar-guided generation using a simplified CFG.

    Maintains a parse stack and constrains tokens to those
    allowed by the current grammar state.

    Args:
        tokenizer: Tokenizer instance.
        rules: List of grammar rules.
        start_symbol: Name of the start symbol.
    """

    def __init__(
        self,
        tokenizer,
        rules: List[GrammarRule],
        start_symbol: str = "root",
    ):
        self.tokenizer = tokenizer
        self.rules = {rule.name: rule for rule in rules}
        self.start_symbol = start_symbol
        self._stack: List[str] = [start_symbol]

    def get_allowed_strings(self) -> List[str]:
        """Get strings that are currently valid next tokens based on grammar state."""
        if not self._stack:
            return []

        top = self._stack[-1]

        if top not in self.rules:
            return [top]

        rule = self.rules[top]
        allowed = set()
        for alt in rule.alternatives:
            if alt:
                first = alt[0]
                if first not in self.rules:
                    allowed.add(first)
                else:
                    sub_rule = self.rules[first]
                    for sub_alt in sub_rule.alternatives:
                        if sub_alt and sub_alt[0] not in self.rules:
                            allowed.add(sub_alt[0])
        return list(allowed)

    def advance(self, consumed: str):
        """Advance the grammar state after consuming a token."""
        if not self._stack:
            return

        top = self._stack[-1]
        if top not in self.rules:
            if consumed.strip() == top.strip():
                self._stack.pop()
            return

        rule = self.rules[top]
        for alt in rule.alternatives:
            if alt and alt[0] not in self.rules and consumed.strip() == alt[0].strip():
                self._stack.pop()
                for symbol in reversed(alt[1:]):
                    self._stack.append(symbol)
                return

    def reset(self):
        """Reset grammar state to the start symbol."""
        self._stack = [self.start_symbol]
