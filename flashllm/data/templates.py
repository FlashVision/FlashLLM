"""Chat prompt templates for different model families."""

from abc import ABC, abstractmethod
from typing import Dict

from flashllm.registry import TEMPLATES


class BaseTemplate(ABC):
    """Base class for chat prompt templates."""

    @abstractmethod
    def format_sample(self, sample: Dict) -> str:
        """Format a single training sample into a prompt string."""
        ...

    @abstractmethod
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format an instruction into an inference prompt."""
        ...


@TEMPLATES.register("alpaca")
class AlpacaTemplate(BaseTemplate):
    """Alpaca-style prompt template.

    Format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
    """

    def format_sample(self, sample: Dict) -> str:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", sample.get("completion", ""))

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return prompt

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


@TEMPLATES.register("chatml")
class ChatMLTemplate(BaseTemplate):
    """ChatML prompt template (used by Phi-3, Qwen).

    Format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>
    """

    def format_sample(self, sample: Dict) -> str:
        system = sample.get("system", "You are a helpful assistant.")
        instruction = sample.get("instruction", sample.get("prompt", ""))
        output = sample.get("output", sample.get("completion", ""))

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        system = "You are a helpful assistant."
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@TEMPLATES.register("llama")
class LlamaTemplate(BaseTemplate):
    """Llama 3.x chat template.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {instruction}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {output}<|eot_id|>
    """

    def format_sample(self, sample: Dict) -> str:
        system = sample.get("system", "You are a helpful assistant.")
        instruction = sample.get("instruction", sample.get("prompt", ""))
        output = sample.get("output", sample.get("completion", ""))

        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        system = "You are a helpful assistant."
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


@TEMPLATES.register("mistral")
class MistralTemplate(BaseTemplate):
    """Mistral instruct template.

    Format:
        [INST] {instruction} [/INST] {output}
    """

    def format_sample(self, sample: Dict) -> str:
        instruction = sample.get("instruction", sample.get("prompt", ""))
        input_text = sample.get("input", "")
        output = sample.get("output", sample.get("completion", ""))

        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"

        return f"[INST] {prompt} [/INST] {output}"

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        return f"[INST] {prompt} [/INST] "


@TEMPLATES.register("gemma")
class GemmaTemplate(BaseTemplate):
    """Gemma chat template.

    Format:
        <start_of_turn>user
        {instruction}<end_of_turn>
        <start_of_turn>model
        {output}<end_of_turn>
    """

    def format_sample(self, sample: Dict) -> str:
        instruction = sample.get("instruction", sample.get("prompt", ""))
        output = sample.get("output", sample.get("completion", ""))

        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def get_template(name: str) -> BaseTemplate:
    """Get a template by name."""
    return TEMPLATES.build(name)
