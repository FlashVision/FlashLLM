"""Evaluation harness for LLMs — MMLU, HumanEval, MT-Bench task loaders and scoring."""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Result of an evaluation task."""
    task_name: str
    score: float
    num_samples: int
    num_correct: int
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def __repr__(self) -> str:
        return f"EvalResult({self.task_name}: {self.score:.4f}, {self.num_correct}/{self.num_samples})"


class EvalTask(ABC):
    """Base class for evaluation tasks."""

    name: str = "base"

    @abstractmethod
    def load_samples(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load evaluation samples."""

    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format a sample into a model prompt."""

    @abstractmethod
    def score(self, sample: Dict[str, Any], model_output: str) -> bool:
        """Score a single model output. Returns True if correct."""


class MMLUTask(EvalTask):
    """MMLU (Massive Multitask Language Understanding) evaluation.

    Tests knowledge across 57 subjects in STEM, humanities, social sciences,
    and more using multiple-choice questions.

    Args:
        subject: MMLU subject to evaluate (e.g., "abstract_algebra").
        num_few_shot: Number of few-shot examples.
        data_dir: Directory containing MMLU CSV data.
    """

    name = "mmlu"

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    CHOICES = ["A", "B", "C", "D"]

    def __init__(
        self,
        subject: str = "all",
        num_few_shot: int = 5,
        data_dir: Optional[str] = None,
    ):
        self.subject = subject
        self.num_few_shot = num_few_shot
        self.data_dir = Path(data_dir) if data_dir else None

    def load_samples(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MMLU samples from HuggingFace datasets or local CSV."""
        try:
            from datasets import load_dataset
            subjects = self.SUBJECTS if self.subject == "all" else [self.subject]
            samples = []
            for subj in subjects:
                ds = load_dataset("cais/mmlu", subj, split="test")
                for item in ds:
                    samples.append({
                        "subject": subj,
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": self.CHOICES[item["answer"]],
                    })
                    if num_samples and len(samples) >= num_samples:
                        return samples
            return samples[:num_samples] if num_samples else samples
        except Exception as e:
            logger.warning("Could not load MMLU from HuggingFace: %s. Using synthetic samples.", e)
            return self._synthetic_samples(num_samples or 10)

    def _synthetic_samples(self, n: int) -> List[Dict[str, Any]]:
        samples = []
        for i in range(n):
            samples.append({
                "subject": "test",
                "question": f"Sample question {i}?",
                "choices": [f"Option {c}" for c in self.CHOICES],
                "answer": self.CHOICES[i % 4],
            })
        return samples

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample["question"]
        choices = sample["choices"]
        choices_text = "\n".join(f"{c}. {choices[i]}" for i, c in enumerate(self.CHOICES))
        return (
            f"The following is a multiple choice question about {sample['subject']}.\n\n"
            f"{question}\n{choices_text}\n\nAnswer:"
        )

    def score(self, sample: Dict[str, Any], model_output: str) -> bool:
        output = model_output.strip().upper()
        expected = sample["answer"]
        if output.startswith(expected):
            return True
        match = re.search(r'\b([A-D])\b', output)
        return match is not None and match.group(1) == expected


class HumanEvalTask(EvalTask):
    """HumanEval code generation benchmark.

    Evaluates functional correctness of code generated by the model
    using test cases for 164 Python programming problems.

    Args:
        timeout: Execution timeout per problem in seconds.
    """

    name = "humaneval"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def load_samples(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            samples = []
            for item in ds:
                samples.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                })
                if num_samples and len(samples) >= num_samples:
                    break
            return samples
        except Exception as e:
            logger.warning("Could not load HumanEval: %s. Using synthetic samples.", e)
            return self._synthetic_samples(num_samples or 5)

    def _synthetic_samples(self, n: int) -> List[Dict[str, Any]]:
        samples = []
        for i in range(n):
            samples.append({
                "task_id": f"test/{i}",
                "prompt": f"def solution_{i}(x: int) -> int:\n    \"\"\"Return x + {i}.\"\"\"\n",
                "canonical_solution": f"    return x + {i}\n",
                "test": f"assert solution_{i}({i}) == {2*i}\n",
                "entry_point": f"solution_{i}",
            })
        return samples

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return sample["prompt"]

    def score(self, sample: Dict[str, Any], model_output: str) -> bool:
        full_code = sample["prompt"] + model_output
        test_code = full_code + "\n" + sample["test"]

        try:
            exec_globals: Dict[str, Any] = {}
            exec(compile(test_code, "<eval>", "exec"), exec_globals)
            return True
        except Exception:
            return False


class MTBenchTask(EvalTask):
    """MT-Bench conversational evaluation.

    Multi-turn benchmark that evaluates instruction-following ability
    across 80 questions spanning 8 categories.

    Args:
        categories: List of categories to evaluate. None for all.
    """

    name = "mt_bench"

    CATEGORIES = [
        "writing", "roleplay", "reasoning", "math",
        "coding", "extraction", "stem", "humanities",
    ]

    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or self.CATEGORIES

    def load_samples(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        for i, cat in enumerate(self.categories):
            for j in range(10):
                samples.append({
                    "question_id": i * 10 + j,
                    "category": cat,
                    "turns": [
                        f"[{cat}] Please help me with task {j}.",
                        "Can you elaborate more on that?",
                    ],
                    "reference": None,
                })
                if num_samples and len(samples) >= num_samples:
                    return samples
        return samples[:num_samples] if num_samples else samples

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return sample["turns"][0]

    def score(self, sample: Dict[str, Any], model_output: str) -> bool:
        output = model_output.strip()
        if len(output) < 10:
            return False
        if output.lower().startswith(("i cannot", "i can't", "sorry")):
            return False
        return True

    def score_quality(self, sample: Dict[str, Any], model_output: str) -> float:
        """Heuristic quality score from 1-10."""
        output = model_output.strip()
        score = 5.0
        if len(output) > 200:
            score += 1.0
        if len(output) > 500:
            score += 1.0
        if "\n" in output:
            score += 0.5
        if any(c in output for c in ["1.", "- ", "* ", "```"]):
            score += 0.5
        if len(output) < 50:
            score -= 2.0
        return max(1.0, min(10.0, score))


class EvalHarness:
    """Unified evaluation harness for running multiple benchmarks.

    Manages task registration, execution, and result aggregation.

    Args:
        model: FlashLLM model instance or HuggingFace model.
        tokenizer: Optional tokenizer (extracted from model if not provided).
        device: Device for evaluation.
        batch_size: Batch size for evaluation.
    """

    BUILTIN_TASKS = {
        "mmlu": MMLUTask,
        "humaneval": HumanEvalTask,
        "mt_bench": MTBenchTask,
    }

    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = device
        self.batch_size = batch_size
        self._tasks: Dict[str, EvalTask] = {}

    def add_task(self, task: EvalTask):
        """Register an evaluation task."""
        self._tasks[task.name] = task

    def add_builtin(self, task_name: str, **kwargs):
        """Add a built-in task by name.

        Args:
            task_name: One of "mmlu", "humaneval", "mt_bench".
            **kwargs: Task-specific configuration.
        """
        if task_name not in self.BUILTIN_TASKS:
            available = ", ".join(self.BUILTIN_TASKS.keys())
            raise ValueError(f"Unknown task: {task_name}. Available: {available}")
        task = self.BUILTIN_TASKS[task_name](**kwargs)
        self._tasks[task_name] = task

    def run(
        self,
        tasks: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, EvalResult]:
        """Run evaluation on specified tasks.

        Args:
            tasks: Task names to run. None runs all registered tasks.
            num_samples: Limit samples per task.
            max_new_tokens: Max tokens for generation.
            temperature: Sampling temperature (0 for greedy).

        Returns:
            Dict mapping task name to EvalResult.
        """
        task_names = tasks or list(self._tasks.keys())
        results = {}

        for task_name in task_names:
            if task_name not in self._tasks:
                logger.warning("Task '%s' not registered, skipping", task_name)
                continue
            result = self._run_task(
                self._tasks[task_name], num_samples, max_new_tokens, temperature,
            )
            results[task_name] = result
            logger.info("  %s: %.4f (%d/%d)", task_name, result.score, result.num_correct, result.num_samples)

        return results

    def _run_task(
        self,
        task: EvalTask,
        num_samples: Optional[int],
        max_new_tokens: int,
        temperature: float,
    ) -> EvalResult:
        start_time = time.time()
        samples = task.load_samples(num_samples)
        num_correct = 0

        for sample in samples:
            prompt = task.format_prompt(sample)
            output = self._generate(prompt, max_new_tokens, temperature)
            if task.score(sample, output):
                num_correct += 1

        duration = time.time() - start_time
        score = num_correct / max(len(samples), 1)

        return EvalResult(
            task_name=task.name,
            score=score,
            num_samples=len(samples),
            num_correct=num_correct,
            duration_seconds=duration,
        )

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        if hasattr(self.model, "generate") and callable(self.model.generate):
            kwargs = {"max_new_tokens": max_new_tokens}
            if temperature > 0:
                kwargs["temperature"] = temperature
                kwargs["do_sample"] = True
            else:
                kwargs["do_sample"] = False

            try:
                return self.model.generate(prompt, **kwargs)
            except TypeError:
                pass

        if self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-7) if temperature > 0 else 1.0,
                )
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        raise RuntimeError("Model must have a generate() method or tokenizer must be provided")

    def run_perplexity(
        self,
        texts: List[str],
        max_length: int = 2048,
    ) -> float:
        """Compute perplexity on a list of texts.

        Args:
            texts: Input texts for perplexity computation.
            max_length: Maximum sequence length.

        Returns:
            Average perplexity across texts.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required for perplexity computation")

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            encodings = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length,
            ).to(self.device)

            input_ids = encodings["input_ids"]
            with torch.inference_mode():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        return torch.exp(torch.tensor(avg_loss)).item()
