"""Function/tool calling for LLMs.

Provides schema definition, JSON extraction from model outputs,
and function dispatch for tool-augmented generation.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FunctionParameter:
    """Schema for a single function parameter."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class FunctionSchema:
    """Schema describing a callable function/tool.

    Args:
        name: Function name.
        description: Human-readable description for the LLM.
        parameters: List of parameter schemas.
        returns: Description of return value.
    """
    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    returns: str = ""

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: Dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_description(self) -> str:
        """Generate a text description for prompt injection."""
        params_desc = []
        for p in self.parameters:
            req = " (required)" if p.required else " (optional)"
            params_desc.append(f"  - {p.name} ({p.type}){req}: {p.description}")
        params_text = "\n".join(params_desc) if params_desc else "  (no parameters)"
        return f"Function: {self.name}\nDescription: {self.description}\nParameters:\n{params_text}"


class FunctionCallExtractor:
    """Extract function call JSON from model output text.

    Supports multiple formats: OpenAI-style, XML-tagged, and raw JSON.
    """

    FUNCTION_CALL_PATTERNS = [
        re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
        re.compile(r"<function_call>\s*(\{.*?\})\s*</function_call>", re.DOTALL),
        re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL),
    ]

    @classmethod
    def extract(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extract a function call from generated text.

        Tries multiple patterns to find a JSON function call. Expects
        format: {"name": "function_name", "arguments": {...}}.

        Args:
            text: Model output text.

        Returns:
            Parsed function call dict or None if not found.
        """
        for pattern in cls.FUNCTION_CALL_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    call = json.loads(match.group(1))
                    if "name" in call:
                        return call
                except json.JSONDecodeError:
                    continue

        try:
            brace_start = text.index("{")
            brace_depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    brace_depth += 1
                elif text[i] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        candidate = text[brace_start:i + 1]
                        call = json.loads(candidate)
                        if "name" in call:
                            return call
                        break
        except (ValueError, json.JSONDecodeError):
            pass

        return None

    @classmethod
    def extract_multiple(cls, text: str) -> List[Dict[str, Any]]:
        """Extract multiple function calls from generated text."""
        calls = []
        for pattern in cls.FUNCTION_CALL_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    call = json.loads(match.group(1))
                    if "name" in call:
                        calls.append(call)
                except json.JSONDecodeError:
                    continue
        return calls


class FunctionDispatcher:
    """Registry and dispatcher for callable functions.

    Register Python functions with schemas, then dispatch calls
    extracted from model outputs.

    Example::

        dispatcher = FunctionDispatcher()

        @dispatcher.register(
            name="get_weather",
            description="Get weather for a city",
            parameters=[
                FunctionParameter("city", "string", "City name"),
            ]
        )
        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny, 72F"

        result = dispatcher.dispatch({"name": "get_weather", "arguments": {"city": "SF"}})
    """

    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._schemas: Dict[str, FunctionSchema] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[List[FunctionParameter]] = None,
        returns: str = "",
    ) -> Callable:
        """Register a function as a decorator.

        Args:
            name: Function name for the LLM.
            description: Description for the LLM.
            parameters: Parameter schemas.
            returns: Description of return value.
        """
        schema = FunctionSchema(
            name=name,
            description=description,
            parameters=parameters or [],
            returns=returns,
        )

        def decorator(func: Callable) -> Callable:
            self._functions[name] = func
            self._schemas[name] = schema
            return func

        return decorator

    def register_function(
        self,
        func: Callable,
        schema: FunctionSchema,
    ):
        """Register a function with an explicit schema."""
        self._functions[schema.name] = func
        self._schemas[schema.name] = schema

    def dispatch(self, call: Dict[str, Any]) -> Any:
        """Execute a function call.

        Args:
            call: Function call dict with "name" and "arguments" keys.

        Returns:
            Function return value.

        Raises:
            KeyError: If function name is not registered.
            TypeError: If arguments don't match the function signature.
        """
        name = call.get("name", "")
        arguments = call.get("arguments", {})

        if name not in self._functions:
            available = ", ".join(sorted(self._functions.keys()))
            raise KeyError(f"Function '{name}' not registered. Available: [{available}]")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                raise TypeError(f"Cannot parse arguments as JSON: {arguments}")

        return self._functions[name](**arguments)

    def get_schemas(self) -> List[FunctionSchema]:
        """Return all registered function schemas."""
        return list(self._schemas.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return schemas in OpenAI tools format."""
        return [schema.to_openai_schema() for schema in self._schemas.values()]

    def get_system_prompt(self) -> str:
        """Generate a system prompt describing available functions."""
        descriptions = [schema.to_prompt_description() for schema in self._schemas.values()]
        funcs_text = "\n\n".join(descriptions)
        return (
            "You have access to the following functions. To call a function, "
            "respond with a JSON object wrapped in <tool_call></tool_call> tags:\n"
            '<tool_call>{"name": "function_name", "arguments": {"arg": "value"}}</tool_call>\n\n'
            f"Available functions:\n\n{funcs_text}"
        )

    @property
    def available_functions(self) -> List[str]:
        return sorted(self._functions.keys())
