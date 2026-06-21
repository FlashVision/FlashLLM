"""Code generation and completion assistant."""



class CodeAssistant:
    """AI-powered code generation, completion, and explanation.

    Args:
        model_id: HuggingFace model ID.
        device: Device for inference.
        max_tokens: Maximum tokens in generated code.
        temperature: Sampling temperature.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B", device: str = "cuda",
                 max_tokens: int = 1024, temperature: float = 0.2):
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from flashllm.engine.predictor import Predictor
            self._predictor = Predictor(model_id=self.model_id, device=self.device)
        return self._predictor

    def generate(self, instruction: str, language: str = "python") -> str:
        """Generate code from a natural language instruction."""
        prompt = f"Write {language} code for the following task. Only output the code.\n\nTask: {instruction}\n\n```{language}\n"
        response = self.predictor.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        if "```" in response:
            response = response.split("```")[0]
        return response.strip()

    def complete(self, code_prefix: str, language: str = "python") -> str:
        """Complete partial code."""
        prompt = f"Complete the following {language} code:\n\n```{language}\n{code_prefix}"
        response = self.predictor.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        if "```" in response:
            response = response.split("```")[0]
        return response.strip()

    def explain(self, code: str, language: str = "python") -> str:
        """Explain what a piece of code does."""
        prompt = f"Explain the following {language} code:\n\n```{language}\n{code}\n```\n\nExplanation:"
        return self.predictor.generate(prompt, max_tokens=self.max_tokens, temperature=0.3)
