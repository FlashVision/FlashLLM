"""Interactive chatbot solution with multi-turn conversation."""

from typing import Dict, List


class Chatbot:
    """Multi-turn conversational chatbot.

    Args:
        model_id: HuggingFace model ID.
        device: Device for inference.
        system_prompt: System prompt for the conversation.
        max_history: Maximum number of turns to keep in history.
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        system_prompt: str = "You are a helpful, harmless, and honest AI assistant.",
        max_history: int = 20,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model_id = model_id
        self.device = device
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from flashllm.engine.predictor import Predictor

            self._predictor = Predictor(model_id=self.model_id, device=self.device)
        return self._predictor

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        response = self.predictor.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2) :]
        return response

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
