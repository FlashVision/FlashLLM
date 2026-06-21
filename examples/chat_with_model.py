"""Interactive multi-turn chat with a language model.

Usage:
    python examples/chat_with_model.py
    python examples/chat_with_model.py --model mistralai/Mistral-7B-v0.3 --device cuda
"""

import argparse

from flashllm.solutions.chatbot import Chatbot


def main():
    parser = argparse.ArgumentParser(description="Chat with a language model")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--system-prompt", default="You are a helpful AI assistant.", help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  FlashLLM Chat — {args.model}")
    print(f"{'='*50}")
    print(f"  Device: {args.device}")
    print(f"  System: {args.system_prompt}")
    print(f"  Type 'quit' to exit, 'reset' to clear history")
    print(f"{'='*50}\n")

    chatbot = Chatbot(
        model_id=args.model,
        device=args.device,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif user_input.strip().lower() == "reset":
            chatbot.reset()
            print("[History cleared]\n")
            continue
        elif not user_input.strip():
            continue

        response = chatbot.chat(user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
