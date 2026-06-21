"""Generate text with various sampling strategies.

Usage:
    python examples/generate_text.py
    python examples/generate_text.py --model microsoft/Phi-3-mini-4k-instruct --prompt "Hello"
"""

import argparse

from flashllm import Predictor


def main():
    parser = argparse.ArgumentParser(description="Text generation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--prompt", default="The future of artificial intelligence is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  FlashLLM — Text Generation")
    print(f"{'='*50}")
    print(f"  Model:       {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p:       {args.top_p}")
    print(f"{'='*50}\n")

    predictor = Predictor(model_id=args.model, device=args.device)

    print(f"Prompt: {args.prompt}\n")
    print("Generated:")
    print("-" * 40)

    if args.stream:
        for token in predictor.stream(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            print(token, end="", flush=True)
        print()
    else:
        output = predictor.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(output)

    print("-" * 40)


if __name__ == "__main__":
    main()
