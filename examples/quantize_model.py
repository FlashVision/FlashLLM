"""Quantize a model using GPTQ, AWQ, or bitsandbytes.

Usage:
    python examples/quantize_model.py
    python examples/quantize_model.py --model mistralai/Mistral-7B-v0.3 --method awq
"""

import argparse

from flashllm.quantization import quantize_model


def main():
    parser = argparse.ArgumentParser(description="Model quantization")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model ID to quantize")
    parser.add_argument("--method", default="gptq", choices=["gptq", "awq", "bitsandbytes"], help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output or f"{args.model.split('/')[-1]}-{args.method}-{args.bits}bit"

    print(f"\n{'=' * 50}")
    print(f"  FlashLLM — Model Quantization")
    print(f"{'=' * 50}")
    print(f"  Model:   {args.model}")
    print(f"  Method:  {args.method}")
    print(f"  Bits:    {args.bits}")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 50}\n")

    result_path = quantize_model(
        model_id=args.model,
        method=args.method,
        bits=args.bits,
        output_dir=output_dir,
    )

    print(f"\n✓ Quantized model saved to: {result_path}")


if __name__ == "__main__":
    main()
