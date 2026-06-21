"""Benchmark model throughput and latency.

Usage:
    python examples/benchmark_model.py
    python examples/benchmark_model.py --model microsoft/Phi-3-mini-4k-instruct --num-runs 20
"""

import argparse

from flashllm.analytics import Benchmark


def main():
    parser = argparse.ArgumentParser(description="Model benchmarking")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--prompt", default="The meaning of life is", help="Benchmark prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Tokens per run")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print(f"  FlashLLM — Model Benchmark")
    print(f"{'=' * 50}")
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Runs:       {args.num_runs}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"{'=' * 50}\n")

    bench = Benchmark(model_id=args.model, device=args.device)
    results = bench.run(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup,
    )

    print(f"\n{'=' * 50}")
    print(f"  Results")
    print(f"{'=' * 50}")
    print(f"  Tokens/sec:     {results['tokens_per_sec']:.1f}")
    print(f"  Latency (ms):   {results['latency_ms']:.1f}")
    print(f"  Memory (MB):    {results['memory_mb']:.0f}")
    print(f"  Parameters:     {results['params']:,}")
    print(f"  Model size:     {results['params_gb']:.2f} GB")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
