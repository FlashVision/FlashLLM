"""FlashLLM CLI — command-line interface for training, chat, generation, and export."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashLLM", "bold") + f" v{_get_version()}")
    print(_colored("Lightweight LLM inference and fine-tuning", "blue"))
    print()


def _get_version():
    from flashllm import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:         {platform.python_version()}")
    print(f"  OS:             {platform.system()} {platform.release()}")
    print(f"  Machine:        {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  NumPy:          {np.__version__}")
    try:
        import transformers
        print(f"  Transformers:   {transformers.__version__}")
    except ImportError:
        print("  Transformers:   Not installed")
    print(f"  CUDA:           {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:          {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:           {mem:.1f} GB")
    else:
        print("  GPU:            None (CPU only)")
    print(f"  CPU cores:      {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic model loading."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashllm  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashllm package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashllm package: {e}")
        errors.append(str(e))

    try:
        from flashllm.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashllm.generation import Sampler, BeamSearch, KVCache  # noqa: F401
        print(f"  {_colored('✓', 'green')} generation (Sampler, BeamSearch, KVCache)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} generation: {e}")
        errors.append(str(e))

    try:
        from flashllm.solutions import Chatbot, Summarizer, CodeAssistant  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (Chatbot, Summarizer, CodeAssistant)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashllm.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import transformers  # noqa: F401
        print(f"  {_colored('✓', 'green')} transformers ({transformers.__version__})")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} transformers: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (inference will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashLLM is ready.", "green"))


def cmd_train(args):
    """Train / fine-tune a model."""
    from flashllm.engine.trainer import Trainer

    if args.config:
        from flashllm.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        if not args.model:
            print(_colored("Error:", "red") + " --model or --config is required")
            sys.exit(1)
        kwargs = {
            "model_id": args.model,
            "dataset": args.dataset,
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "save_dir": args.save_dir,
        }
        if args.lora:
            kwargs["method"] = "lora"
        if args.lr:
            kwargs["learning_rate"] = args.lr
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_chat(args):
    """Interactive chat with a model."""
    from flashllm.solutions.chatbot import Chatbot

    chatbot = Chatbot(
        model_id=args.model,
        device=args.device,
        system_prompt=args.system_prompt,
    )

    _print_banner()
    print(_colored("Interactive Chat", "bold") + f" (model: {args.model})")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        try:
            user_input = input(_colored("You: ", "green"))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = chatbot.chat(user_input)
        print(f"{_colored('Assistant:', 'blue')} {response}\n")


def cmd_generate(args):
    """Generate text from a prompt."""
    from flashllm.engine.predictor import Predictor

    predictor = Predictor(model_id=args.model, device=args.device)
    text = predictor.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


def cmd_export(args):
    """Export model to ONNX or GGUF."""
    from flashllm.engine.exporter import Exporter

    exporter = Exporter(model_path=args.model)
    path = exporter.export(output=args.output, format=args.format)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def cmd_benchmark(args):
    """Benchmark model throughput and latency."""
    from flashllm.analytics.benchmark import Benchmark

    bench = Benchmark(model_id=args.model, device=args.device)
    results = bench.run(
        prompt=args.prompt or "The meaning of life is",
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
    )

    print(f"\n{_colored('Benchmark Results', 'bold')}")
    print(f"  Model:          {args.model}")
    print(f"  Device:         {args.device}")
    print(f"  Tokens/sec:     {results['tokens_per_sec']:.1f}")
    print(f"  Latency (ms):   {results['latency_ms']:.1f}")
    print(f"  Memory (MB):    {results['memory_mb']:.0f}")


def cmd_quantize(args):
    """Quantize a model."""
    from flashllm.quantization import quantize_model

    output = quantize_model(
        model_id=args.model,
        method=args.method,
        bits=args.bits,
        output_dir=args.output,
    )
    print(f"\n{_colored('✓', 'green')} Quantized model saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashllm",
        description="FlashLLM: Lightweight LLM inference and fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashllm check                              Verify installation
  flashllm chat --model meta-llama/Llama-3.1-8B-Instruct
  flashllm train --config configs/flashllm_llama_7b_sft.yaml
  flashllm generate --model microsoft/Phi-3-mini-4k-instruct --prompt "Hello"
  flashllm quantize --model meta-llama/Llama-3.1-8B --method gptq --bits 4

Documentation: https://github.com/FlashVision/FlashLLM
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # train
    train_p = subparsers.add_parser("train", help="Train / fine-tune a model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model", default=None, help="HuggingFace model ID")
    train_p.add_argument("--dataset", default=None, help="Path to training dataset")
    train_p.add_argument("--method", default="sft", choices=["sft", "lora", "qlora", "dpo"],
                         help="Training method (default: sft)")
    train_p.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    train_p.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    train_p.add_argument("--save-dir", default="workspace/train", help="Output directory")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")

    # chat
    chat_p = subparsers.add_parser("chat", help="Interactive chat with a model")
    chat_p.add_argument("--model", required=True, help="HuggingFace model ID")
    chat_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    chat_p.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for the conversation")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_p.add_argument("--model", required=True, help="HuggingFace model ID")
    gen_p.add_argument("--prompt", required=True, help="Input prompt")
    gen_p.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    gen_p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    gen_p.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling (default: 0.9)")
    gen_p.add_argument("--device", default="cuda", help="Device (default: cuda)")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX/GGUF")
    exp_p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    exp_p.add_argument("--output", default="model.onnx", help="Output path")
    exp_p.add_argument("--format", default="onnx", choices=["onnx", "gguf"], help="Export format (default: onnx)")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark model speed")
    bench_p.add_argument("--model", required=True, help="HuggingFace model ID")
    bench_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    bench_p.add_argument("--prompt", default=None, help="Prompt for benchmarking")
    bench_p.add_argument("--max-tokens", type=int, default=128, help="Tokens per generation (default: 128)")
    bench_p.add_argument("--num-runs", type=int, default=10, help="Number of runs (default: 10)")

    # quantize
    quant_p = subparsers.add_parser("quantize", help="Quantize a model")
    quant_p.add_argument("--model", required=True, help="HuggingFace model ID")
    quant_p.add_argument("--method", default="gptq", choices=["gptq", "awq", "bitsandbytes"],
                         help="Quantization method (default: gptq)")
    quant_p.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (default: 4)")
    quant_p.add_argument("--output", default=None, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "chat": cmd_chat,
        "generate": cmd_generate,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
        "quantize": cmd_quantize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
