"""Fine-tune a model with LoRA on custom instruction data.

Usage:
    python examples/finetune_lora.py
    python examples/finetune_lora.py --model mistralai/Mistral-7B-v0.3 --dataset data/my_data.jsonl
"""

import argparse

from flashllm import Trainer, apply_lora, merge_lora_weights


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Base model ID")
    parser.add_argument("--dataset", default="data/instructions.jsonl", help="Training data (JSONL)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--save-dir", default="workspace/lora_finetune", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights after training")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  FlashLLM — LoRA Fine-Tuning")
    print(f"{'='*50}")
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  LR:        {args.lr}")
    print(f"{'='*50}\n")

    trainer = Trainer(
        model_id=args.model,
        dataset=args.dataset,
        method="lora",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )

    trainer.train()

    if args.merge:
        print("\nMerging LoRA weights into base model...")
        merge_lora_weights(trainer.model)
        trainer.model.save_pretrained(f"{args.save_dir}/merged")
        print(f"Merged model saved to: {args.save_dir}/merged")


if __name__ == "__main__":
    main()
