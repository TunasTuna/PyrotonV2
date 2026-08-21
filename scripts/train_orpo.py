"""
train_orpo.py — ORPO fine-tuning for Pyroton on top of the existing adapter.

ORPO (Odds Ratio Preference Optimization) trains directly on (prompt, chosen,
rejected) triples without a separate reference model or reward model -- it
folds the "prefer chosen over rejected" signal into a single loss term added
to standard SFT loss. That makes it a good fit for a lightweight project like
this: no second model to load, no extra VRAM for a frozen reference copy.

Usage:
    python scripts/train_orpo.py \
        --data data/orpo_pairs.jsonl \
        --base-adapter shohuu/pyroton-primefix-v3 \
        --output-dir ./pyroton-orpo-v1 \
        --epochs 2

After training, push to the Hub with --push --hub-id yourname/pyroton-orpo-v1
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOConfig, ORPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/orpo_pairs.jsonl"))
    p.add_argument("--base-adapter", type=str, default="shohuu/pyroton-primefix-v3",
                    help="Existing adapter to continue from. Pass 'none' to start a fresh LoRA.")
    p.add_argument("--output-dir", type=str, default="./pyroton-orpo-v1")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5,
                    help="ORPO typically wants a lower LR than SFT; 5e-5 is a safe starting point")
    p.add_argument("--beta", type=float, default=0.1,
                    help="ORPO lambda weighting the odds-ratio preference term vs. NLL loss")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--push", action="store_true")
    p.add_argument("--hub-id", type=str, default=None)
    return p.parse_args()


def load_base_and_adapter(base_adapter: str):
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if base_adapter.lower() == "none":
        print("Starting a fresh LoRA adapter (no prior weights).")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        return base, tokenizer, None

    print(f"Loading existing adapter: {base_adapter}")
    model = PeftModel.from_pretrained(base, base_adapter, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(base_adapter)
    return model, tokenizer, base_adapter


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. Run prepare_dataset.py first to generate ORPO pairs."
        )

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    print(f"Loaded {len(dataset)} preference pairs from {args.data}")

    model, tokenizer, existing_adapter = load_base_and_adapter(args.base_adapter)
    tokenizer.pad_token = tokenizer.eos_token

    # If starting fresh (no existing adapter), attach a new LoRA config.
    if existing_adapter is None:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting ORPO training...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push:
        if not args.hub_id:
            raise ValueError("--push requires --hub-id yourname/repo-name")
        print(f"Pushing to Hub: {args.hub_id}")
        trainer.model.push_to_hub(args.hub_id)
        tokenizer.push_to_hub(args.hub_id)

    print("Done.")


if __name__ == "__main__":
    main()
