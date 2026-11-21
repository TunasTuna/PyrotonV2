"""Run only the audited, length-matched ORPO pilot on stock Qwen Coder 1.5B."""

import argparse
import hashlib
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
MAX_LENGTH = 512
MAX_COMPLETION = 384
SEED = 42
EPOCHS = 2
LR = 4e-5
BETA = 0.1
CHECKPOINT_FILES = {"adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pairs(pairs_path, quality_path, noise_path):
    quality = json.loads(quality_path.read_bytes())
    noise = json.loads(noise_path.read_bytes())
    if quality["draws"] != 2720 or quality["original_zero_pass_count"] != 167:
        raise ValueError("Use the completed Step 2b data-quality report.")
    if set(noise["suites"]) != {"dev", "mbppplus"} or noise["decoding_seed"] != SEED:
        raise ValueError("The stock repeat report must cover dev and MBPP+ only.")
    for suite, count in (("dev", 60), ("mbppplus", 162)):
        scope = noise["suites"][suite]
        if (scope["stock_1"]["tasks"] != count or scope["stock_2"]["tasks"] != count
                or scope["stock_configuration"]["slots"] != 1 or scope["stock_configuration"]["new_tokens"] != 512
                or scope["stock_configuration"]["temperature"] != 0):
            raise ValueError("The stock repeat report has incompatible settings.")
    pairs = [json.loads(line) for line in pairs_path.read_bytes().splitlines() if line.strip()]
    if not pairs or len(pairs) != quality["length_matched_pairs"]["pairs"]:
        raise ValueError("Pair count differs from the completed data-quality report.")
    if digest(pairs_path) != quality["length_matched_pairs_sha256"]:
        raise ValueError("Pair data differs from the audited file.")
    ids = set()
    messages = []
    for row in pairs:
        short, long = sorted((row["chosen_tokens"], row["rejected_tokens"]))
        if not (short > 0 and 4 * long <= 5 * short and row["chosen"] != row["rejected"]
                and row["source_split"] in {"train", "validation"} and row["problem_id"] not in ids):
            raise ValueError(f"Invalid preference pair for {row['problem_id']}.")
        ids.add(row["problem_id"])
        messages.append({"prompt": row["prompt"],
                         "chosen": [{"role": "assistant", "content": row["chosen"]}],
                         "rejected": [{"role": "assistant", "content": row["rejected"]}]})
    return messages


def complete_checkpoints(root):
    checkpoints = {}
    for marker in root.glob("checkpoint-*/pyroton-complete.json"):
        meta = json.loads(marker.read_bytes())
        epoch, step = meta["epoch"], meta["step"]
        if (epoch not in (1, 2) or marker.parent.name != f"checkpoint-{step}"
                or not CHECKPOINT_FILES <= {path.name for path in marker.parent.iterdir()}):
            raise ValueError(f"Checkpoint is incomplete or mislabeled: {marker.parent}")
        state = json.loads((marker.parent / "trainer_state.json").read_bytes())
        if state["global_step"] != step or state["epoch"] != epoch or epoch in checkpoints:
            raise ValueError(f"Checkpoint has inconsistent state: {marker.parent}")
        checkpoints[epoch] = marker.parent
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "quality-report", "noise-report", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    messages = load_pairs(args.pairs, args.quality_report, args.noise_report)
    if args.validate_only:
        print(f"Validated {len(messages)} audited preference pairs.")
        return

    import torch
    from datasets import Dataset
    from packaging.version import Version
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select a Colab T4 GPU runtime.")
    if Version(torch.__version__) < Version("2.6"):
        raise RuntimeError("Use a Colab runtime with torch 2.6 or newer for checkpoint resume.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "pilot-run.json"
    settings = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "seed": SEED,
                "rank": 16, "epochs": EPOCHS, "learning_rate": LR, "beta": BETA,
                "max_length": MAX_LENGTH, "max_completion_length": MAX_COMPLETION,
                "pairs_sha256": digest(args.pairs), "quality_sha256": digest(args.quality_report),
                "noise_sha256": digest(args.noise_report), "script_sha256": digest(Path(__file__))}
    if manifest.exists():
        if json.loads(manifest.read_bytes()) != settings:
            raise ValueError("The saved run uses different data or settings.")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("Use a fresh Drive output directory for this training protocol.")
        write_json(manifest, settings)
    available = complete_checkpoints(args.output_dir)
    if set(available) == {1, 2}:
        print("Both completed epoch checkpoints are already available.")
        return
    if 2 in available:
        raise ValueError("The epoch 1 checkpoint is missing.")

    class MarkComplete(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            epoch = round(state.epoch)
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if epoch not in (1, 2) or abs(epoch - state.epoch) > 1e-6:
                raise ValueError(f"Unexpected save at epoch {state.epoch}.")
            missing = CHECKPOINT_FILES - {path.name for path in checkpoint.iterdir()}
            if missing:
                raise ValueError(f"Incomplete training checkpoint. Missing {sorted(missing)}")
            write_json(checkpoint / "pyroton-complete.json", {"epoch": epoch, "step": state.global_step})

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for row in messages:
        prompt_size = len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True))
        for preference in ("chosen", "rejected"):
            full_size = len(tokenizer.apply_chat_template(row["prompt"] + row[preference]))
            if full_size > MAX_LENGTH or full_size - prompt_size > MAX_COMPLETION:
                raise ValueError("A preference pair would be truncated. Review its token budget.")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float16)
    config = ORPOConfig(output_dir=str(args.output_dir), num_train_epochs=EPOCHS, beta=BETA, learning_rate=LR,
                        per_device_train_batch_size=1, gradient_accumulation_steps=4, seed=SEED, data_seed=SEED,
                        fp16=True, bf16=False, max_length=MAX_LENGTH, max_completion_length=MAX_COMPLETION,
                        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                        save_strategy="epoch", save_only_model=False, remove_unused_columns=False,
                        report_to=[], logging_steps=5)
    adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = ORPOTrainer(model=model, args=config, processing_class=tokenizer,
                          train_dataset=Dataset.from_list(messages), peft_config=adapter, callbacks=[MarkComplete()])
    trainer.train(resume_from_checkpoint=str(available[1]) if 1 in available else None)
    if set(complete_checkpoints(args.output_dir)) != {1, 2}:
        raise RuntimeError("The run did not save both epoch checkpoints.")


if __name__ == "__main__":
    main()
