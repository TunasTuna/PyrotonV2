"""Run only the audited, length-matched ORPO pilot on stock Qwen Coder 1.5B."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
MAX_LENGTH = 512
MAX_COMPLETION = 384
SEED = 42
EPOCHS = 2
LR = 4e-5
BETA = 0.1
LEGACY_KAGGLE_TRAINER = "69f224bd11387f094bf6cef95026066c685af0920de9b9e0f6d1ca4e64076e50"
ORIGINAL_PILOT_TRAINER = "1646d2d18c03c46fa33cff659de3cf74ad944aa586c99b30cbc0ca6ed58f3432"
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


def single_turn_completion(rendered, answer, eos):
    expected = answer + eos
    if not eos or eos in answer or rendered != expected + "\n":
        raise ValueError("The Qwen chat template did not produce the expected assistant turn.")
    return expected


def render_single_turn_pairs(messages, tokenizer, render):
    eos = tokenizer.eos_token
    rows = []
    for row in messages:
        rendered = render(row, tokenizer)
        for side in ("chosen", "rejected"):
            rendered[side] = single_turn_completion(rendered[side], row[side][0]["content"], eos)
        rows.append(rendered)
    return rows


def verify_single_turn_labels(rows, dataset, tokenizer):
    if len(rows) != len(dataset):
        raise ValueError("ORPO preprocessing changed the number of preference pairs.")
    for index, (row, encoded) in enumerate(zip(rows, dataset, strict=True), 1):
        prompt = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        for side in ("chosen", "rejected"):
            ids = tokenizer(row["prompt"] + row[side], add_special_tokens=False).input_ids
            completion = ids[len(prompt):]
            if (ids[:len(prompt)] != prompt or len(ids) > MAX_LENGTH or len(completion) > MAX_COMPLETION
                    or completion.count(tokenizer.eos_token_id) != 1 or completion[-1] != tokenizer.eos_token_id
                    or encoded[f"{side}_input_ids"] != ids
                    or encoded[f"{side}_labels"] != [-100] * len(prompt) + completion):
                raise ValueError(f"Unexpected ORPO tokens or labels in pair {index}, {side}.")
    print(f"Verified one supervised end-of-turn token in all {2 * len(rows)} completions.", flush=True)


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


def adapter_archive(checkpoint, epoch, destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"pyroton-epoch-{epoch}-adapter.zip"
    temporary = archive_path.with_suffix(".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "pyroton-complete.json"):
            archive.write(checkpoint / name, name)
        manifest = checkpoint.parent / "pilot-run.json"
        if manifest.is_file():
            archive.write(manifest, manifest.name)
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Invalid epoch {epoch} adapter archive.")
    temporary.replace(archive_path)


def training_metrics(history):
    margins = {str(epoch): [row["rewards/margins"] for row in history
                           if "rewards/margins" in row and math.ceil(row.get("epoch", 0)) == epoch
                           and math.isfinite(row["rewards/margins"])] for epoch in (1, 2)}
    means = {epoch: mean(values) if values else None for epoch, values in margins.items()}
    abnormal = []
    for row in history:
        for key in ("loss", "grad_norm", "rewards/margins", "train_loss"):
            if key in row and not math.isfinite(row[key]):
                abnormal.append(f"Non-finite {key} at step {row.get('step', 'unknown')}.")
    if any(value is None for value in means.values()):
        abnormal.append("A mean reward margin is missing for at least one epoch.")
    elif means["2"] <= means["1"]:
        abnormal.append("Mean reward margin did not grow from epoch 1 to epoch 2.")
    return {"mean_reward_margin_by_epoch": means, "abnormal": abnormal,
            "loss_and_margin_log": [{key: (value if not isinstance(value, float) or math.isfinite(value) else str(value))
                                    for key, value in row.items()} for row in history]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "quality-report", "noise-report", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--runtime", choices=("colab", "kaggle"), default="colab")
    parser.add_argument("--adapter-output-dir", type=Path)
    parser.add_argument("--single-eot", action="store_true", help="Run the single end-of-turn ablation.")
    args = parser.parse_args()
    if args.runtime == "kaggle" and args.adapter_output_dir is None and not args.validate_only:
        parser.error("Kaggle runs require --adapter-output-dir.")
    messages = load_pairs(args.pairs, args.quality_report, args.noise_report)
    if args.validate_only:
        print(f"Validated {len(messages)} audited preference pairs.")
        return

    import torch
    from datasets import Dataset
    from packaging.version import Version
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl.data_utils import maybe_apply_chat_template
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    kaggle = args.runtime == "kaggle"
    if not torch.cuda.is_available() or (kaggle and torch.cuda.device_count() != 1):
        raise RuntimeError("The training process needs exactly one visible Kaggle GPU.")
    if not kaggle and "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select a Colab T4 GPU runtime.")
    if Version(torch.__version__) < Version("2.6"):
        raise RuntimeError("Use a runtime with torch 2.6 or newer for checkpoint resume.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "pilot-run.json"
    settings = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "seed": SEED,
                "rank": 16, "alpha": 32, "runtime": args.runtime, "epochs": EPOCHS, "learning_rate": LR, "beta": BETA,
                "max_length": MAX_LENGTH, "max_completion_length": MAX_COMPLETION,
                "pairs_sha256": digest(args.pairs), "quality_sha256": digest(args.quality_report),
                "noise_sha256": digest(args.noise_report), "script_sha256": digest(Path(__file__))}
    if args.single_eot:
        settings["completion_termination"] = "single-eot"
    if manifest.exists():
        previous = json.loads(manifest.read_bytes())
        if previous != settings:
            original = {**settings, "script_sha256": ORIGINAL_PILOT_TRAINER}
            legacy = {**settings, "script_sha256": LEGACY_KAGGLE_TRAINER}
            if not args.single_eot and previous == original:
                write_json(manifest, settings)
            elif not args.single_eot and kaggle and previous == legacy and set(complete_checkpoints(args.output_dir)) == {1}:
                write_json(manifest, settings)
                print("Resuming the completed first epoch after fixing the adapter export callback.", flush=True)
            else:
                raise ValueError("The saved run uses different data or settings.")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("Use a fresh output directory for this training protocol.")
        write_json(manifest, settings)
    available = complete_checkpoints(args.output_dir)
    if set(available) == {1, 2}:
        if args.adapter_output_dir:
            for epoch, checkpoint in sorted(available.items()):
                adapter_archive(checkpoint, epoch, args.adapter_output_dir)
            state = json.loads((available[2] / "trainer_state.json").read_bytes())
            write_json(args.adapter_output_dir / "training-metrics.json", training_metrics(state["log_history"]))
        print("Both completed epoch checkpoints are already available.")
        return
    if 2 in available:
        raise ValueError("The epoch 1 checkpoint is missing.")

    class MarkComplete(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            epoch = round(state.epoch)
            checkpoint = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            if epoch not in (1, 2) or abs(epoch - state.epoch) > 1e-6:
                raise ValueError(f"Unexpected save at epoch {state.epoch}.")
            missing = CHECKPOINT_FILES - {path.name for path in checkpoint.iterdir()}
            if missing:
                raise ValueError(f"Incomplete training checkpoint. Missing {sorted(missing)}")
            write_json(checkpoint / "pyroton-complete.json", {"epoch": epoch, "step": state.global_step})
            if args.adapter_output_dir:
                adapter_archive(checkpoint, epoch, args.adapter_output_dir)

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if not kaggle:
                return
            logs = logs or {}
            flags = [key for key in ("loss", "grad_norm", "rewards/margins", "train_loss")
                     if key in logs and not math.isfinite(logs[key])]
            if flags:
                write_json(args.adapter_output_dir / "training-metrics.json", training_metrics(state.log_history))
                raise FloatingPointError(f"Non-finite training values {flags} at step {state.global_step}.")

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
    configuration = dict(output_dir=str(args.output_dir), num_train_epochs=EPOCHS, beta=BETA, learning_rate=LR,
                        per_device_train_batch_size=1, gradient_accumulation_steps=4, seed=SEED, data_seed=SEED,
                        fp16=True, bf16=False, max_length=MAX_LENGTH, max_completion_length=MAX_COMPLETION,
                        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                        save_strategy="epoch", save_only_model=False, remove_unused_columns=False,
                        report_to=[], logging_steps=1 if kaggle else 5, logging_nan_inf_filter=False)
    accepted = inspect.signature(ORPOConfig).parameters
    required = {"output_dir", "num_train_epochs", "beta", "learning_rate", "fp16", "bf16", "save_strategy"}
    if required - accepted.keys():
        raise RuntimeError(f"ORPOConfig is missing required options {sorted(required - accepted.keys())}.")
    if args.adapter_output_dir:
        args.adapter_output_dir.mkdir(parents=True, exist_ok=True)
    omitted = sorted(configuration.keys() - accepted.keys())
    if omitted:
        print(f"ORPOConfig does not support {omitted} in this version.", flush=True)
    config = ORPOConfig(**{key: value for key, value in configuration.items() if key in accepted})
    adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    if args.single_eot:
        messages = render_single_turn_pairs(messages, tokenizer, maybe_apply_chat_template)
    trainer = ORPOTrainer(model=model, args=config, processing_class=tokenizer,
                          train_dataset=Dataset.from_list(messages), peft_config=adapter, callbacks=[MarkComplete()])
    if args.single_eot:
        verify_single_turn_labels(messages, trainer.train_dataset, tokenizer)
    trainer.train(resume_from_checkpoint=str(available[1]) if 1 in available else None)
    if set(complete_checkpoints(args.output_dir)) != {1, 2}:
        raise RuntimeError("The run did not save both epoch checkpoints.")
    if args.adapter_output_dir:
        report = training_metrics(trainer.state.log_history)
        write_json(args.adapter_output_dir / "training-metrics.json", report)
        for issue in report["abnormal"]:
            print(issue, flush=True)


if __name__ == "__main__":
    main()
