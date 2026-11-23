"""Prepare dependencies and run or resume the audited single-GPU Kaggle pilot."""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
import subprocess
import sys

from train_step3_pilot import adapter_archive, complete_checkpoints, training_metrics, write_json


PINS = {"transformers": "4.57.6", "trl": "0.29.0", "peft": "0.18.1", "accelerate": "1.12.0",
        "datasets": "4.5.0", "huggingface-hub": "0.36.0", "tokenizers": "0.22.2", "pyarrow": "22.0.0",
        "fsspec": "2025.10.0", "safetensors": "0.7.0", "dill": "0.4.0", "multiprocess": "0.70.18"}


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def prepare(working):
    protections = {name: installed(name) for name in ("torch", "torchao")}
    if not protections["torch"]:
        raise RuntimeError("Kaggle must provide its GPU-enabled torch installation.")
    constraints = working / "runtime-constraints.txt"
    constraints.write_text("".join(f"{name}=={value}\n" for name, value in protections.items() if value))
    if any(installed(name) != value for name, value in PINS.items()):
        subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(constraints),
                        *(f"{name}=={value}" for name, value in PINS.items())], check=True)
    if installed("torch") != protections["torch"] or installed("torchao") != protections["torchao"]:
        raise RuntimeError("The Kaggle dependency setup changed a protected runtime package.")
    probe = [sys.executable, "-c", "import peft"]
    outcome = subprocess.run(probe, capture_output=True, text=True)
    if outcome.returncode and installed("torchao") and "torchao" in outcome.stderr.lower():
        print(outcome.stderr, flush=True)
        print("Removing torchao because the PEFT import failed in it.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    subprocess.run([sys.executable, "-c", "import peft, torch; "
                    "from trl.experimental.orpo import ORPOConfig, ORPOTrainer; "
                    "assert torch.cuda.device_count() == 1; print(torch.__version__, torch.cuda.get_device_name(0))"],
                   check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})


def check_variant(root, single_eot, completion_only):
    manifest = json.loads((root / "pilot-run.json").read_bytes())
    expected_eot = "single-eot" if single_eot else None
    expected_nll = "completion-only" if completion_only else None
    if manifest.get("completion_termination") != expected_eot or manifest.get("chosen_nll") != expected_nll:
        raise ValueError("The checkpoints belong to a different training variant.")


def restore(source, destination, single_eot, completion_only):
    if destination.exists() or not source.exists():
        return
    checkpoints = complete_checkpoints(source)
    if 1 not in checkpoints or not (source / "pilot-run.json").is_file():
        raise ValueError("The attached resume input lacks a complete first epoch or run manifest.")
    check_variant(source, single_eot, completion_only)
    temporary = destination.with_name(destination.name + ".restoring")
    temporary.mkdir(exist_ok=True)
    shutil.copy2(source / "pilot-run.json", temporary)
    for checkpoint in checkpoints.values():
        shutil.copytree(checkpoint, temporary / checkpoint.name, dirs_exist_ok=True)
    if set(complete_checkpoints(temporary)) != set(checkpoints):
        raise ValueError("The imported checkpoints did not validate.")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "prepare", "train", "export"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--single-eot", action="store_true")
    parser.add_argument("--completion-only-nll", action="store_true")
    args = parser.parse_args()
    if args.completion_only_nll and not args.single_eot:
        parser.error("--completion-only-nll requires --single-eot.")
    if args.completion_only_nll:
        args.working = args.working / "single-eot-completion-only"
    elif args.single_eot:
        args.working = args.working / "single-eot"
    elif args.resume is None:
        args.resume = Path("/kaggle/input/pyroton-v2-resume/checkpoints")
    args.working.mkdir(parents=True, exist_ok=True)
    checkpoints = args.working / "checkpoints"
    if args.mode == "prepare":
        prepare(args.working)
        return
    if args.mode == "train":
        prepare(args.working)
    if args.mode in {"train", "export"} and args.resume is not None:
        restore(args.resume, checkpoints, args.single_eot, args.completion_only_nll)
    if args.mode == "export":
        saved = complete_checkpoints(checkpoints)
        if not saved:
            raise RuntimeError("There are no completed epoch checkpoints to export.")
        check_variant(checkpoints, args.single_eot, args.completion_only_nll)
        for epoch, checkpoint in sorted(saved.items()):
            adapter_archive(checkpoint, epoch, args.working)
        state = json.loads((saved[max(saved)] / "trainer_state.json").read_bytes())
        report_path = args.working / "training-metrics.json"
        if not report_path.exists():
            write_json(report_path, training_metrics(state["log_history"]))
        return
    command = [sys.executable, str(Path(__file__).with_name("train_step3_pilot.py")), "--runtime", "kaggle",
               "--pairs", str(args.inputs / "orpo-length-matched.jsonl"),
               "--quality-report", str(args.inputs / "quality-report.json"),
               "--noise-report", str(args.inputs / "stock-noise.json"),
               "--output-dir", str(checkpoints), "--adapter-output-dir", str(args.working)]
    if args.mode == "validate":
        command.append("--validate-only")
    if args.single_eot:
        command.append("--single-eot")
    if args.completion_only_nll:
        command.append("--completion-only-nll")
    subprocess.run(command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})


if __name__ == "__main__":
    main()
