"""Merge both completed epoch adapters into the pinned base and quantize Q4_K_M."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from train_step3_pilot import BASE_MODEL, BASE_REVISION, complete_checkpoints, write_json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-root", "converter", "quantize", "scratch", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = complete_checkpoints(args.checkpoint_root)
    if set(checkpoints) != {1, 2}:
        raise ValueError("Both completed training epochs are required.")
    training = json.loads((args.checkpoint_root / "pilot-run.json").read_bytes())
    if training["base_model"] != BASE_MODEL or training["base_revision"] != BASE_REVISION:
        raise ValueError("Checkpoint base differs from the pinned stock model.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(args.converter.parent), "rev-parse", "HEAD"], text=True).strip()
    if commit != "6f311d7ad63dce97a3ad1deaafb7cc044bb8c26c":
        raise ValueError("Use the llama.cpp b12521 converter checkout.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    for epoch, checkpoint in sorted(checkpoints.items()):
        output = args.output_dir / f"epoch-{epoch}-Q4_K_M.gguf"
        manifest = output.with_suffix(".json")
        provenance = {"epoch": epoch, "base": BASE_MODEL, "revision": BASE_REVISION,
                      "adapter_sha256": digest(checkpoint / "adapter_model.safetensors"),
                      "llama_cpp_commit": commit, "quantizer_sha256": digest(args.quantize),
                      "quantization": "Q4_K_M"}
        if output.is_file() and manifest.is_file():
            saved = json.loads(manifest.read_bytes())
            if saved == {**provenance, "gguf_sha256": digest(output)}:
                print(f"Using existing epoch {epoch} export.", flush=True)
                continue
            raise ValueError(f"Existing epoch {epoch} GGUF differs from its manifest.")
        with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
            scratch = Path(scratch)
            hf = scratch / "merged"
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float16)
            model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload(safe_merge=True)
            model.save_pretrained(hf, safe_serialization=True)
            tokenizer.save_pretrained(hf)
            del model, base
            f16, gguf = scratch / "merged-f16.gguf", scratch / "merged-Q4_K_M.gguf"
            subprocess.run([sys.executable, str(args.converter), str(hf), "--outtype", "f16", "--outfile", str(f16)], check=True)
            subprocess.run([str(args.quantize), str(f16), str(gguf), "Q4_K_M"], check=True)
            temporary = output.with_suffix(".tmp")
            shutil.copyfile(gguf, temporary)
            sha256 = digest(temporary)
            temporary.replace(output)
            write_json(manifest, {**provenance, "gguf_sha256": sha256})
            print(f"Verified epoch {epoch} GGUF {sha256}.", flush=True)


if __name__ == "__main__":
    main()
