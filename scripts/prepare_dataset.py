"""
prepare_dataset.py — Build an ORPO preference dataset for Pyroton.

Pipeline:
  1. Load the current Pyroton adapter (base + LoRA) on Qwen2.5-Coder-0.5B-Instruct.
  2. For each seed instruction, sample N candidate completions (temperature > 0).
  3. Score each candidate with a sandboxed execution check (does it run? does it
     pass basic property tests derived from the instruction, when available?).
  4. Pick one PASSING completion as "chosen" and one FAILING completion as
     "rejected" per prompt -> this is the core signal ORPO needs.
  5. Merge in hand-curated repair pairs (buggy -> fixed), e.g. from primefix-v3
     work, as additional high-confidence (chosen, rejected) pairs.
  6. Write everything to data/orpo_pairs.jsonl in the standard ORPO format:
        {"prompt": ..., "chosen": ..., "rejected": ...}

Usage:
    python scripts/prepare_dataset.py \
        --seed-file data/seed_instructions.jsonl \
        --repair-file data/repair_pairs.jsonl \
        --out data/orpo_pairs.jsonl \
        --num-candidates 6 \
        --adapter shohuu/pyroton-primefix-v3

Notes on safety:
  Candidate code is executed in a subprocess with a hard timeout and no
  network access assumptions. This is NOT a full sandbox (no seccomp/container
  isolation) -- do not run untrusted model output from an unknown source here.
  Since candidates come from your own fine-tuned model on your own prompts,
  risk is low, but keep this script off of production/shared machines.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
EXEC_TIMEOUT_SECONDS = 5


# --------------------------------------------------------------------------- #
# Model loading + generation
# --------------------------------------------------------------------------- #

def load_model(adapter_id: str):
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loading adapter: {adapter_id}")
    model = PeftModel.from_pretrained(base, adapter_id)
    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, instruction: str, n: int, max_new_tokens: int = 256):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    candidates = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            completion = text[len(prompt):].strip()
            candidates.append(completion)
    return candidates


# --------------------------------------------------------------------------- #
# Execution-based scoring
# --------------------------------------------------------------------------- #

def extract_code(completion: str) -> str:
    """Pull the first ```python ... ``` block, or fall back to the raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return completion.strip()


def run_sandboxed(code: str, test_snippet: str = "") -> tuple[bool, str]:
    """
    Execute `code` (+ optional `test_snippet`) in a subprocess with a timeout.
    Returns (passed, stderr_or_reason).
    """
    full_script = code + "\n\n" + test_snippet
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_script)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
        passed = result.returncode == 0
        reason = result.stderr.strip()[-500:] if not passed else ""
        return passed, reason
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    finally:
        Path(path).unlink(missing_ok=True)


def score_candidate(instruction: str, completion: str, test_snippet: str) -> tuple[bool, str]:
    code = extract_code(completion)
    if not code:
        return False, "empty completion"
    # Always sanity-check it at least imports/parses cleanly, then run tests if given.
    passed, reason = run_sandboxed(code, test_snippet)
    return passed, reason


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #

def build_pairs_from_seeds(model, tokenizer, seeds: list[dict], num_candidates: int) -> list[dict]:
    pairs = []
    for i, item in enumerate(seeds):
        instruction = item["instruction"]
        test_snippet = item.get("test", "")  # optional property-based test code

        print(f"[{i+1}/{len(seeds)}] Sampling {num_candidates} candidates for: {instruction[:60]}...")
        candidates = generate_candidates(model, tokenizer, instruction, num_candidates)

        scored = []
        for c in candidates:
            passed, reason = score_candidate(instruction, c, test_snippet)
            scored.append((c, passed, reason))

        passing = [c for c, p, _ in scored if p]
        failing = [c for c, p, _ in scored if not p]

        if passing and failing:
            pairs.append({
                "prompt": PROMPT_TEMPLATE.format(instruction=instruction),
                "chosen": passing[0],
                "rejected": failing[0],
            })
        else:
            print(f"  -> skipped (all-pass or all-fail, no contrast): "
                  f"{len(passing)} passing / {len(failing)} failing")

    return pairs


def load_repair_pairs(repair_file: Path) -> list[dict]:
    """
    repair_file format, one JSON object per line:
        {"instruction": "...", "buggy": "<code>", "fixed": "<code>"}
    """
    pairs = []
    if not repair_file.exists():
        print(f"No repair file at {repair_file}, skipping.")
        return pairs

    with open(repair_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pairs.append({
                "prompt": PROMPT_TEMPLATE.format(instruction=item["instruction"]),
                "chosen": item["fixed"],
                "rejected": item["buggy"],
            })
    return pairs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-file", type=Path, default=Path("data/seed_instructions.jsonl"),
                         help="JSONL file of {'instruction': ..., 'test': ...} seed prompts")
    parser.add_argument("--repair-file", type=Path, default=Path("data/repair_pairs.jsonl"),
                         help="JSONL file of {'instruction', 'buggy', 'fixed'} hand-curated repair pairs")
    parser.add_argument("--out", type=Path, default=Path("data/orpo_pairs.jsonl"))
    parser.add_argument("--adapter", type=str, default="shohuu/pyroton-primefix-v3",
                         help="HF repo id (or local path) of the LoRA adapter to sample from")
    parser.add_argument("--num-candidates", type=int, default=6,
                         help="Completions sampled per seed instruction")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Skip model sampling; only emit the repair pairs (fast dry run)")
    args = parser.parse_args()

    all_pairs = []

    if not args.skip_generation:
        if not args.seed_file.exists():
            print(f"WARNING: seed file {args.seed_file} not found. "
                  f"Create it with lines like:\n"
                  f'  {{"instruction": "Write a function to reverse a string", '
                  f'"test": "assert reverse(\'abc\') == \'cba\'"}}\n'
                  f"Skipping generated pairs.")
        else:
            model, tokenizer = load_model(args.adapter)
            with open(args.seed_file) as f:
                seeds = [json.loads(line) for line in f if line.strip()]
            all_pairs.extend(build_pairs_from_seeds(model, tokenizer, seeds, args.num_candidates))

    repair_pairs = load_repair_pairs(args.repair_file)
    print(f"Loaded {len(repair_pairs)} hand-curated repair pairs.")
    all_pairs.extend(repair_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(all_pairs)} ORPO pairs to {args.out}")


if __name__ == "__main__":
    main()
