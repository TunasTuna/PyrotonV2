"""Evaluate Kaggle adapters directly on the pinned stock Q4_K_M model."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys
from zipfile import ZipFile

from generate_holdout import digest
from pilot_io import write_json_atomic
from report_pilot_baseline import generations, load_run


STOCK_HASH = "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"
BASE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
BASE_CONFIG_HASH = "88f9a17863c05fb313515d2ff74b1098e0c35579f99068e32beda00618508ae0"
PAIRS_HASH = "16908db487a509bdf8c3c7401843dc0b97c95eae9f6820f3cbe8c80c3bb18cc9"
CONVERTER_REVISION = "6f311d7ad63dce97a3ad1deaafb7cc044bb8c26c"
SUITES = {"dev": (60, 37, "61a87707820ae8ee7faff5a6b37b93d6e614e62993080aec5d80c58b382be0dd"),
          "mbppplus": (162, 96, "96e59832ea1be6a12154ff57b5f63ed97996750c5f3c7bc81c0f3dfc207ddbad")}
MIN_MEDIAN_RATIO = 0.75


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def exact_mcnemar(wins, losses):
    total = wins + losses
    return min(1.0, 2 * sum(comb(total, count) for count in range(min(wins, losses) + 1)) / (2**total))


def boundary(value):
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return not value or isinstance(value, (list, tuple)) and any(boundary(item) for item in value)
    return False


def categorize(task, error):
    message = error or "No diagnostic returned."
    if re.search(r"\b(?:SyntaxError|IndentationError|NameError|KeyError|ImportError|ModuleNotFoundError)\b|deadline|missing \d+ required|positional argument", message):
        return "other", "Load, interface, or execution failure."
    if task["source"] == "original" and re.search(r"Expected (?:ValueError|TypeError)|expected exception", message, re.I):
        return "edge case", "A strict validation assertion was not satisfied."
    match = re.search(r"\bcase (\d+):", message)
    if match:
        index = int(match.group(1)) - 1
        if "base_input" in task:
            if len(task["base_input"]) <= index < len(task["base_input"]) + len(task["plus_input"]):
                return "edge case", "The first reported mismatch was on an MBPP+ extra fixture."
            if 0 <= index < len(task["base_input"]):
                return "logic", "The first reported mismatch was on an original MBPP fixture."
        elif "cases" in task and 0 <= index < len(task["cases"]):
            if any(boundary(arg) for arg in task["cases"][index]):
                return "edge case", "The first mismatched fixture includes an empty or null input."
            return "logic", "A result differed from the reference on the first reported fixture."
    if "AssertionError" in message:
        return "logic", "A behavioral assertion failed."
    return "other", "The scorer reported an error without identifying its input fixture."


def lengths(records):
    values = [record["tokens"] for record in records.values()]
    duration = sum(record["decode_seconds"] for record in records.values())
    return {"median_output_tokens": median(values), "mean_output_tokens": mean(values),
            "decode_tokens_per_second": sum(values) / duration}


def comparison(tasks, stock, stock_scores, candidate, candidate_scores):
    wins = sorted(key for key in tasks if candidate_scores[key]["passed"] and not stock_scores[key]["passed"])
    losses = sorted(key for key in tasks if stock_scores[key]["passed"] and not candidate_scores[key]["passed"])
    regressions = []
    for key in losses:
        error = candidate_scores[key]["error"]
        label, evidence = categorize(tasks[key], error)
        regressions.append({"id": key, "category": label, "evidence": evidence, "error": error})
    start, end = lengths(stock), lengths(candidate)
    sharp_drop = end["median_output_tokens"] < MIN_MEDIAN_RATIO * start["median_output_tokens"]
    return {"tasks": len(tasks), "stock_passed": sum(row["passed"] for row in stock_scores.values()),
            "passed": sum(row["passed"] for row in candidate_scores.values()), "wins": wins, "losses": losses,
            "exact_two_sided_mcnemar_p": exact_mcnemar(len(wins), len(losses)),
            "stock": start, "candidate": end, "sharp_median_drop": sharp_drop,
            "sharp_drop_without_gain": sharp_drop and len(wins) <= len(losses),
            "regression_counts": dict(Counter(row["category"] for row in regressions)), "regressions": regressions}


def archive_to_gguf(archive_path, epoch, converter, base_config, destination):
    source_hash = digest(archive_path)
    receipt = destination.with_suffix(".json")
    if destination.exists() and receipt.is_file():
        if json.loads(receipt.read_bytes()) == {"archive_sha256": source_hash, "gguf_sha256": digest(destination),
                                               "base_config_sha256": digest(base_config / "config.json")}:
            return
        raise ValueError(f"Conversion provenance mismatch for epoch {epoch}. Use a fresh output directory.")
    adapter = destination.parent / f"adapter-epoch-{epoch}"
    adapter.mkdir(exist_ok=True)
    with ZipFile(archive_path) as archive:
        expected = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                    "pyroton-complete.json", "pilot-run.json"}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected or archive.testzip():
            raise ValueError(f"Invalid epoch {epoch} adapter archive.")
        run_info = json.loads(archive.read("pilot-run.json"))
        marker = json.loads(archive.read("pyroton-complete.json"))
        config = json.loads(archive.read("adapter_config.json"))
        if (marker["epoch"] != epoch or run_info["base_model"] != BASE_ID or run_info["base_revision"] != BASE_REVISION
                or run_info["runtime"] != "kaggle" or run_info["pairs_sha256"] != PAIRS_HASH
                or config["r"] != 16 or config["lora_alpha"] != 32
                or config["base_model_name_or_path"] != BASE_ID):
            raise ValueError("Adapter lineage or LoRA settings do not match the Kaggle pilot.")
        archive.extractall(adapter, members=("adapter_config.json", "adapter_model.safetensors"))
    temporary = destination.with_name(destination.stem + ".partial.gguf")
    run(converter, "--base", base_config, "--outtype", "f16", "--outfile", temporary, adapter)
    with temporary.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("The converter did not produce GGUF.")
    temporary.replace(destination)
    write_json_atomic(receipt, {"archive_sha256": source_hash, "gguf_sha256": digest(destination),
                                "base_config_sha256": digest(base_config / "config.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "server", "converter", "base-config", "dev", "mbppplus", "stock-dir", "adapters", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.model) != STOCK_HASH:
        raise ValueError("Use the exact stock Q4_K_M file from the frozen baseline.")
    if digest(args.base_config / "config.json") != BASE_CONFIG_HASH:
        raise ValueError("Use the configuration of the pinned stock Hugging Face revision.")
    commit = subprocess.check_output(["git", "-C", str(args.converter.parent), "rev-parse", "HEAD"], text=True).strip()
    if commit != CONVERTER_REVISION:
        raise ValueError("Use the pinned b12521 llama.cpp converter.")
    suites = {name: getattr(args, name) for name in SUITES}
    baseline = {}
    for name, dataset in suites.items():
        count, passed, fingerprint = SUITES[name]
        if digest(dataset) != fingerprint:
            raise ValueError(f"Unexpected {name} dataset. Only the frozen allowed suites may be used.")
        prefix = f"stock-{name}-1"
        raw, scores, config = load_run(dataset, args.stock_dir / f"{prefix}.jsonl",
                                       args.stock_dir / f"{prefix}-scores.json", 512)
        if (len(raw) != count or sum(row["passed"] for row in scores.values()) != passed
                or config["server_sha256"] != digest(args.server) or config["base_sha256"] != STOCK_HASH
                or config["slots"] != 1 or config["threads"] != 4 or config["temperature"] != 0
                or config.get("gpu_layers", 0) != 0):
            raise ValueError(f"Incorrect frozen stock comparison for {name}.")
        baseline[name] = raw, scores, config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": 42, "threads": 4, "temperature": 0, "new_tokens": 512,
              "sharp_drop_ratio": MIN_MEDIAN_RATIO, "checkpoints": {}}
    scripts = Path(__file__).resolve().parent
    for epoch in (1, 2):
        adapter = args.output_dir / f"epoch-{epoch}-adapter.gguf"
        archive_to_gguf(args.adapters / f"pyroton-epoch-{epoch}-adapter.zip", epoch, args.converter,
                        args.base_config, adapter)
        outcomes = {}
        for name, dataset in suites.items():
            raw_file = args.output_dir / f"epoch-{epoch}-{name}.jsonl"
            score_file = args.output_dir / f"epoch-{epoch}-{name}-scores.json"
            run(scripts / "generate_holdout.py", "--model", args.model, "--server", args.server,
                "--dataset", dataset, "--output", raw_file, "--adapter", adapter, "--tokens", 512,
                "--threads", 4, "--slots", 1, "--gpu-layers", 0)
            run(scripts / "score_holdout.py", "--dataset", dataset, "--generations", raw_file, "--output", score_file)
            raw = generations(raw_file)
            stock, stock_scores, config = baseline[name]
            adapter_config = {**config, "adapter_sha256": digest(adapter)}
            if raw.keys() != stock.keys() or any(row["configuration"] != adapter_config for row in raw.values()):
                raise ValueError("Adapter evaluation settings differ from stock.")
            tasks = {row["id"]: row for row in json.loads(dataset.read_bytes())}
            scores = {row["id"]: row for row in json.loads(score_file.read_bytes())}
            outcomes[name] = comparison(tasks, stock, stock_scores, raw, scores)
        report["checkpoints"][f"epoch-{epoch}"] = {"suites": outcomes,
            "beats_stock_on_both_without_sharp_drop": all(row["passed"] > row["stock_passed"]
                                                           and not row["sharp_median_drop"] for row in outcomes.values())}
        write_json_atomic(args.output_dir / "step3-cpu-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
