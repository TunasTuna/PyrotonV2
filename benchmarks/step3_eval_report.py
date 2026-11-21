"""Compare stock repeats and merged pilot checkpoints on dev and MBPP+ only."""

import argparse
import json
from pathlib import Path

from pilot_io import write_json_atomic
from report_pilot_baseline import load_run, stats


def load(run_dir, dataset, suite, label):
    filename = f"stock-{suite}-{label.removeprefix('stock-')}" if label.startswith("stock-") else f"{label}-{suite}"
    prefix = run_dir / filename
    generations, scores, config = load_run(dataset, prefix.with_suffix(".jsonl"),
                                            prefix.with_name(prefix.name + "-scores.json"), 512)
    if config["temperature"] != 0 or config["slots"] != 1:
        raise ValueError("Use greedy decoding and one request slot for this comparison.")
    outcomes = {key: row["passed"] for key, row in scores.items()}
    summary = stats(set(generations), outcomes, {key: row["tokens"] for key, row in generations.items()}, 512)
    summary["pass_rate"] = summary["passed"] / summary["tasks"]
    seconds = sum(row["decode_seconds"] for row in generations.values())
    summary["decode_tokens_per_second"] = sum(row["tokens"] for row in generations.values()) / seconds
    return generations, outcomes, config, summary


def paired(first, second):
    a, a_pass, _, a_summary = first
    b, b_pass, _, b_summary = second
    wins = sorted(key for key in a if not a_pass[key] and b_pass[key])
    losses = sorted(key for key in a if a_pass[key] and not b_pass[key])
    return {"wins": wins, "losses": losses, "outcome_disagreements": len(wins) + len(losses),
            "different_answers": sum(a[key]["answer"] != b[key]["answer"] for key in a),
            "net_passes": len(wins) - len(losses),
            "delta_percentage_points": 100 * (b_summary["pass_rate"] - a_summary["pass_rate"])}


def compare_suite(run_dir, dataset, suite, checkpoints):
    first = load(run_dir, dataset, suite, "stock-1")
    second = load(run_dir, dataset, suite, "stock-2")
    if first[2] != second[2]:
        raise ValueError("Stock repeats require identical settings and model files.")
    noise = paired(first, second)
    noise["absolute_net_passes"] = abs(noise["net_passes"])
    output = {"stock_configuration": first[2], "stock_1": first[3], "stock_2": second[3], "noise": noise}
    for label in checkpoints:
        candidate = load(run_dir, dataset, suite, label)
        base = first[2]["base_sha256"]
        if {**candidate[2], "base_sha256": base} != first[2] or candidate[2]["base_sha256"] == base:
            raise ValueError("A checkpoint must use the same evaluation settings and a distinct merged model.")
        comparisons = {"vs_stock_1": paired(first, candidate), "vs_stock_2": paired(second, candidate)}
        for comparison in comparisons.values():
            comparison["absolute_net_change_exceeds_observed_noise"] = (
                abs(comparison["net_passes"]) > noise["absolute_net_passes"]
            )
        output[label] = {"configuration": candidate[2], "metrics": candidate[3], **comparisons}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "mbppplus", "run-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="*", choices=("epoch-1", "epoch-2"), default=[])
    args = parser.parse_args()
    for path, expected in ((args.dev, 60), (args.mbppplus, 162)):
        if len(json.loads(path.read_bytes())) != expected:
            raise ValueError(f"Unexpected holdout size for {path}.")
    result = {"decoding_seed": 42, "throughput_metric": "total generated tokens / total decode seconds",
              "suites": {suite: compare_suite(args.run_dir, path, suite, args.checkpoints)
                         for suite, path in (("dev", args.dev), ("mbppplus", args.mbppplus))}}
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
