"""Report full-suite and dev-slice stock baselines with matching task IDs."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_bytes())


def generations(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    values = {row["id"]: row for row in rows}
    if len(values) != len(rows):
        raise ValueError(f"Duplicate generations: {path}")
    return values


def load_run(dataset, generation_log, scores_file, token_cap):
    ids = {row["id"] for row in read(dataset)}
    generation = generations(generation_log)
    scores = {row["id"]: row for row in read(scores_file)}
    if ids != set(generation) or ids != set(scores):
        raise ValueError("IDs in dataset, generations and scores must agree.")
    configs = {json.dumps(item["configuration"], sort_keys=True) for item in generation.values()}
    if len(configs) != 1:
        raise ValueError("Mixed generation settings.")
    config = json.loads(configs.pop())
    if config["new_tokens"] != token_cap or config["adapter_sha256"] is not None:
        raise ValueError("Incorrect cap or adapter for stock run.")
    if config["dataset_sha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Incorrect dataset snapshot.")
    return generation, scores, config


def stats(ids, outcomes, raw_tokens, cap):
    values = [raw_tokens[key] for key in ids]
    return {
        "passed": sum(outcomes[key] for key in ids), "tasks": len(ids),
        "median_tokens": median(values), "max_tokens": max(values),
        "at_cap": sum(count >= cap for count in values),
    }


def matched_control(dataset, generations_file, scores_file, comparison_config):
    generation, scores, config = load_run(dataset, generations_file, scores_file, 256)
    if {**config, "new_tokens": 512} != comparison_config:
        raise ValueError("Matched runs must differ only in the output cap.")
    return generation, scores, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "dev", "mbppplus", "old-baseline", "primary-generations",
                 "primary-scores", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mbppplus-generations", type=Path)
    parser.add_argument("--mbppplus-scores", type=Path)
    for name in ("matched256-primary-generations", "matched256-primary-scores",
                 "matched256-mbppplus-generations", "matched256-mbppplus-scores"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    primary, primary_scores, primary_config = load_run(
        args.primary, args.primary_generations, args.primary_scores, 512,
    )
    if bool(args.mbppplus_generations) != bool(args.mbppplus_scores):
        parser.error("Supply both MBPP+ files when reporting that suite.")
    matched_primary = (args.matched256_primary_generations, args.matched256_primary_scores)
    matched_mbpp = (args.matched256_mbppplus_generations, args.matched256_mbppplus_scores)
    for suite, pair in (("primary", matched_primary), ("MBPP+", matched_mbpp)):
        if any(pair) and not all(pair):
            parser.error(f"Supply both 256-token files for the {suite} suite.")
    if all(matched_mbpp) and not args.mbppplus_generations:
        parser.error("Supply the 512-token MBPP+ files alongside its 256-token control.")
    old_report = read(args.old_baseline)
    primary_tasks = {row["id"]: row for row in read(args.primary)}
    final_dev = read(args.dev)
    old = {row["id"]: row["stock"] for row in old_report["results"]}
    dev = {row["id"] for row in final_dev}
    if set(primary) != set(old) or not dev <= primary.keys() or len(dev) != 60:
        raise ValueError("The old baseline or dev IDs differ from the frozen primary suite.")
    if (any(row["prompt"] != primary_tasks[row["id"]]["prompt"] for row in old_report["results"])
            or any(row != primary_tasks[row["id"]] for row in final_dev)):
        raise ValueError("The old baseline or dev prompts differ from the frozen primary suite.")
    if old_report["stock_configuration"]["base_sha256"] != primary_config["base_sha256"]:
        raise ValueError("Stock model differs from the 256-token run.")
    old_scores = {key: row["passed"] for key, row in old.items()}
    old_tokens = {key: row["tokens"] for key, row in old.items()}
    methodology = ["Dev scores use the corresponding full primary generations."]
    output = {
        "complete": bool(args.mbppplus_generations),
        "primary512_configuration": primary_config,
        "primary256_configuration": old_report["stock_configuration"], "suites": {},
    }
    for name, ids in (("dev", dev), ("full160", set(primary))):
        output["suites"][name] = {
            "256": stats(ids, old_scores, old_tokens, 256),
            "512": stats(ids, {key: row["passed"] for key, row in primary_scores.items()},
                         {key: row["tokens"] for key, row in primary.items()}, 512),
        }
    if args.mbppplus_generations:
        mbpp, mbpp_scores, mbpp_config = load_run(
            args.mbppplus, args.mbppplus_generations, args.mbppplus_scores, 512,
        )
        if primary_config["base_sha256"] != mbpp_config["base_sha256"]:
            raise ValueError("The model differs across suites.")
        output["mbppplus512_configuration"] = mbpp_config
        output["suites"]["mbppplus"] = {
            "512": stats(set(mbpp), {key: row["passed"] for key, row in mbpp_scores.items()},
                         {key: row["tokens"] for key, row in mbpp.items()}, 512),
        }
    if all(matched_primary):
        match_primary, match_scores, match_config = matched_control(args.primary, *matched_primary, primary_config)
        output["matched256_primary_configuration"] = match_config
        methodology.append("The primary cap comparison uses matched settings and preserves the historical 256-token result.")
        for name, ids in (("dev", dev), ("full160", set(primary))):
            suite = output["suites"][name]
            suite["historical256"] = suite["256"]
            suite["256"] = stats(ids, {key: row["passed"] for key, row in match_scores.items()},
                                 {key: row["tokens"] for key, row in match_primary.items()}, 256)
    else:
        methodology.append("The primary 256-token scores come from the historical Step 1 run.")
    if all(matched_mbpp):
        match_mbpp, match_mbpp_scores, match_mbpp_config = matched_control(args.mbppplus, *matched_mbpp, mbpp_config)
        output["matched256_mbppplus_configuration"] = match_mbpp_config
        output["suites"]["mbppplus"]["256"] = stats(
            set(mbpp), {key: row["passed"] for key, row in match_mbpp_scores.items()},
            {key: row["tokens"] for key, row in match_mbpp.items()}, 256,
        )
        outcome_changes = {"passed_at_512_only": [], "passed_at_256_only": []}
        for key in sorted(mbpp):
            passed_at_256 = match_mbpp_scores[key]["passed"]
            passed_at_512 = mbpp_scores[key]["passed"]
            if passed_at_256 != passed_at_512:
                bucket = "passed_at_512_only" if passed_at_512 else "passed_at_256_only"
                outcome_changes[bucket].append(key)
        output["mbppplus_outcome_changes"] = outcome_changes
        methodology.append("The MBPP+ runs differ only in the token cap.")
    output = {"methodology": " ".join(methodology), **output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["suites"], indent=2))


if __name__ == "__main__":
    main()
