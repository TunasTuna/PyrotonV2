"""Prepare and report the fixed 167-problem stock sampling probe."""

import argparse
from collections import Counter
import json
from pathlib import Path

from generate_holdout import digest
from judge_pilot import load_records
from pilot_io import write_json_atomic
from sample_pilot import MAX_TOKENS, TEMPERATURE, ZERO_PASS_PER_STYLE, identity, jobs_for, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--corrected-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--stock-model-sha256")
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_bytes())
    original = set(cohort["original"])
    still_zero = set(cohort["still_zero_after_regrade"])
    if len(original) != 167 or still_zero - original:
        raise ValueError("The probe must use the sealed original 167-problem zero-pass cohort.")
    tasks = [row for row in json.loads(args.corrected_corpus.read_bytes()) if row["id"] in original]
    if len(tasks) != 167 or {row["id"] for row in tasks} != original:
        raise ValueError("The audited corpus does not cover the full zero-pass cohort.")
    if args.judgments is None:
        write_json_atomic(args.output, tasks)
        return
    rows = list(load_records(args.judgments).values())
    jobs = {identity(row): row for row in jobs_for(tasks, ZERO_PASS_PER_STYLE)}
    if len(rows) != len(jobs) or set(jobs) != {identity(row) for row in rows}:
        raise ValueError("The complete report requires exactly 32 judgments per original problem.")
    configs = {json.dumps(row["configuration"], sort_keys=True) for row in rows}
    expected = next(iter(rows))["configuration"]
    if (not args.stock_model_sha256 or len(configs) != 1 or expected["gpu_layers"] < 1
            or expected["model_sha256"] != args.stock_model_sha256
            or expected["verified_corpus_sha256"] != digest(args.corrected_corpus)
            or expected["protocol"] != protocol(ZERO_PASS_PER_STYLE)
            or expected["temperature"] != TEMPERATURE or expected["max_tokens"] != MAX_TOKENS
            or expected["limit_problems"] is not None):
        raise ValueError("The sampling probe must use the pinned stock checkpoint on GPU.")
    passes = Counter({key: 0 for key in original})
    for row in rows:
        if any(row[field] != value for field, value in jobs[identity(row)].items()):
            raise ValueError("A judgment does not match its planned draw.")
        passes[row["problem_id"]] += int(row["passed"])
    solvable = {key for key, count in passes.items() if count}
    report = {"original_zero_pass_problems": len(original), "new_draws": len(rows), "draws_per_problem": 32,
              "already_recovered_by_regrade": len(original - still_zero), "solvable_with_32_draws": len(solvable),
              "solvable_among_still_zero_after_regrade": len(solvable & still_zero),
              "still_zero_with_32_draws": len(original - solvable),
              "passes_by_problem": dict(sorted(passes.items())), "configuration": expected}
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
