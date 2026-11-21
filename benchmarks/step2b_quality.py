"""Audit frozen pilot edges and rebuild reproducible ORPO pairs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from statistics import median

from pilot_edges import decode, encode
from pilot_verifier import PilotSandbox, checks, verify_candidate
from report_pilot import (pick_examples, read_json, read_jsonl, run_provenance, unique_index,
                          validate_judgments, write_json, write_jsonl)
from sample_pilot import identity
from step2b_edge_policy import FIXED_ORACLE_TASKS, KNOWN_DROPS, REVIEWED


def audit_task(task):
    number = int(task["id"].split("/")[-1])
    rule = REVIEWED.get(number)
    original_edges = task["generated_edges"]
    reviewed, reasons = [], []
    for edge in original_edges:
        args, kwargs = decode(edge["args"]), decode(edge["kwargs"])
        if rule is None:
            reason = KNOWN_DROPS.get(number, "No reviewed contract covers this input.")
            reasons.append(("dropped", reason))
            continue
        reason, eligible, oracle = rule
        try:
            if not eligible(*args, **kwargs):
                reasons.append(("dropped", f"Outside the reviewed domain. {reason}"))
                continue
            expected = {"outcome": ["return", encode(oracle(*args, **kwargs))]}
        except (TypeError, ValueError, OverflowError) as error:
            reasons.append(("dropped", f"No supported outcome for this input. {type(error).__name__}."))
            continue
        if task["compare_mutations"]:
            reasons.append(("dropped", "The task requires a reviewed mutation oracle."))
            continue
        state = "kept" if edge["expected"] == expected else "corrected"
        if state == "corrected" and number not in FIXED_ORACLE_TASKS:
            reasons.append(("dropped", "Reference and independent contract check disagree."))
            continue
        reviewed.append({**edge, "expected": expected})
        reasons.append((state, reason))
    suite = {
        "entry_point": task["entry_point"], "original_checks": checks(task),
        "test_setup": task.get("test_setup", ""), "reference_signature": task["reference_signature"],
        "generated_edges": reviewed, "compare_mutations": task["compare_mutations"],
    }
    digest = hashlib.sha256(json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    counts = {**task["verifier_counts"], "edges": dict(Counter(edge["kind"] for edge in reviewed))}
    return {**task, "generated_edges": reviewed, "test_suite_sha256": digest, "verifier_counts": counts}, reasons


def first_stage(row):
    return {"signature": "signature", "original": "original_tests", "edge": "extra_edge_only",
            "compile": "before_signature", "runtime": "sandbox_phase_unresolved", "complete": "passed"}[row["stage"]]


def expect_text(expected):
    kind, value = expected["outcome"]
    result = f"raises {value}" if kind == "raise" else repr(decode(value))
    if "after_args" in expected:
        result += f" (args after call {decode(expected['after_args'])!r}, kwargs {decode(expected['after_kwargs'])!r})"
    return result


def review_examples(tasks, corrected_tasks, rows, decisions, corrected_by_key, location):
    candidates = sorted((row for row in rows if first_stage(row) == "extra_edge_only"), key=identity)
    sampled = random.Random(20261121).sample(candidates, min(30, len(candidates)))
    lines = ["# Thirty randomly selected edge-only failures", "",
             "The sample uses seed `20261121` and draws from the original pilot's edge-only failures.", ""]
    for number, row in enumerate(sampled, 1):
        task = tasks[row["problem_id"]]
        index = int(row["first_failed_check"].rsplit(":", 1)[1]) - 1
        edge = task["generated_edges"][index]
        args, kwargs = decode(edge["args"]), decode(edge["kwargs"])
        state, reason = decisions[task["id"]][index]
        corrected = corrected_by_key[identity(row)]
        fixed = ""
        if state == "corrected":
            replacement = next(item for item in corrected_tasks[task["id"]]["generated_edges"]
                               if (item["args"], item["kwargs"]) == (edge["args"], edge["kwargs"]))
            fixed = f" Updated expected outcome `{expect_text(replacement['expected'])}`."
        lines.extend([f"## {number}. {task['id']} · {row['style']} · draw {row['sample_index']}", "",
                      task["description"], "", f"- Arguments `{args!r}` and keyword arguments `{kwargs!r}`.",
                      f"- Original reference expected `{expect_text(edge['expected'])}`.",
                      f"- Review **{state}**. {reason}{fixed}", f"- Corrected verdict `{first_stage(corrected)}`.",
                      "", "````text", row["error"], "````", "", "````python", row["code"], "````", ""])
    location.write_text("\n".join(lines), encoding="utf-8")


def length_summary(rows):
    return {"pairs": len(rows), "median_chosen": median(row["chosen_tokens"] for row in rows) if rows else None,
            "median_rejected": median(row["rejected_tokens"] for row in rows) if rows else None}


def summarize_styles(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["style"]].append(row)
    return {style: {"draws": len(draws), "passed": sum(row["passed"] for row in draws),
                    "pass_rate": sum(row["passed"] for row in draws) / len(draws)}
            for style, draws in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verified-corpus", "judgments", "pilot-audit", "pilot-orpo", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tasks = unique_index(read_json(args.verified_corpus), lambda row: row["id"], "task")
    old = validate_judgments(read_jsonl(args.judgments), tasks, complete=True)
    corpus_hash = hashlib.sha256(args.verified_corpus.read_bytes()).hexdigest()
    provenance = run_provenance(old, corpus_hash, complete=True)
    deadline = provenance["verifier"]["deadline"]
    audit = read_json(args.pilot_audit)
    blocked = {identity(row): row["holdout_code_collision"] for row in audit["sample_code_exclusions"]}
    if (set(tasks) - set(audit["accepted_ids"]) or set(blocked) - {identity(row) for row in old}):
        raise ValueError("The existing final holdout audit does not cover this pilot.")
    corrected, decisions = {}, {}
    for task in tasks.values():
        corrected[task["id"]], decisions[task["id"]] = audit_task(task)
    changed_oracles = {key for key, values in decisions.items() if any(state == "corrected" for state, _ in values)}
    new = []
    worker = PilotSandbox()
    try:
        for row in old:
            task = corrected[row["problem_id"]]
            needs_regrade = row["stage"] in ("edge", "runtime") or (row["passed"] and task["id"] in changed_oracles)
            verdict = verify_candidate(task, row["code"], worker, deadline) if needs_regrade else {}
            new.append({**row, **verdict, "original_test_suite_sha256": row["test_suite_sha256"],
                        "test_suite_sha256": task["test_suite_sha256"], "original_stage": row["stage"],
                        "holdout_code_collision": blocked.get(identity(row), [])})
    finally:
        worker.close()
    new_by_key = {identity(row): row for row in new}
    review_examples(tasks, corrected, old, decisions, new_by_key, output / "edge-only-review-30.md")
    groups = defaultdict(list)
    for row in new:
        groups[row["problem_id"]].append(row)
    original_passing_ids = {row["problem_id"] for row in old if row["passed"]}
    zero_ids = sorted(set(tasks) - original_passing_ids)
    corrected_zero_ids = [key for key in zero_ids if not any(row["passed"] for row in groups[key])]
    unbounded, matched, sft = [], [], []
    for key, group in sorted(groups.items()):
        chosen, pair, _ = pick_examples(corrected[key], group)
        _, close_pair, _ = pick_examples(corrected[key], group, max_pair_ratio=1.25)
        if chosen:
            sft.append(chosen)
        if pair:
            unbounded.append(pair)
        if close_pair:
            matched.append(close_pair)
    original_pairs = read_jsonl(args.pilot_orpo)
    report = {
        "draws": len(old), "problems": len(tasks),
        "before": dict(Counter(map(first_stage, old))), "after": dict(Counter(map(first_stage, new))),
        "prompt_style_before": summarize_styles(old), "prompt_style_after": summarize_styles(new),
        "edge_decisions": dict(Counter(state for group in decisions.values() for state, _ in group)),
        "problems_with_retained_edges": sum(bool(task["generated_edges"]) for task in corrected.values()),
        "original_pairs": length_summary(original_pairs), "unrestricted_corrected_pairs": length_summary(unbounded),
        "original_pairs_within_ratio": sum(max(row["chosen_tokens"], row["rejected_tokens"])
                                          <= 1.25 * min(row["chosen_tokens"], row["rejected_tokens"])
                                          for row in original_pairs),
        "length_matched_pairs": length_summary(matched), "corrected_sft_examples": len(sft),
        "max_long_short_ratio": 1.25,
        "original_zero_pass_count": len(zero_ids), "still_zero_after_regrade": len(corrected_zero_ids),
        "pilot_judgments_sha256": hashlib.sha256(args.judgments.read_bytes()).hexdigest(),
        "pilot_overlap_audit_sha256": hashlib.sha256(args.pilot_audit.read_bytes()).hexdigest(),
    }
    write_json(output / "edge-decisions.json", decisions)
    write_json(output / "corrected-corpus.json", list(corrected.values()))
    write_json(output / "zero-pass-cohort.json", {"original": zero_ids, "still_zero_after_regrade": corrected_zero_ids})
    write_jsonl(output / "sample-stages.jsonl", [
        {"problem_id": row["problem_id"], "style": row["style"], "sample_index": row["sample_index"],
         "original_category": first_stage(row), "corrected_category": first_stage(new_by_key[identity(row)]),
         "original_failed_check": row["first_failed_check"],
         "corrected_failed_check": new_by_key[identity(row)]["first_failed_check"]}
        for row in old
    ])
    write_jsonl(output / "orpo-length-matched.jsonl", matched)
    report["length_matched_pairs_sha256"] = hashlib.sha256((output / "orpo-length-matched.jsonl").read_bytes()).hexdigest()
    write_json(output / "quality-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
