"""Report the MBPP rejection-sampling pilot and write audited training records."""

import argparse
import ast
import difflib
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from build_holdout import code_key
from build_pilot_corpus import load_mbppplus, load_old, load_primary


STYLES = ("mbpp", "humaneval")
SAMPLES_PER_STYLE = 4
AUDIT_INPUTS = {
    "mbpp", "sanitized", "primary", "splits_dir", "old_report",
    "mbppplus_release", "mbppplus_audit",
}
OPTIONAL_AUDIT_INPUTS = {"old_comparison"}
BEHAVIORAL_FAILURES = {
    "logic", "logic_error", "edge", "edge_case", "edge_case_or_validation",
    "edge_validation", "validation", "missing_import", "wrong_result", "original_test", "generated_edge",
}
BOILERPLATE = re.compile(
    r"^(?:write|implement|create|define)\s+(?:a\s+)?(?:python\s+)?function\s+(?:to|that|which)\s+",
    re.IGNORECASE,
)
TRIPLE = re.compile(r"(?ms)^\s*(?:[rRuU])?('''|\"\"\")(.*?)(?:\1|\Z)")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)


def unique_index(rows, key, label):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f"Duplicate {label}: {identity}")
        result[identity] = row
    return result


def words(text):
    return re.findall(r"\w+", BOILERPLATE.sub("", text.strip()).casefold())


def matching_text(candidate, target, strict):
    sample = words(candidate)
    if not sample or not target:
        return False
    matcher = difflib.SequenceMatcher(None, target, sample, autojunk=False)
    common_run = matcher.find_longest_match().size
    if len(target) < 5:
        return len(target) >= 3 and common_run == len(target)
    if strict:
        return common_run >= max(5, math.ceil(0.75 * len(target)))
    return common_run >= max(4, math.ceil(0.45 * len(target))) or (
        len(sample) >= 5 and matcher.ratio() >= 0.70
    )


def copied_description(code, raw, description):
    """Return (literal docstring copy, loose docstring copy, broader description reuse)."""
    target = words(description)
    docstrings = []
    try:
        tree = ast.parse(code)
    except (ValueError, SyntaxError):
        # A response cut off inside a triple-quoted docstring still copied it.
        # Restrict this fallback to a bare string at the start of a source line.
        docstrings.extend(match.group(2) for match in TRIPLE.finditer(raw))
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node, clean=False)
                if docstring:
                    docstrings.append(docstring)

    strict = any(matching_text(docstring, target, True) for docstring in docstrings)
    if strict:
        return True, True, True
    loose_docstring = any(matching_text(docstring, target, False) for docstring in docstrings)
    comment_lines = [line.lstrip()[1:] for line in raw.splitlines() if line.lstrip().startswith("#")]
    prose = raw.partition("```")[0] if "```" in raw else ""
    reuse = loose_docstring or any(matching_text(text, target, False) for text in (" ".join(comment_lines), prose))
    return False, loose_docstring, reuse


def user_prompt(prompt):
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Prompt must be a string or chat messages.")
    if any(not isinstance(item, dict) or not isinstance(item.get("content"), str) for item in prompt):
        raise ValueError("Invalid chat messages.")
    roles = [message.get("role") for message in prompt]
    if roles not in (["user"], ["system", "user"]):
        raise ValueError("Expected a user prompt with an optional system message.")
    return prompt[-1]["content"]


def validate_judgments(judgments, tasks, complete):
    if any(type(row["sample_index"]) is not int for row in judgments):
        raise ValueError("Sample indices must be integers.")
    index = unique_index(judgments, lambda row: (
        row["problem_id"], row["style"], row["sample_index"],
    ), "sample")
    expected = {(problem, style, position) for problem in tasks for style in STYLES
                for position in range(SAMPLES_PER_STYLE)}
    if not set(index) <= expected:
        raise ValueError(f"Unexpected sample keys: {sorted(set(index) - expected)[:8]}")
    if complete and set(index) != expected:
        raise ValueError(f"Pilot is incomplete: {len(index)}/{len(expected)} judgments.")

    annotated = []
    for row in judgments:
        task = tasks[row["problem_id"]]
        style = row["style"]
        if user_prompt(row["prompt"]) != task["prompts"][style]:
            raise ValueError(f"Prompt drift in {task['id']} / {style}.")
        if row["test_suite_sha256"] != task["test_suite_sha256"]:
            raise ValueError(f"Test-suite hash drift for {task['id']}.")
        if type(row["tokens"]) is not int or not 0 <= row["tokens"] <= 512:
            raise ValueError(f"Invalid generation token count for {task['id']}.")
        if type(row["passed"]) is not bool or type(row["signature_ok"]) is not bool:
            raise ValueError(f"Missing boolean verdict fields for {task['id']}.")
        if not isinstance(row["code"], str) or not isinstance(row["answer"], str):
            raise ValueError(f"Expected source and raw answer strings for {task['id']}.")
        if row["passed"] and (not row["signature_ok"] or not row["code"].strip()):
            raise ValueError(f"Passing solution has no valid signature or source for {task['id']}.")
        solution_tokens = row.get("solution_tokens")
        if solution_tokens is not None and (type(solution_tokens) is not int or solution_tokens < 0):
            raise ValueError(f"Invalid solution token count for {task['id']}.")
        if complete and row["code"].strip() and not solution_tokens:
            raise ValueError(f"Normalized solution token count is missing for {task['id']}.")
        strict, loose, reuse = copied_description(row["code"], row["answer"], task["description"])
        annotated.append({**row, "copied_docstring": strict, "copied_docstring_loose": loose,
                          "description_reuse": reuse})
    return annotated


def run_provenance(judgments, verified_hash, complete):
    if not judgments:
        return {}
    generation = judgments[0]["configuration"]
    verifier = judgments[0]["verifier_configuration"]
    if any(row["configuration"] != generation or row["verifier_configuration"] != verifier
           for row in judgments):
        raise ValueError("Judgments combine different sampling or verifier runs.")
    if (generation["verified_corpus_sha256"] != verified_hash
            or verifier["corpus_sha256"] != verified_hash
            or generation["max_tokens"] != 512 or generation["temperature"] != 0.8
            or generation["per_style"] != SAMPLES_PER_STYLE):
        raise ValueError("Judgments were generated outside the audited pilot configuration.")
    if complete and generation["limit_problems"] is not None:
        raise ValueError("A limited smoke run is ineligible for final training files.")
    return {"sampling": generation, "verifier": verifier}


def row_length(row):
    tokens = row.get("solution_tokens")
    return tokens if tokens is not None else len(row["code"].encode("utf-8"))


def pass_rank(row):
    return row["copied_docstring_loose"], row_length(row), row["style"], row["sample_index"]


def behavioral_failure(row):
    category = str(row.get("failure_category") or "").casefold()
    key = re.sub(r"[ -]+", "_", category)
    return not row["passed"] and row["signature_ok"] and key in BEHAVIORAL_FAILURES


def pick_examples(task, rows):
    by_code = defaultdict(set)
    for row in rows:
        by_code[row["code"]].add(row["passed"])
    conflicting = {code for code, outcomes in by_code.items() if len(outcomes) > 1}
    eligible_rows = [row for row in rows if not row.get("holdout_code_collision") and row["code"] not in conflicting]
    positive = sorted((row for row in eligible_rows if row["passed"]), key=pass_rank)
    best = positive[0] if positive else None
    mixed = any(row["passed"] for row in rows) and any(not row["passed"] for row in rows)

    metadata = {
        "problem_id": task["id"], "source": task["source"], "source_split": task["split"],
        "test_suite_sha256": task["test_suite_sha256"],
    }
    sft = None
    if best:
        sft = {**metadata, "prompt": best["prompt"], "completion": best["code"],
               "completion_tokens": best.get("solution_tokens"), "prompt_style": best["style"]}

    pairs = []
    for chosen in positive if mixed else []:
        eligible = [row for row in eligible_rows if behavioral_failure(row)
                    and row["code"] != chosen["code"]
                    and row["prompt"] == chosen["prompt"] and row["style"] == chosen["style"]]
        if eligible:
            rejected = min(eligible, key=lambda row: (
                row["copied_docstring_loose"] != chosen["copied_docstring_loose"],
                abs(row_length(row) - row_length(chosen)), row_length(row), row["sample_index"],
            ))
            pairs.append((chosen, rejected))
    pair = None
    if pairs:
        chosen, rejected = min(pairs, key=lambda pair: pass_rank(pair[0]))
        pair = {**metadata, "prompt": chosen["prompt"], "chosen": chosen["code"],
                "rejected": rejected["code"], "chosen_tokens": chosen.get("solution_tokens"),
                "rejected_tokens": rejected.get("solution_tokens"),
                "prompt_style": chosen["style"],
                "failure_category": rejected["failure_category"],
                "first_failed_check": rejected.get("first_failed_check")}
    return sft, pair, len(conflicting)


def length_stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered), "min": ordered[0], "median": statistics.median(ordered),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1], "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 3),
    }


def sample_summary(rows):
    passed = sum(row["passed"] for row in rows)
    return {
        "samples": len(rows), "passed": passed,
        "pass_rate": round(passed / len(rows), 6) if rows else None,
        "copied_docstrings": sum(row["copied_docstring"] for row in rows),
        "copied_docstrings_loose": sum(row["copied_docstring_loose"] for row in rows),
        "description_reuse_any": sum(row["description_reuse"] for row in rows),
        "stop_reasons": dict(Counter(row.get("stop_reason") or "unknown" for row in rows)),
        "generated_tokens": length_stats([row["tokens"] for row in rows]),
        "normalized_code_tokens": length_stats([row["solution_tokens"] for row in rows
                                                if row.get("solution_tokens") is not None]),
    }


def summarize(tasks, judgments, audited_count):
    by_problem = defaultdict(list)
    for row in judgments:
        by_problem[row["problem_id"]].append(row)
    distribution = {"0%": 0, "1-99%": 0, "100%": 0, "incomplete": 0}
    problems, sft_data, pairs = [], [], []
    for key, task in tasks.items():
        samples = by_problem[key]
        passed = sum(row["passed"] for row in samples)
        expected = len(STYLES) * SAMPLES_PER_STYLE
        if len(samples) != expected:
            bucket = "incomplete"
        elif passed == expected:
            bucket = "100%"
        else:
            bucket = "1-99%" if passed else "0%"
        distribution[bucket] += 1
        sft, pair, conflicts = pick_examples(task, samples)
        problems.append({
            "problem_id": key, "split": task["split"], "attempted": len(samples), "passed": passed,
            "bucket": bucket, "usable_sft": sft is not None, "usable_orpo": pair is not None,
            "conflicting_normalized_solutions": conflicts,
            "copied_docstrings": sum(row["copied_docstring"] for row in samples),
            "by_style": {style: {"attempted": len(section), "passed": sum(row["passed"] for row in section)}
                         for style in STYLES
                         for section in [[row for row in samples if row["style"] == style]]},
        })
        if sft:
            sft_data.append(sft)
        if pair:
            pairs.append(pair)
    report = {
        "audited_problem_count": audited_count, "verified_problem_count": len(tasks),
        "expected_samples_per_problem": len(STYLES) * SAMPLES_PER_STYLE,
        "distribution": distribution,
        "all_samples": sample_summary(judgments),
        "passing_samples": sample_summary([row for row in judgments if row["passed"]]),
        "failing_samples": sample_summary([row for row in judgments if not row["passed"]]),
        "by_style": {style: sample_summary([row for row in judgments if row["style"] == style])
                     for style in STYLES},
        "failures_by_category": dict(Counter(row.get("failure_category") or "unknown"
                                             for row in judgments if not row["passed"])),
        "problems_yielding_sft": len(sft_data), "problems_yielding_orpo": len(pairs),
        "sft_by_style": dict(Counter(row["prompt_style"] for row in sft_data)),
        "orpo_by_style": dict(Counter(row["prompt_style"] for row in pairs)),
        "samples_with_exact_holdout_code_collision": sum(bool(row.get("holdout_code_collision")) for row in judgments),
        "per_problem": problems,
        "copy_detection": {
            "copied_docstrings": "Normalized contiguous description inside a Python docstring, including cut-off triple-quoted text.",
            "copied_docstrings_loose": "Strict copies plus high-overlap or partial descriptions inside docstrings.",
            "description_reuse_any": "Loose docstring copies plus high-overlap comments and leading prose.",
        },
    }
    return report, sft_data, pairs


def rerun_audit(config_path, verified, output_dir):
    config = read_json(config_path)
    if not AUDIT_INPUTS <= set(config) <= AUDIT_INPUTS | OPTIONAL_AUDIT_INPUTS:
        raise ValueError(f"Audit config requires: {', '.join(sorted(AUDIT_INPUTS))}. "
                         f"Optional: {', '.join(sorted(OPTIONAL_AUDIT_INPUTS))}.")
    paths = {key: (config_path.parent / value).resolve() for key, value in config.items()}
    with tempfile.TemporaryDirectory(prefix="pyroton-final-audit-", dir=output_dir) as temporary:
        scratch = Path(temporary)
        command = [sys.executable, str(Path(__file__).with_name("build_pilot_corpus.py"))]
        for key, path in sorted(paths.items()):
            command.extend(("--" + key.replace("_", "-"), str(path)))
        command.extend(("--output-dir", str(scratch)))
        subprocess.run(command, check=True, timeout=600)
        corpus_path = scratch / "v2-pilot-corpus.json"
        audit = read_json(scratch / "v2-pilot-corpus-audit.json")
        if audit["corpus_sha256"] != hashlib.sha256(corpus_path.read_bytes()).hexdigest():
            raise ValueError("The rebuilt corpus and its audit digest disagree.")
        expected_hashes = {
            key.replace("_", "-"): hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items() if key != "splits_dir"
        }
        for name in ("dev", "final-test", "split-manifest"):
            expected_hashes[name] = hashlib.sha256((paths["splits_dir"] / (name + ".json")).read_bytes()).hexdigest()
        if any(audit["input_sha256"].get(name) != digest for name, digest in expected_hashes.items()):
            raise ValueError("The fresh audit did not bind all expected training and holdout snapshots.")
        audited = unique_index(read_json(corpus_path), lambda row: row["id"], "audited problem")
        if set(audited) != set(audit["accepted_ids"]):
            raise ValueError("The rebuilt corpus and its audit IDs disagree.")
        for key, task in verified.items():
            if key not in audited or any(task.get(field) != value for field, value in audited[key].items()):
                raise ValueError(f"{key} is absent or changed in the fresh holdout audit.")
    args = SimpleNamespace(**{"old_comparison": None, **paths})
    code_index = defaultdict(set)
    for group in (load_primary(args)[0], load_old(args)[0], load_mbppplus(args)[0]):
        for row in group:
            if row["code"]:
                code_index[row["code"]].add(row["label"])
    return audit, set(audited) - set(verified), len(audited), code_index


def gate_candidate_code(rows, code_index):
    collisions = []
    for row in rows:
        matches = code_index.get(code_key(row["code"]), ()) if row["code"] else ()
        if matches:
            row["holdout_code_collision"] = sorted(matches)
            collisions.append({field: row[field] for field in (
                "problem_id", "style", "sample_index", "holdout_code_collision",
            )})
    return collisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, help="JSON mapping of corpus builder input names to paths.")
    parser.add_argument("--progress-only", action="store_true", help="Report scored progress without writing training data.")
    args = parser.parse_args()
    tasks = unique_index(read_json(args.verified_corpus), lambda row: row["id"], "verified problem")
    if not tasks:
        raise ValueError("The verified corpus is empty.")
    judgments = validate_judgments(read_jsonl(args.judgments), tasks, complete=not args.progress_only)
    verified_hash = hashlib.sha256(args.verified_corpus.read_bytes()).hexdigest()
    provenance = run_provenance(judgments, verified_hash, complete=not args.progress_only)
    if not args.progress_only and not args.audit_config:
        parser.error("--audit-config is required before writing training data.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.progress_only:
        report, _, _ = summarize(tasks, judgments, None)
        report["run_configuration"] = provenance
        write_json(args.output_dir / "pilot-progress-report.json", report)
    else:
        audit, filtered, audited_count, code_index = rerun_audit(args.audit_config, tasks, args.output_dir)
        audit["sample_code_exclusions"] = gate_candidate_code(judgments, code_index)
        report, sft, pairs = summarize(tasks, judgments, audited_count)
        report["run_configuration"] = provenance
        report["excluded_before_sampling_ids"] = sorted(filtered)
        report["holdout_audit_inputs_sha256"] = audit["input_sha256"]
        report["verified_corpus_sha256"] = verified_hash
        report["judgments_sha256"] = hashlib.sha256(args.judgments.read_bytes()).hexdigest()
        write_json(args.output_dir / "pilot-final-overlap-audit.json", audit)
        write_jsonl(args.output_dir / "sft.jsonl", sft)
        write_jsonl(args.output_dir / "orpo.jsonl", pairs)
        write_json(args.output_dir / "pilot-report.json", report)
    print(json.dumps({key: report[key] for key in (
        "distribution", "problems_yielding_sft", "problems_yielding_orpo",
    )}, indent=2))


if __name__ == "__main__":
    main()
