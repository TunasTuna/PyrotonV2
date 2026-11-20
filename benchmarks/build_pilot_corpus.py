"""Freeze the official MBPP train/validation pool after evaluation overlap checks."""

import argparse
import ast
import gzip
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path

from build_mbppplus_holdout import (
    RELEASE_SHA256,
    description,
    features,
    matches,
)
from score_holdout import extract

MBPP_REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
MBPP_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
SANITIZED_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
URL = f"https://raw.githubusercontent.com/google-research/google-research/{MBPP_REVISION}/mbpp/mbpp.jsonl"
# The MBPP v1 curation's explicit benchmark-related exclusions, plus two
# train-side matches from its primary-suite overlap audit.
PRIOR_BLOCKED = {605, 610, 612, 614, 627, 631, 632, 730, 733, 736, 743,
                 755, 771, 778, 779, 786, 792, 793, 800, 806}
# The review prioritizes equivalent tested behavior even when argument
# counts, implementation algorithms, collection types or polarity differ.
MANUAL_NEAR = {
    516: "MBPP+/71 general sorting despite a radix-specific prompt",
    538: "MBPP+/310 split a string into a tuple of its characters",
    549: "MBPP+/267 power sum over the first n odd integers",
    554: "MBPP+/412 filter the odd integers by removing evens",
    555: "MBPP+/422 first-n natural-number cube-sum variant",
    557: "HumanEval/27 swap the case of each character",
    562: "MBPP+/290 select the maximum sublist length",
    566: "Old40/digit_sum decimal digit sum on the nonnegative domain",
    567: "HumanEval/126 sorted-list predicate",
    576: "MBPP+/69 same contiguous sublist membership predicate",
    578: "Old40/interleave equal-length specialization",
    586: "Old40/rotate_left rotates a list instead of a string",
    622: "HumanEval/47 median after joining two lists",
    649: "MBPP+/170 sum a range of list elements by index",
    652: "Old40/transpose tuple-form matrix transpose",
    666: "Old40/char_counts single-character specialization",
    667: "HumanEval/64 vowel counting without the special final-y rule",
    669: "Old40/parse_ipv4 same IPv4 component validation with boolean output",
    670: "HumanEval/57 monotonicity restricted to one direction",
    674: "Old40/dedupe sequence of words rather than integers",
    677: "Old40/triangle_valid same side-length validity predicate",
    678: "Old40/collapse_ws removing instead of condensing whitespace",
    680: "HumanEval/57 monotonicity restricted to one direction",
    686: "Old40/char_counts counts all list items instead of characters",
    687: "Old40/gcd same calculation using recursive Euclid",
    693: "Old40/collapse_ws same repeated-space handling",
    705: "HumanEval/149 length and value ordering over nested lists",
    712: "Old40/dedupe sequence of lists rather than integers",
    773: "HumanEval/18 substring occurrences with their positions",
    802: "HumanEval/109 same sorted-by-rotation property with a count output",
    819: "Old40/rle same consecutive-run counting on a list",
    821: "MBPP+/87 dictionary merge with two inputs instead of three",
    824: "MBPP+/412 parity-complement filtering with the same list contract",
    829: "Old40/top_k_words fixed rank variant of frequency selection",
    841: "MBPP+/296 same inversion count",
    853: "MBPP+/453 sum divisors of the opposite parity",
    862: "Old40/top_k_words same selection with counts returned",
    864: "Old40/palindrome repeated over a list of inputs",
    871: "HumanEval/154 exact rotation rather than its substring variant",
    873: "HumanEval/55 same Fibonacci sequence with a recursion instruction",
    888: "MBPP+/273 elementwise tuple subtraction extended to nesting",
    892: "Old40/collapse_ws same repeated-space handling",
    902: "Old40/merge_counts same addition of values for shared keys",
    910: "HumanEval/124 same calendar validity with a different input shape",
    916: "HumanEval/40 triplet sum with a variable target",
    921: "Old40/chunks same chunking operation for tuples",
    932: "Old40/dedupe words instead of integers",
    933: "MBPP+/102 inverse conversion between snake_case and camelCase",
    940: "MBPP+/428 same sorting behavior with a different requested algorithm",
    942: "Old40/intersection returns existence rather than the common values",
    943: "Old40/merge_sorted same merge with a heapq implementation",
    947: "MBPP+/90 opposite extremum of word lengths",
    958: "HumanEval/156 same integer-to-Roman mapping",
    964: "MBPP+/79 same word-length parity expressed oppositely",
    965: "MBPP+/102 inverse conversion between snake_case and camelCase",
    973: "Old40/rotate_left same left rotation of a string",
}
ALLOWED_MODULES = {
    "array", "bisect", "cmath", "collections", "copy", "datetime", "decimal",
    "fractions", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "sys", "typing",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}
OUTPUT_INSTRUCTION = "Return only complete Python code, including needed imports."


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def parse(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def portable_source(nodes):
    for node in nodes:
        for item in ast.walk(node):
            if isinstance(item, ast.Import):
                if any(alias.name.split(".")[0] not in ALLOWED_MODULES for alias in item.names):
                    return False
            if isinstance(item, ast.ImportFrom):
                if item.level or not item.module or item.module.split(".")[0] not in ALLOWED_MODULES:
                    return False
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                if item.func.id in BLOCKED_CALLS:
                    return False
    return True


def normalize_task(row):
    reference = row["code"].replace("\r\n", "\n").strip()
    tests, challenge = row["test_list"], row["challenge_test_list"]
    setup = row["test_setup_code"].replace("\r\n", "\n").strip()
    if not reference or not tests:
        return None, "missing reference code or original tests"
    try:
        tree = parse(reference)
        test_trees = [parse(test) for test in tests]
        supporting = [parse(setup), *(parse(test) for test in challenge)]
    except SyntaxError:
        return None, "reference or original tests are invalid Python 3"
    if not portable_source([tree, *test_trees, *supporting]):
        return None, "reference or tests require a nonportable import or dynamic execution"
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = Counter(
        call.func.id for test in test_trees for call in ast.walk(test)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in functions
    )
    if not calls:
        return None, "no top-level synchronous reference function in the original tests"
    ranked = calls.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, "ambiguous tested entry point"
    entry = ranked[0][0]
    function = functions[entry]
    signature = f"{entry}({ast.unparse(function.args)})"
    if function.returns is not None:
        signature += f" -> {ast.unparse(function.returns)}"
    if "\n" in signature:
        return None, "multiline function signature"
    task_text = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", row["text"])).strip()
    if not task_text:
        return None, "missing problem statement"
    task_id = row["task_id"]
    stub_text = task_text.replace(chr(92), chr(92) * 2).replace('"""', '\\"\\"\\"')
    plain = f"{task_text}\n\nImplement `{signature}`. {OUTPUT_INSTRUCTION}"
    stub = f'Implement the function below. {OUTPUT_INSTRUCTION}\n\ndef {signature}:\n    """{stub_text}"""'
    return {
        "id": f"Mbpp/{task_id}", "source": "MBPP",
        "split": "train" if task_id >= 601 else "validation",
        "entry_point": entry, "signature": signature, "description": task_text,
        "reference": reference, "checks": tests, "test_setup": setup, "challenge_checks": challenge,
        "prompts": {"mbpp": plain, "humaneval": stub},
    }, None


def overlap_reason(match):
    if match["same_normalized_text"]:
        return "exact normalized task text"
    if match["same_normalized_ast"]:
        return "exact normalized reference AST"
    jaccard = match["keyword_jaccard"]
    text = match["text_sequence_similarity"]
    code = match["ast_similarity"]
    if jaccard >= 0.64 and text >= 0.70:
        return "high task-text similarity"
    if jaccard >= 0.35 and text >= 0.45 and code >= 0.82:
        return "combined text and reference-code near match"
    return None


def audit_overlap(task, sanitized, evaluation):
    representations = [("original", features(task["id"], task["description"], task["reference"]))]
    if sanitized is not None:
        representations.append(("sanitized", features(task["id"], sanitized["prompt"], sanitized["code"])))
    findings = {}
    for variant, representation in representations:
        for match in matches(representation, evaluation):
            candidate = {**match, "task_representation": variant, "reason": overlap_reason(match)}
            old = findings.get(match["against"])
            score = (bool(candidate["reason"]), candidate["keyword_jaccard"], candidate["ast_similarity"])
            if old is None or score > (bool(old["reason"]), old["keyword_jaccard"], old["ast_similarity"]):
                findings[match["against"]] = candidate
    return sorted(findings.values(), key=lambda match: (not bool(match["reason"]), match["against"]))


def load_primary(args):
    primary = read_json(args.primary)
    dev = read_json(args.splits_dir / "dev.json")
    final = read_json(args.splits_dir / "final-test.json")
    manifest = read_json(args.splits_dir / "split-manifest.json")
    dev_keys = {row["id"] for row in dev}
    final_keys = {row["id"] for row in final}
    if (len(primary), len(dev), len(final)) != (160, 60, 100):
        raise ValueError("Expected the full original 160-task set and the 60/100 partitions.")
    if dev_keys & final_keys or {row["id"]: row for row in dev + final} != {row["id"]: row for row in primary}:
        raise ValueError("The partitions must be disjoint and exactly cover the original suite.")
    if manifest["source_sha256"] != sha256(args.primary):
        raise ValueError("The original holdout differs from the frozen split source.")
    for filename in ("dev.json", "final-test.json"):
        if sha256(args.splits_dir / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Unexpected contents in the frozen partition {filename}.")
    index = [features(row["id"], row["description"], row["reference"]) for row in primary]
    return index, dev_keys, final_keys


def load_old(args):
    report = read_json(args.old_report)[:40]
    if len(report) != 40 or len({row["name"] for row in report}) != 40:
        raise ValueError("Expected all 40 original benchmark prompts.")
    index = [
        features(f"Old40/{row['name']}", row["prompt"], extract(row["answer"]) if row["passed"] else "")
        for row in report
    ]
    if args.old_comparison:
        ids = {row["name"] for row in report}
        for model in read_json(args.old_comparison)["models"]:
            for row in model["rows"][:40]:
                if row["name"] not in ids:
                    raise ValueError("Old benchmark comparison does not match the original 40 IDs.")
                if row["passed"]:
                    index.append(features(f"Old40/{row['name']}", row["prompt"], extract(row["answer"])))
    return index, {row["name"] for row in report}


def load_mbppplus(args):
    if sha256(args.mbppplus_release) != RELEASE_SHA256:
        raise ValueError("Unexpected MBPP+ archive revision.")
    with gzip.open(args.mbppplus_release, "rt", encoding="utf-8") as stream:
        release = [json.loads(line) for line in stream if line.strip()]
    official = [row for row in release if 11 <= int(row["task_id"].rsplit("/", 1)[1]) <= 510]
    audit = read_json(args.mbppplus_audit)
    prior = {row["id"]: row for row in audit["audit"]}
    if (len(release) != 378 or len(official) != 224 or audit["release_sha256"] != RELEASE_SHA256
            or set(prior) != {row["task_id"] for row in official}):
        raise ValueError("The MBPP+ audit does not match the pinned official test membership.")
    previously_excluded = {}
    for row in prior.values():
        for decision in row["exclusions"]:
            if decision["against"].startswith("MBPP/"):
                number = int(decision["against"].rsplit("/", 1)[1])
                previously_excluded.setdefault(number, set()).add(row["id"])
    index = [
        features(f"MBPP+/{row['task_id'].rsplit('/', 1)[1]}", description(row["prompt"]),
                 row["prompt"] + row["canonical_solution"])
        for row in official
    ]
    kept = {row["id"] for row in prior.values() if not row["exclusions"]}
    return index, previously_excluded, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("mbpp", "sanitized", "primary", "splits-dir", "old-report", "mbppplus-release",
                   "mbppplus-audit", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--old-comparison", type=Path)
    args = parser.parse_args()
    if sha256(args.mbpp) != MBPP_SHA256 or sha256(args.sanitized) != SANITIZED_SHA256:
        raise ValueError("MBPP inputs must match the pinned Google Research revision.")
    official_mbpp = read_lines(args.mbpp)
    if sorted(row["task_id"] for row in official_mbpp) != list(range(1, 975)):
        raise ValueError("Unexpected official MBPP membership.")
    sanitized = {row["task_id"]: row for row in read_json(args.sanitized)}
    primary_index, dev, final = load_primary(args)
    old_index, old = load_old(args)
    mbppplus_index, prior, mbppplus_kept = load_mbppplus(args)
    evaluation = primary_index + old_index + mbppplus_index

    accepted, decisions = [], []
    source_counts = Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for row in official_mbpp:
            number = row["task_id"]
            if not 511 <= number <= 974:
                continue
            split = "train" if number >= 601 else "validation"
            source_counts[split] += 1
            task_id = f"Mbpp/{number}"
            exclusions = []
            if number in PRIOR_BLOCKED:
                exclusions.append("previously blocked for coupling to the primary or original benchmark")
            if number in prior:
                exclusions.append("prior MBPP+ overlap finding against " + ", ".join(sorted(prior[number])))
            task, error = normalize_task(row)
            if error:
                exclusions.append(error)
            # Audit every original prompt and reference, including rows rejected
            # by the portability check.
            candidate = task or {"id": task_id, "description": row["text"], "reference": row["code"]}
            related = audit_overlap(candidate, sanitized.get(number), evaluation)
            for hit in related:
                if hit["reason"]:
                    exclusions.append(f"{hit['reason']} against {hit['against']}")
            if number in MANUAL_NEAR:
                exclusions.append("manual near match against " + MANUAL_NEAR[number])
            decisions.append({
                "id": task_id, "split": split, "entry_point": task["entry_point"] if task else None,
                "description": candidate["description"], "exclusions": exclusions, "matched_evaluation": related,
            })
            if not exclusions:
                accepted.append(task)
    if not accepted:
        raise ValueError("No eligible MBPP problems survived the overlap and portability audit.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "v2-pilot-corpus.json"
    write_json(corpus_path, accepted)
    input_paths = {
        "mbpp": args.mbpp, "sanitized": args.sanitized, "primary": args.primary,
        "dev": args.splits_dir / "dev.json", "final-test": args.splits_dir / "final-test.json",
        "split-manifest": args.splits_dir / "split-manifest.json", "old-report": args.old_report,
        "mbppplus-release": args.mbppplus_release, "mbppplus-audit": args.mbppplus_audit,
    }
    if args.old_comparison:
        input_paths["old-comparison"] = args.old_comparison
    report = {
        "mbpp_source": URL, "mbpp_revision": MBPP_REVISION, "input_sha256": {
            key: sha256(path) for key, path in input_paths.items()
        },
        "corpus_sha256": sha256(corpus_path), "accepted_ids": [row["id"] for row in accepted],
        "sample_configuration": {"samples_per_problem": 8, "per_style": 4, "temperature": 0.8, "max_tokens": 512},
        "counts": {"candidates_by_split": dict(source_counts),
                   "accepted_by_split": dict(Counter(row["split"] for row in accepted)),
                   "candidates": len(decisions), "accepted": len(accepted), "excluded": len(decisions) - len(accepted)},
        "reserved": {
            "old40": len(old), "dev": len(dev), "final": len(final),
            "mbppplus_official_test_pool": len(mbppplus_index), "mbppplus_final_holdout": len(mbppplus_kept),
        },
        "overlap_method": "Exact official split IDs, normalized task text/reference AST, conservative text+AST similarity thresholds, manual near-match review, and recorded prior exclusions. Old40 code compares passing outputs as proxies.",
        "audit": decisions,
    }
    write_json(args.output_dir / "v2-pilot-corpus-audit.json", report)
    print(json.dumps(report["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
