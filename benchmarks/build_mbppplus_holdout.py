"""Freeze the audited MBPP+ v0.2.0 test subset for final evaluation."""

import argparse
import ast
import difflib
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from build_holdout import code_key, load_original_examples, words
from mbppplus_cases import SOURCE, oracle_for
from score_holdout import extract


RELEASE_SHA256 = "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
RELEASE = "https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0"
MBPP_SPLITS = "https://github.com/google-research/google-research/tree/master/mbpp#evaluation-details"
COMMON_WORDS = set(
    "write python function given takes returns return find check code the to of in a an and for "
    "from by is or that given which as it whether if".split()
)

# Conservative review: the central transformation is strongly related, even
# where an input type, output type, or a fixed argument differs.
REVIEW_EXCLUSIONS = {
    19: ("Old40/dedupe", "Duplicate detection versus duplicate removal."),
    65: ("Old40/flatten_one", "Flattening wrapped in a sum."),
    67: ("MBPP/608", "Both compute Bell numbers."),
    68: ("HumanEval/57", "Both check whether a collection is monotonic."),
    71: ("MBPP/635", "Same unrestricted list-sorting prompt."),
    88: ("Old40/char_counts", "Counting items in a list versus counting characters in a string."),
    90: ("HumanEval/12", "Longest string length versus the longest string itself."),
    91: ("HumanEval/7", "Same substring containment over a list of strings."),
    93: ("MBPP/623", "Scalar power versus the same power mapped over a list."),
    97: ("Old40/char_counts", "Same frequency count after flattening nested input."),
    99: ("HumanEval/79", "Decimal-to-binary conversion with different formatting."),
    101: ("PyrotonStrict/safe_index", "Direct list indexing with a one-based variant."),
    106: ("MBPP/750", "Concatenating the same list and tuple in the opposite result type."),
    108: ("Old40/merge_sorted", "Three-way variant of sorted merging."),
    111: ("Old40/intersection", "Intersection generalized to nested collections."),
    118: ("HumanEval/101", "Space-only variant of splitting a string into words."),
    140: ("Old40/flatten_one", "Flattening with a set result."),
    141: ("MBPP/635", "Same unrestricted list-sorting prompt."),
    142: ("MBPP/723", "Counting equal positions over three lists instead of two."),
    161: ("MBPP/769", "Both compute list difference."),
    172: ("HumanEval/18", "Substring occurrence count with a fixed substring."),
    230: ("MBPP/628", "Same space replacement with a configurable replacement."),
    232: ("HumanEval/120", "Both select the largest n values of a list."),
    237: ("MBPP/758", "Both count occurrences of nested sequences keyed by tuple."),
    251: ("HumanEval/5", "Same repeated delimiter insertion with different endpoint rules."),
    253: ("HumanEval/22", "Same integer filter counted instead of returned."),
    256: ("HumanEval/96", "Same below-bound prime enumeration with a count result."),
    261: ("MBPP/618", "Same elementwise division over tuples instead of lists."),
    265: ("Old40/chunks", "Both split a list into consecutive fixed-size chunks."),
    267: ("MBPP/770", "Sum powers of the first n odd natural numbers with a different exponent."),
    270: ("HumanEval/85", "Same index-and-value parity sum with an index parity change."),
    271: ("MBPP/770", "Sum powers of a parity-restricted range with different parity and exponent."),
    281: ("Old40/dedupe", "Duplicate detection versus duplicate removal."),
    282: ("MBPP/618", "Same elementwise two-list operation with a different operator."),
    284: ("MBPP/760", "Checking one distinct value with an explicit expected value."),
    285: ("MBPP/787", "Matching 'a' followed by repeated 'b' with a small quantifier change."),
    287: ("MBPP/770", "Sum powers of a parity-restricted range with different parity and exponent."),
    290: ("HumanEval/12", "Same longest-sequence selection over sublists instead of strings."),
    297: ("Old40/flatten_one", "Recursive variant of list flattening."),
    305: ("HumanEval/29", "Fixed-prefix, count-limited version of prefix filtering."),
    394: ("Old40/dedupe", "Duplicate detection over tuples instead of removal over lists."),
    395: ("Old40/first_unique", "Both return the first non-repeating character."),
    397: ("HumanEval/47", "Three-value specialization of median."),
    398: ("Old40/digit_sum", "Same digit sum mapped over a list."),
    405: ("MBPP/808", "Same tuple membership check."),
    412: ("MBPP/629", "Removing odd values is filtering even values."),
    414: ("Old40/intersection", "Checks whether the same intersection is nonempty."),
    418: ("HumanEval/12", "Same longest-sequence selection."),
    420: ("MBPP/770", "Sum powers of a parity-restricted range with different parity and exponent."),
    422: ("MBPP/770", "Related consecutive powers aggregation with a normalization step."),
    433: ("HumanEval/52", "Same strict upper-bound predicate over a collection."),
    447: ("MBPP/623", "Cube specialization of per-element exponentiation."),
    451: ("Old40/collapse_ws", "Removing whitespace versus collapsing whitespace."),
    454: ("MBPP/643", "Same character search with different word-boundary rules."),
    455: ("MBPP/762", "Same month-length classification with 31 rather than 30 days."),
    473: ("Old40/intersection", "Same intersection operation over tuples."),
    474: ("MBPP/628", "Same string replacement with arbitrary characters."),
    477: ("MBPP/624", "Case-conversion counterpart with reversed direction."),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_number(task_id):
    return int(task_id.rsplit("/", 1)[-1])


def split(number):
    if number <= 10:
        return "prompt"
    if number <= 510:
        return "test"
    if number <= 600:
        return "validation"
    return "train"


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def description(prompt):
    try:
        doc = ast.get_docstring(ast.parse(prompt))
    except SyntaxError:
        doc = None
    if doc is not None:
        return "\n".join(line for line in doc.splitlines() if not line.lstrip().startswith("assert ")).strip()
    return prompt


def features(label, prompt, code):
    return {
        "label": label,
        "words": words(prompt),
        "terms": set(words(prompt)) - COMMON_WORDS,
        "code": code_key(code) if code else "",
    }


def matches(task, index):
    flagged = []
    for reference in index:
        text_equal = bool(task["words"] and task["words"] == reference["words"])
        ast_equal = bool(task["code"] and task["code"] == reference["code"])
        word_union = task["terms"] | reference["terms"]
        shared_terms = len(task["terms"] & reference["terms"])
        jaccard = shared_terms / len(word_union) if word_union else 0.0
        if not text_equal and not ast_equal and (jaccard < 0.35 or shared_terms < 3):
            continue
        text_ratio = difflib.SequenceMatcher(None, task["words"], reference["words"], autojunk=False).ratio()
        ast_ratio = (
            difflib.SequenceMatcher(None, task["code"].split(", "), reference["code"].split(", "), autojunk=False).ratio()
            if task["code"] and reference["code"] else 0.0
        )
        flagged.append({
            "against": reference["label"],
            "same_normalized_text": text_equal,
            "same_normalized_ast": ast_equal,
            "text_sequence_similarity": round(text_ratio, 3),
            "keyword_jaccard": round(jaccard, 3),
            "ast_similarity": round(ast_ratio, 3),
        })
    return flagged


def load_training(args):
    metadata = json.loads(args.training_metadata.read_text(encoding="utf-8"))
    expected = metadata["dataset"]
    if sha256(args.mbpp) != expected["source_sha256"]:
        raise ValueError("The MBPP source does not match the recorded training manifest.")
    selected = set(expected["task_ids"])
    rows = json.loads(args.mbpp.read_text(encoding="utf-8"))
    corpus = [
        features(f"MBPP/{row['task_id']}", row["prompt"], row["code"])
        for row in rows if row["task_id"] in selected
    ]
    if len(corpus) != len(selected):
        raise ValueError("Could not resolve every recorded MBPP training ID.")
    pilot = load_original_examples(args.notebook)
    corpus.extend(features(f"Pilot/{i}", row[0], row[1]) for i, row in enumerate(pilot, 1))
    if len(corpus) != metadata["examples"]:
        raise ValueError("Reconstructed training count differs from the recorded count.")

    repository_rows = {}
    for path in sorted(args.repo_data.glob("*.jsonl")):
        entries = jsonl(path)
        repository_rows[path.name] = len(entries)
        for i, entry in enumerate(entries, 1):
            prompt = entry.get("instruction") or entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Unrecognized training row at {path}:{i}.")
            codes = [entry[key] for key in ("code", "solution", "chosen", "rejected", "fixed", "buggy")
                     if isinstance(entry.get(key), str)]
            for code in codes or [""]:
                corpus.append(features(f"Repo/{path.name}:{i}", prompt, extract(code)))
    return corpus, selected, len(pilot), repository_rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "mbpp", "training-metadata", "notebook", "primary", "old-report", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repo-data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--reference-scores", type=Path,
                        help="Reference preflight scores from score_holdout.py. Freeze only passing items.")
    args = parser.parse_args()
    actual_hash = sha256(args.release)
    if actual_hash != RELEASE_SHA256:
        raise ValueError("Input is not the pinned MBPP+ v0.2.0 release archive.")

    training, trained_ids, pilot_count, repo_rows = load_training(args)
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    if len(primary) != 160 or len({row["id"] for row in primary}) != 160:
        raise ValueError("Expected the frozen, scored 160-task primary suite.")
    old = json.loads(args.old_report.read_text(encoding="utf-8"))[:40]
    if len(old) != 40 or len({row["name"] for row in old}) != 40:
        raise ValueError("Expected 40 distinct original benchmark tasks.")
    evaluation = [
        features(row["id"], row["description"], row["reference"]) for row in primary
    ] + [
        features(f"Old40/{row['name']}", row["prompt"], extract(row["answer"]) if row["passed"] else "")
        for row in old
    ]
    indexed_labels = {row["label"] for row in (*training, *evaluation)}
    if any(label not in indexed_labels for label, _ in REVIEW_EXCLUSIONS.values()):
        raise ValueError("A manual exclusion refers to a task outside the audited corpora.")
    with gzip.open(args.release, "rt", encoding="utf-8") as stream:
        release = [json.loads(line) for line in stream if line.strip()]
    release_ids = [task_number(row["task_id"]) for row in release]
    if len(set(release_ids)) != len(release_ids) or len(release_ids) != 378:
        raise ValueError("Unexpected release membership.")

    accepted, details = [], []
    for row in release:
        task_id = row["task_id"]
        number = task_number(task_id)
        partition = split(number)
        if partition != "test":
            continue
        task = features(task_id, description(row["prompt"]), row["canonical_solution"])
        train_hits = matches(task, training)
        eval_hits = matches(task, evaluation)
        reasons = []
        if number in trained_ids:
            reasons.append({"against": f"MBPP/{number}", "reason": "Exact recorded training task ID."})
        if number in REVIEW_EXCLUSIONS:
            against, reason = REVIEW_EXCLUSIONS[number]
            reasons.append({"against": against, "reason": reason})
        for hit in (*train_hits, *eval_hits):
            if hit["same_normalized_text"] or hit["same_normalized_ast"]:
                reasons.append({"against": hit["against"], "reason": "Exact normalized text or reference AST."})
        oracle = oracle_for(row["entry_point"])
        if oracle in ("unsupported", "any-output"):
            reasons.append({"against": "EvalPlus oracle", "reason": f"Cannot score comparator {oracle!r} in WASI."})
        details.append({
            "id": task_id, "description": description(row["prompt"]), "oracle": oracle,
            "exclusions": reasons, "training_matches": train_hits, "eval_matches": eval_hits,
            "base_cases": len(row["base_input"]), "plus_cases": len(row["plus_input"]),
        })
        if not reasons:
            accepted.append(row)

    if not accepted or len(details) != 224 or not set(REVIEW_EXCLUSIONS).issubset({task_number(row["id"]) for row in details}):
        raise ValueError("Unexpected test membership or invalid review exclusions.")
    overlap_count = len(details) - len(accepted)
    if args.reference_scores:
        scores = json.loads(args.reference_scores.read_text(encoding="utf-8"))
        by_id = {row["id"]: row for row in scores}
        if len(by_id) != len(scores):
            raise ValueError("Duplicate IDs in reference preflight scores.")
        failed = {}
        for row in accepted:
            score = by_id.get(row["task_id"])
            case_count = len(row["base_input"]) + len(row["plus_input"])
            if not score or score["source"] != SOURCE or score["suite_cases"] != case_count:
                raise ValueError(f"Missing or mismatched reference score for {row['task_id']}.")
            if not score["passed"]:
                failed[row["task_id"]] = score["error"] or "reference failed in the sandbox"
        for row in details:
            if row["id"] in failed:
                row["exclusions"].append({"against": "Reference preflight", "reason": failed[row["id"]]})
        accepted = [row for row in accepted if row["task_id"] not in failed]
    record = {
        "release": RELEASE,
        "release_sha256": actual_hash,
        "official_split_definition": MBPP_SPLITS,
        "release_split_counts": dict(Counter(map(split, release_ids))),
        "recorded_mbpp_training_ids_in_release": sorted(trained_ids.intersection(release_ids)),
        "recorded_training_id_collisions_in_test": sorted(trained_ids.intersection(
            task_number(row["id"]) for row in details
        )),
        "recorded_mbpp_training_count": len(trained_ids),
        "pilot_count": pilot_count,
        "repo_data_rows": repo_rows,
        "primary_count": len(primary),
        "old_suite_count": len(old),
        "official_test_candidates": len(details),
        "accepted": len(accepted),
        "excluded": len(details) - len(accepted),
        "excluded_before_reference_preflight": overlap_count,
        "reference_preflight_sha256": sha256(args.reference_scores) if args.reference_scores else None,
        "accepted_base_cases": sum(len(row["base_input"]) for row in accepted),
        "accepted_plus_cases": sum(len(row["plus_input"]) for row in accepted),
        "sources_sha256": {
            name: sha256(getattr(args, name))
            for name in ("mbpp", "training_metadata", "notebook", "primary", "old_report")
        },
        "audit": details,
    }
    generation_rows = [
        {
            "id": row["task_id"], "source": SOURCE, "entry_point": row["entry_point"],
            "prompt": "Implement the function below. Return only complete Python code, including imports.\n\n"
                      + row["prompt"],
            "reference": row["prompt"] + row["canonical_solution"],
            "base_input": row["base_input"], "plus_input": row["plus_input"],
            "atol": row["atol"], "oracle": oracle_for(row["entry_point"]),
        }
        for row in accepted
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = args.output_dir / "v2-mbppplus-test.json"
    evalplus_path = args.output_dir / "v2-mbppplus-test-evalplus.jsonl"
    write_json(generation_path, generation_rows)
    with evalplus_path.open("w", encoding="utf-8") as stream:
        for entry in accepted:
            stream.write(json.dumps(entry, ensure_ascii=True) + "\n")
    record["output_sha256"] = {path.name: sha256(path) for path in (generation_path, evalplus_path)}
    write_json(args.output_dir / "v2-mbppplus-audit.json", record)
    print({key: record[key] for key in ("official_test_candidates", "accepted", "excluded",
                                        "accepted_base_cases", "accepted_plus_cases")})


if __name__ == "__main__":
    main()
