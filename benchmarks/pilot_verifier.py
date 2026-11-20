"""Run MBPP checks and reference-derived edge fixtures in a WASI subprocess."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import secrets
import sys
from collections import Counter
from pathlib import Path

from pilot_edges import decode, encode, proposals
from score_holdout import ScoreProcess

WIRE_SOURCE = "\n".join(inspect.getsource(function) for function in (encode, decode))
RUNTIME = """
import copy
import inspect
import json
import math
import sys
import traceback

def short(value, limit=500):
    try:
        return repr(value)[:limit]
    except Exception:
        return "<unrepresentable>"

def failure(category, stage, message, check=None):
    return {"passed": False, "error": str(message)[:700], "failure_category": category,
            "stage": stage, "first_failed_check": check,
            "signature_ok": stage not in ("compile", "signature", "runtime")}

def success():
    return {"passed": True, "error": None, "failure_category": None, "stage": "complete",
            "first_failed_check": None, "signature_ok": True}

def signature_of(function):
    return [
        [parameter.name, parameter.kind.name,
         ["required"] if parameter.default is inspect.Parameter.empty else encode(parameter.default)]
        for parameter in inspect.signature(function, follow_wrapped=False).parameters.values()
    ]

def import_missing(error, import_names):
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return True
    return (
        isinstance(error, NameError)
        and getattr(error, "name", None) in import_names
        and any(frame.filename == "<candidate>" for frame in traceback.extract_tb(error.__traceback__))
    )

def load(code, entry, expected=None, import_names=()):
    namespace = {"__name__": "__candidate__"}
    try:
        exec(compile(code, "<candidate>", "exec"), namespace)
    except SyntaxError as error:
        return None, failure("format", "compile", f"{type(error).__name__}: {error}")
    except Exception as error:
        category = "missing_import" if import_missing(error, import_names) else "other"
        return None, failure(category, "compile", f"{type(error).__name__}: {error}")
    function = namespace.get(entry)
    if not callable(function):
        return None, failure("signature", "signature", f"Missing callable {entry!r}.")
    try:
        actual = signature_of(function)
    except (TypeError, ValueError) as error:
        return None, failure("signature", "signature", error)
    if expected is not None and not wire_signature_equal(actual, expected):
        return None, failure("signature", "signature", f"Expected {short(expected)}, got {short(actual)}.")
    return function, None

def wire_signature_equal(actual, expected):
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if left[:2] != right[:2]:
            return False
        if left[2] == ["required"] or right[2] == ["required"]:
            if left[2] != right[2]:
                return False
        elif not wire_equal(left[2], right[2]):
            return False
    return True

def original(function, entry, tests, setup, import_names):
    check_globals = {"__name__": "__checks__", entry: function}
    try:
        if setup:
            exec(compile(setup, "<test setup>", "exec"), check_globals)
        check_globals[entry] = function
    except Exception as error:
        return failure("other", "original", f"Test setup: {type(error).__name__}: {error}", "setup")
    for check, source in tests:
        try:
            exec(compile(source, f"<{check}>", "exec"), check_globals)
        except Exception as error:
            category = "missing_import" if import_missing(error, import_names) else "logic"
            return failure(category, "original", f"{type(error).__name__}: {error}", check)
    return None

def number_equal(left, right):
    if left == right:
        return True
    try:
        return (
            (isinstance(left, float) and isinstance(right, float)
             and math.isnan(left) and math.isnan(right))
            or math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)
        )
    except (OverflowError, TypeError):
        return False

def wire_equal(left, right):
    left_type, left_value = left
    right_type, right_value = right
    if left_type in ("int", "float") and right_type in ("int", "float"):
        lvalue, rvalue = decode(left), decode(right)
        if left_type == right_type == "int":
            return lvalue == rvalue
        return number_equal(lvalue, rvalue)
    if left_type != right_type:
        return False
    if left_type == "complex":
        return all(number_equal(float(a), float(b)) for a, b in zip(left_value, right_value))
    if left_type in ("list", "tuple", "set", "frozenset"):
        return len(left_value) == len(right_value) and all(
            wire_equal(a, b) for a, b in zip(left_value, right_value)
        )
    if left_type == "dict":
        left_by_key = {repr(key): value for key, value in left_value}
        right_by_key = {repr(key): value for key, value in right_value}
        return left_by_key.keys() == right_by_key.keys() and all(
            wire_equal(value, right_by_key[key]) for key, value in left_by_key.items()
        )
    return left_value == right_value

def observe(function, case, mutation_required, import_names):
    args, kwargs = decode(case["args"]), decode(case["kwargs"])
    try:
        value = function(*args, **kwargs)
    except Exception as error:
        return {"outcome": ["raise", type(error).__name__],
                "import_missing": import_missing(error, import_names),
                "message": str(error)[:350]}
    try:
        result = {"outcome": ["return", encode(value)]}
        if mutation_required:
            result["after_args"] = encode(args)
            result["after_kwargs"] = encode(kwargs)
        if len(json.dumps(result, ensure_ascii=True)) > 2600:
            return {"skip": "Result exceeds fixture size limit."}
        return result
    except (ValueError, OverflowError, TypeError) as error:
        return {"skip": str(error)[:350]}

def freeze_edges(function, cases, rules):
    accepted = []
    skipped = []
    for case in cases:
        observed = observe(function, case, rules["mutation_required"], ())
        if "skip" in observed:
            skipped.append((case["kind"], observed["skip"]))
            continue
        event, result = observed["outcome"]
        specified = rules["expected_exceptions"]
        if event == "raise" and (
            result not in specified and not (
                case["kind"] == "wrong_type" and not specified and result in ("TypeError", "ValueError")
            )
        ):
            skipped.append((case["kind"], "Reference raised an exception outside the stated contract."))
            continue
        if case["kind"] == "wrong_type" and event == "return" and not (
            rules["invalid_sentinel"] and result in (["none", None], ["bool", False], ["int", -1])
        ):
            skipped.append((case["kind"], "No documented invalid-input result was observed."))
            continue
        accepted.append({**case, "expected": {key: value for key, value in observed.items()
                                              if key in ("outcome", "after_args", "after_kwargs")}})
    return {"edges": accepted, "skipped": skipped}

def edge_result(function, edges, mutation_required, import_names):
    for index, case in enumerate(edges, 1):
        expected = case["expected"]
        observed = observe(function, case, mutation_required, import_names)
        check = f"edge:{case['kind']}:{index}"
        if "skip" in observed:
            return failure("edge_validation", "edge", observed["skip"], check)
        got_event, got_value = observed["outcome"]
        expected_event, expected_value = expected["outcome"]
        matched = got_event == expected_event and (
            wire_equal(got_value, expected_value) if got_event == "return" else got_value == expected_value
        )
        if matched and "after_args" in expected:
            matched = (wire_equal(observed["after_args"], expected["after_args"])
                       and wire_equal(observed["after_kwargs"], expected["after_kwargs"]))
        if not matched:
            category = "missing_import" if observed.get("import_missing") else "edge_validation"
            return failure(category, "edge",
                           f"Expected {short(expected)}, observed {short(observed)}.", check)
    return None
"""


def imported_names(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    names.update(sys.stdlib_module_names)
    return sorted(names)


def checks(task):
    return [(f"{label}:{index}", code)
            for label, collection in (("original", task["checks"]), ("challenge", task.get("challenge_checks", [])))
            for index, code in enumerate(collection, 1)]


def build_program(body, **data):
    serialized = json.dumps(data, ensure_ascii=True, allow_nan=False)
    return WIRE_SOURCE + RUNTIME + f"\ndata = json.loads({serialized!r})\n" + body


def reference_check(task, worker, deadline):
    program = build_program("""
function, error = load(data["code"], data["entry"])
if error is None:
    error = original(function, data["entry"], data["tests"], data["setup"], ())
if error is None:
    __pilot_emit({"signature": signature_of(function)})
else:
    __pilot_emit(error)
""", code=task["reference"], entry=task["entry_point"], tests=checks(task),
                            setup=task.get("test_setup", ""))
    return worker.check(program, deadline)


def make_edge_batch(task, batch, rules, worker, deadline):
    program = build_program("""
function, error = load(data["code"], data["entry"])
if error is None:
    __pilot_emit(freeze_edges(function, data["edges"], data["rules"]))
else:
    __pilot_emit(error)
""", code=task["reference"], entry=task["entry_point"], edges=batch, rules=rules)
    return worker.check(program, deadline)


def preflight_task(task, worker, deadline=8):
    """Return a frozen test suite or an explicit rejection, never inferred passes."""
    reference = reference_check(task, worker, deadline)
    if "signature" not in reference:
        return {"accepted": False, "reason": "reference_original_failed", "detail": reference}
    candidates, rules = proposals(task)
    if not candidates:
        return {"accepted": False, "reason": "no_safe_novel_literal_cases"}

    pending = [candidates]
    attempts = 0
    edges = []
    skipped = Counter()
    while pending and attempts < 7:
        batch = pending.pop(0)
        result = make_edge_batch(task, batch, rules, worker, deadline)
        attempts += 1
        if "edges" in result:
            edges.extend(result["edges"])
            skipped.update(f"{kind}: {reason}" for kind, reason in result["skipped"])
            continue
        if len(batch) > 1 and result.get("stage") == "runtime":
            midpoint = len(batch) // 2
            pending.extend((batch[:midpoint], batch[midpoint:]))
        else:
            skipped.update({f"sandbox: {result.get('error', 'execution failed')}": len(batch)})
    if pending:
        skipped["sandbox: bounded retries exhausted"] += sum(map(len, pending))
    if not edges:
        return {"accepted": False, "reason": "no_reference_supported_edge_cases", "detail": dict(skipped)}

    order = {json.dumps(case, sort_keys=True): index for index, case in enumerate(candidates)}
    edges.sort(key=lambda edge: order[json.dumps({k: v for k, v in edge.items() if k != "expected"}, sort_keys=True)])
    suite = {
        "entry_point": task["entry_point"],
        "original_checks": checks(task),
        "test_setup": task.get("test_setup", ""),
        "reference_signature": reference["signature"],
        "generated_edges": edges,
        "compare_mutations": rules["mutation_required"],
    }
    digest = hashlib.sha256(json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "accepted": True,
        "prepared": {
            "reference_signature": reference["signature"],
            "compare_mutations": rules["mutation_required"],
            "generated_edges": edges,
            "test_suite_sha256": digest,
            "verifier_counts": {"original": len(task["checks"]), "challenge": len(task.get("challenge_checks", [])),
                                "edge_proposals": len(candidates), "edges": dict(Counter(edge["kind"] for edge in edges)),
                                "skipped_edges": dict(skipped)},
        },
    }


def verify_candidate(task, code, worker, deadline=8):
    """Score the exact code supplied by the caller against a prepared task."""
    program = build_program("""
function, error = load(data["code"], data["entry"], data["signature"], data["imports"])
if error is None:
    error = original(function, data["entry"], data["tests"], data["setup"], data["imports"])
if error is None:
    error = edge_result(function, data["edges"], data["mutations"], data["imports"])
__pilot_emit(error or success())
""", code=code, entry=task["entry_point"], signature=task["reference_signature"],
                            imports=imported_names(task["reference"]), tests=checks(task),
                            setup=task.get("test_setup", ""), edges=task["generated_edges"],
                            mutations=task["compare_mutations"])
    return worker.check(program, deadline)


class PilotSandbox(ScoreProcess):
    """ScoreProcess with a bounded JSON result channel for reference fixtures."""

    def __init__(self):
        super().__init__(worker_script=__file__)


def worker_main():
    from wasi_python import run

    def execute(source, deadline):
        marker = "PYROTON_PILOT_" + secrets.token_hex(12) + " "
        prefix = "import json as __pilot_json\n"
        prefix += ("def __pilot_emit(value):\n"
                   f"    print({marker!r} + __pilot_json.dumps(value, ensure_ascii=True))\n")
        cache = os.environ.get("PYROTON_WASI_CACHE") or False
        result = run(["-c", prefix + source], deadline_s=deadline, cache_dir=cache,
                     memory_bytes=256 * 1024**2, output_bytes=350_000)
        stderr = result.stderr.decode(errors="replace").strip().splitlines()
        if result.exit_code == 0 and not result.trapped:
            for line in reversed(result.stdout.decode(errors="replace").splitlines()):
                if line.startswith(marker):
                    return json.loads(line[len(marker):])
        error = str(result.trapped) if result.trapped else stderr[-1] if stderr else "No verifier result."
        return {"passed": False, "error": error[:700], "failure_category": "other",
                "stage": "runtime", "first_failed_check": None, "signature_ok": False}

    if execute("__pilot_emit({'ready': True})", 15) != {"ready": True}:
        raise RuntimeError("Cannot initialize the pilot WASI worker.")
    print('{"ready": true}', flush=True)
    for line in sys.stdin:
        call = json.loads(line)
        print(json.dumps(execute(call["code"], call["deadline"]), ensure_ascii=True), flush=True)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--deadline", type=float, default=8)
    args = parser.parse_args()
    if args.worker:
        return worker_main()
    if not args.corpus or not args.output or not args.report:
        parser.error("--corpus, --output and --report are required")
    if args.deadline <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("deadline and limit must be positive")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8-sig"))
    ids = [task["id"] for task in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("The training corpus has duplicate problem IDs.")
    tasks = corpus[:args.limit]
    journal = args.output.with_suffix(".progress.jsonl")
    signature = {"corpus_sha256": file_sha(args.corpus), "verifier_sha256": file_sha(__file__),
                 "edge_generator_sha256": file_sha(Path(__file__).with_name("pilot_edges.py")),
                 "deadline": args.deadline}
    completed = {}
    if journal.exists():
        records = journal.read_text(encoding="utf-8").splitlines()
        if not records or json.loads(records[0]) != signature:
            raise ValueError("Preflight journal belongs to another corpus or verifier.")
        completed = {row["id"]: row["verdict"] for row in map(json.loads, records[1:])}
    else:
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps(signature) + "\n", encoding="utf-8")

    unfinished = [task for task in tasks if task["id"] not in completed]
    if unfinished:
        sandbox = PilotSandbox()
        try:
            with journal.open("a", encoding="utf-8") as stream:
                for index, task in enumerate(unfinished, 1):
                    verdict = preflight_task(task, sandbox, args.deadline)
                    completed[task["id"]] = verdict
                    stream.write(json.dumps({"id": task["id"], "verdict": verdict}, ensure_ascii=True) + "\n")
                    stream.flush()
                    if index % 20 == 0 or index == len(unfinished):
                        print(f"Preflight {index}/{len(unfinished)} new tasks.", flush=True)
        finally:
            sandbox.close()

    accepted = [{**task, **completed[task["id"]]["prepared"]}
                for task in tasks if completed[task["id"]]["accepted"]]
    rejected = [{"id": task["id"], **completed[task["id"]]}
                for task in tasks if not completed[task["id"]]["accepted"]]
    edge_counts = Counter()
    for row in accepted:
        edge_counts.update(row["verifier_counts"]["edges"])
    write_atomic(args.output, accepted)
    write_atomic(args.report, {**signature, "audited_tasks": len(tasks), "total_corpus_tasks": len(corpus),
                                "accepted": len(accepted), "rejected": len(rejected), "edge_counts": dict(edge_counts),
                                "accepted_ids": [row["id"] for row in accepted], "rejected_tasks": rejected,
                                "complete": len(tasks) == len(corpus)})
    print(f"Accepted {len(accepted)}/{len(tasks)} tasks; {sum(edge_counts.values())} frozen edge checks.", flush=True)


if __name__ == "__main__":
    main()
