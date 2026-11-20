"""Score generated Python in a WASI worker subprocess with per-task deadlines."""

import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

from mbppplus_cases import SOURCE as MBPPPLUS_SOURCE, test_program as mbppplus_test_program
from pilot_io import read_jsonl_repair, write_json_atomic

RUNNER = """
import copy
import math
from datetime import date, datetime

def equal(left, right, atol, strict=False):
    if strict and type(left) is not type(right):
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return left == right or math.isclose(left, right, rel_tol=1e-7, abs_tol=atol or 1e-6)
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b, atol, strict) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k], atol, strict) for k in right)
    return left == right
"""

FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.IGNORECASE | re.DOTALL)


def extract(answer):
    match = FENCE.search(answer)
    return (match.group(1) if match else answer).strip()


def program(row, answer, reference_only):
    code = row["reference"] if reference_only else extract(answer)
    if not code:
        code = "pass"
    body = RUNNER + f"""
candidate = {{}}
exec(compile({code!r}, '<candidate>', 'exec'), candidate)
fn = candidate[{row['entry_point']!r}]
"""
    if row["source"] == "original":
        checks = "\n".join(row["checks"])
        body += (
            "def strict_equal(left, right, atol):\n    return equal(left, right, atol, strict=True)\n"
            f"exec({checks!r}, {{'equal': strict_equal, 'date': date, 'datetime': datetime, {row['entry_point']!r}: fn}})\n"
        )
    elif row["source"] == MBPPPLUS_SOURCE:
        body += mbppplus_test_program(row)
    else:
        body += f"""
reference = {{}}
exec(compile({row['reference']!r}, '<reference>', 'exec'), reference)
expected_fn = reference[{row['entry_point']!r}]
for index, args in enumerate({row['cases']!r}, 1):
    expected = expected_fn(*copy.deepcopy(args))
    got = fn(*copy.deepcopy(args))
    assert equal(got, expected, {row['atol']!r}), f'case {{index}}: expected {{repr(expected)[:150]}}, got {{repr(got)[:150]}}'
"""
    return body + "\nprint('PYROTON_CHECKS_PASSED')\n"


def worker():
    from wasi_python import run

    def execute(source, deadline):
        result = run(["-c", source], deadline_s=deadline, cache_dir=False,
                     memory_bytes=256 * 1024**2, output_bytes=16384)
        stderr = result.stderr.decode(errors="replace").strip().splitlines()
        passed = (result.exit_code == 0 and not result.trapped
                  and b"PYROTON_CHECKS_PASSED" in result.stdout.splitlines())
        return {"passed": passed, "error": str(result.trapped) if result.trapped else stderr[-1] if stderr else None}

    ready = execute("print('PYROTON_CHECKS_PASSED')", 15)
    if not ready["passed"]:
        raise RuntimeError(f"Cannot initialize WASI: {ready['error']}")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        print(json.dumps(execute(command["code"], command["deadline"])), flush=True)


class ScoreProcess:
    def __init__(self, worker_script=None):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(worker_script or __file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            encoding="utf-8", bufsize=1,
        )
        self.responses = queue.Queue()

        def receive():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        threading.Thread(target=receive, daemon=True).start()
        try:
            if not self.receive(180).get("ready"):
                raise RuntimeError("The WASI worker sent an unexpected startup response.")
        except Exception:
            self.close()
            raise

    def receive(self, seconds):
        response = self.responses.get(timeout=seconds)
        if response is None:
            raise RuntimeError("The WASI worker exited before returning a result.")
        return json.loads(response)

    def check(self, code, deadline):
        self.process.stdin.write(json.dumps({"code": code, "deadline": deadline}) + "\n")
        self.process.stdin.flush()
        return self.receive(deadline + 15)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline", type=int, default=8)
    args = parser.parse_args()
    if args.worker:
        return worker()
    if not args.dataset or not args.output:
        parser.error("--dataset and --output are required")
    data = args.dataset.read_bytes()
    tasks = json.loads(data)
    answers = {}
    generation_hash = None
    if args.generations:
        generation_data = args.generations.read_bytes()
        generation_hash = hashlib.sha256(generation_data).hexdigest()
        answers = {row["id"]: row for row in map(json.loads, generation_data.decode("utf-8").splitlines())}
        if set(answers) != {row["id"] for row in tasks}:
            raise ValueError("Generation IDs do not match the evaluation set")
        if any(row["configuration"]["dataset_sha256"] != hashlib.sha256(data).hexdigest() for row in answers.values()):
            raise ValueError("The generations were produced from a different evaluation snapshot.")
    signature = {"dataset": hashlib.sha256(data).hexdigest(), "generations": generation_hash,
                 "deadline": args.deadline, "checker": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if any(row["source"] == MBPPPLUS_SOURCE for row in tasks):
        signature["mbppplus_checker"] = hashlib.sha256(
            Path(__file__).with_name("mbppplus_cases.py").read_bytes()
        ).hexdigest()
    progress = args.output.with_suffix(".progress.jsonl")
    completed = {}
    records = read_jsonl_repair(progress) if progress.exists() else []
    if records:
        if records[0] != signature:
            raise ValueError("Existing scoring progress belongs to a different run.")
        completed = {record["id"]: record for record in records[1:]}
        if len(completed) != len(records) - 1 or not completed.keys() <= {row["id"] for row in tasks}:
            raise ValueError("Existing scoring progress contains duplicate or unknown task IDs.")
    else:
        progress.write_text(json.dumps(signature) + "\n", encoding="utf-8")
    results = []
    if len(completed) == len(tasks):
        results = [completed[row["id"]] for row in tasks]
        write_json_atomic(args.output, results)
        print(f"{sum(row['passed'] for row in results)}/{len(results)} passed (saved judgments)", flush=True)
        return
    scorer = ScoreProcess()
    try:
        with progress.open("a", encoding="utf-8") as journal:
            for row in tasks:
                if row["id"] in completed:
                    results.append(completed[row["id"]])
                    continue
                candidate = answers.get(row["id"])
                try:
                    result = scorer.check(program(row, candidate["answer"] if candidate else "", not args.generations),
                                          args.deadline)
                except queue.Empty:
                    raise TimeoutError(f"Scoring subprocess timed out on {row['id']}; that result was not accepted.")
                suite_cases = (
                    len(row["base_input"]) + len(row["plus_input"])
                    if row["source"] == MBPPPLUS_SOURCE else len(row.get("checks", row.get("cases", [])))
                )
                record = {"id": row["id"], "source": row["source"], "suite_cases": suite_cases, **result}
                results.append(record)
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                if not result["passed"]:
                    print(row["id"], result["error"], flush=True)
    finally:
        scorer.close()
    write_json_atomic(args.output, results)
    print(f"{sum(row['passed'] for row in results)}/{len(results)} passed", flush=True)


if __name__ == "__main__":
    main()
