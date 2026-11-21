"""Replay watchdog cases with the original and extra checks separated."""

import argparse
from collections import Counter
import json
from pathlib import Path

from judge_pilot import load_records
from pilot_io import write_json_atomic
from pilot_verifier import PilotSandbox, build_program, imported_names, verify_candidate
from report_pilot import write_jsonl
from sample_pilot import identity


def replay(task, code, worker, deadline):
    signature = worker.check(build_program("""
function, error = load(data["code"], data["entry"], data["signature"], data["imports"])
__pilot_emit(error or success())
""", code=code, entry=task["entry_point"], signature=task["reference_signature"],
        imports=imported_names(task["reference"])), deadline)
    if not signature["passed"]:
        stage = "before_signature" if signature["stage"] in ("compile", "runtime") else "signature"
        return stage, signature["error"]
    original = verify_candidate({**task, "generated_edges": []}, code, worker, deadline)
    if not original["passed"]:
        if original["stage"] in ("compile", "signature"):
            return "inconsistent_signature_replay", original["error"]
        return "original_tests", original["error"]
    if task["generated_edges"]:
        complete = verify_candidate(task, code, worker, deadline)
        if not complete["passed"]:
            if complete["stage"] in ("edge", "runtime"):
                return "extra_checks_enabled", complete["error"]
            return "inconsistent_stage_replay", complete["error"]
    return "watchdog_not_reproduced", None


def resolved_stages(stages, details):
    lookup = {(row["verifier"], identity(row)): row for row in details}
    normalized = {"extra_checks_enabled": "extra_edge_only", "before_signature": "before_signature",
                  "signature": "signature", "original_tests": "original_tests"}
    for key, row in stages.items():
        updated = dict(row)
        for label in ("original", "corrected"):
            source = row[f"{label}_category"]
            detail = lookup.get((label, key))
            if source == "sandbox_phase_unresolved" and detail is None:
                raise ValueError(f"Missing watchdog replay for {label} {key}.")
            updated[f"{label}_resolved_category"] = normalized.get(detail["replay_stage"], source) if detail else source
            if detail:
                updated[f"{label}_replay"] = {field: detail[field] for field in ("replay_stage", "replay_error")}
        yield updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-corpus", "corrected-corpus", "judgments", "stages", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resolved-stages", type=Path)
    args = parser.parse_args()
    corpora = {label: {row["id"]: row for row in json.loads(path.read_bytes())} for label, path in
               (("original", args.original_corpus), ("corrected", args.corrected_corpus))}
    stages, judgments = load_records(args.stages), load_records(args.judgments)
    if stages.keys() != judgments.keys():
        raise ValueError("The stage records and archived judgments must cover the same draws.")
    worker = PilotSandbox()
    details = []
    try:
        for label, corpus in corpora.items():
            for key, row in stages.items():
                if row[f"{label}_category"] != "sandbox_phase_unresolved":
                    continue
                judgment = judgments[key]
                result, error = replay(corpus[row["problem_id"]], judgment["code"], worker,
                                       judgment["verifier_configuration"]["deadline"])
                detail = {"verifier": label, "problem_id": key[0], "style": key[1], "sample_index": key[2],
                          "replay_stage": result, "replay_error": error}
                details.append(detail)
                print(json.dumps(detail), flush=True)
    finally:
        worker.close()
    write_json_atomic(args.output, {"observation": "Isolated watchdog replays supplement the original stage labels.",
                                    "counts": {label: dict(Counter(row["replay_stage"] for row in details
                                                                   if row["verifier"] == label)) for label in corpora},
                                    "samples": details})
    if args.resolved_stages:
        write_jsonl(args.resolved_stages, resolved_stages(stages, details))


if __name__ == "__main__":
    main()
