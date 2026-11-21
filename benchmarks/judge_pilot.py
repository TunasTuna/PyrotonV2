"""Check the submitted pilot samples using the original tests and frozen edge oracles."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_holdout import digest
from pilot_verifier import PilotSandbox, verify_candidate
from sample_pilot import (MAX_TOKENS, PER_STYLE, TEMPERATURE, ZERO_PASS_PER_STYLE,
                          identity, jobs_for, protocol, read_journal)
from score_holdout import extract


def load_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    indexed = {identity(row): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate sample IDs in {path}.")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verified-corpus", "samples", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline", type=int, default=8)
    parser.add_argument("--draws-per-style", type=int, default=PER_STYLE, choices=(PER_STYLE, ZERO_PASS_PER_STYLE))
    parser.add_argument("--allow-partial", action="store_true", help="Smoke testing only.")
    args = parser.parse_args()
    if args.deadline < 1:
        parser.error("--deadline must be positive")
    corpus = json.loads(args.verified_corpus.read_bytes())
    rows = {row["id"]: row for row in corpus}
    expected = {identity(job): job for job in jobs_for(corpus, args.draws_per_style)}
    samples = load_records(args.samples)
    if not samples:
        parser.error("The sample journal is empty.")
    if not args.allow_partial and set(samples) != set(expected):
        raise ValueError(f"Expected the full {len(expected)} samples, found {len(samples)}.")
    if not samples.keys() <= expected.keys():
        raise ValueError("Unknown sample IDs.")
    config = next(iter(samples.values()))["configuration"]
    if (config.get("verified_corpus_sha256") != digest(args.verified_corpus)
            or config.get("max_tokens") != MAX_TOKENS or config.get("temperature") != TEMPERATURE
            or config.get("per_style") != args.draws_per_style or config.get("protocol") != protocol(args.draws_per_style)):
        raise ValueError("Generation provenance did not match this pilot.")
    for key, sample in samples.items():
        if (sample["configuration"] != config
                or any(sample.get(k) != v for k, v in expected[key].items())
                or sample["code"] != extract(sample["answer"])
                or not 0 <= sample["tokens"] <= MAX_TOKENS
                or sample["solution_tokens"] < 0):
            raise ValueError(f"Invalid sample content: {key}")

    implementation = Path(__file__).with_name("pilot_verifier.py")
    verifier_config = {
        "deadline": args.deadline, "verifier_sha256": digest(implementation),
        "corpus_sha256": digest(args.verified_corpus),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    finished = {}
    if args.output.is_file() and args.output.stat().st_size:
        config = next(iter(samples.values()))["configuration"]
        finished = read_journal(args.output, expected, config)
    for key, record in finished.items():
        sample = samples.get(key)
        if (sample is None or any(record.get(field) != value for field, value in sample.items())
                or record["verifier_configuration"] != verifier_config):
            raise ValueError(f"Existing result came from a different verifier or sample: {key}")

    pending = [key for key in samples if key not in finished]
    if not pending:
        print(f"{len(finished)}/{len(samples)} judgments already saved.")
        return
    worker = PilotSandbox()
    try:
        with args.output.open("a", encoding="utf-8") as journal:
            for index, key in enumerate(pending, len(finished) + 1):
                record = samples[key]
                result = verify_candidate(rows[key[0]], record["code"], worker, args.deadline)
                error = (result.get("error") or "").lower()
                if (record["tokens"] == MAX_TOKENS and result.get("failure_category") == "format"
                        and any(message in error for message in ("never closed", "unterminated", "unexpected eof", "indented block"))):
                    result["failure_category"] = "token_cap"
                judged = {**record, **result, "verifier_configuration": verifier_config,
                          "sample_sha256": hashlib.sha256(record["answer"].encode()).hexdigest()}
                journal.write(json.dumps(judged, ensure_ascii=True) + "\n")
                journal.flush()
                if index % 32 == 0 or index == len(samples):
                    print(f"{index}/{len(samples)} judgments saved.", flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
