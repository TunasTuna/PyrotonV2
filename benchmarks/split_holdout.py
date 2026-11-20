"""Freeze disjoint, source-stratified dev and final splits by task identity."""

import argparse
import hashlib
import json
from pathlib import Path


SEED = "pyroton-v2-split-v1"
DEV_SIZE = {"HumanEval+ v0.1.10": 54, "original": 6}

# Keep close families together so threshold tuning on dev does not see their
# near neighbors across the partition. Group membership uses the task specs.
FAMILIES = (
    (25, 59),        # Prime factorization
    (27, 161),       # Case inversion
    (33, 37),        # Sort values at selected strides
    (45, 71, 157),   # Triangle geometry
    (46, 55, 63),    # Fibonacci variants
    (85, 121),       # Sum values at alternating indices by parity
    (88, 116),       # sort_array variants
    (133, 142),      # sum_squares variants
)


def rank(ids):
    return hashlib.sha256((SEED + "|" + "|".join(ids)).encode("utf-8")).digest()


def split(rows):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("The input contains duplicate task IDs.")
    quotas = {"HumanEval+ v0.1.10": 145, "original": 15}
    found = {source: sum(row["source"] == source for row in rows) for source in quotas}
    if len(rows) != 160 or found != quotas:
        raise ValueError("Expected the frozen 160-task Step 1 holdout.")
    groups = [tuple(f"HumanEval/{number}" for number in family) for family in FAMILIES]
    grouped = {task for group in groups for task in group}
    if not grouped <= set(ids) or len(grouped) != sum(map(len, groups)):
        raise ValueError("The fixed task families do not match this holdout.")
    groups.extend((task,) for task in ids if task not in grouped)
    sources = {row["id"]: row["source"] for row in rows}
    remaining = DEV_SIZE.copy()
    dev = set()
    for group in sorted(groups, key=rank):
        source = sources[group[0]]
        if any(sources[task] != source for task in group):
            raise ValueError("A task family crosses two data sources.")
        if len(group) <= remaining[source]:
            dev.update(group)
            remaining[source] -= len(group)
    if any(remaining.values()):
        raise ValueError("Could not satisfy the dev quotas.")
    return ([row for row in rows if row["id"] in dev],
            [row for row in rows if row["id"] not in dev])


def write(path, value):
    text = json.dumps(value, ensure_ascii=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    dev, final = split(json.loads(source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, partition in (("dev.json", dev), ("final-test.json", final)):
        files[filename] = {
            "sha256": write(args.output_dir / filename, partition),
            "task_ids": [row["id"] for row in partition],
            "by_source": {name: sum(row["source"] == name for row in partition) for name in DEV_SIZE},
        }
    write(args.output_dir / "split-manifest.json", {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "seed": SEED,
        "assigned_from": "Task IDs, source strata, and prompt-derived family groups. No scores were used for assignment.",
        "previous_exposure": "Both baselines and the failure audit have already inspected the unsplit parent suite.",
        "reserved_for": {"dev.json": "model selection, never training",
                         "final-test.json": "final evaluation, never training or model selection"},
        "grouped_families": FAMILIES,
        "files": files,
    })
    print(f"Frozen dev/final split: {len(dev)}/{len(final)}")


if __name__ == "__main__":
    main()
