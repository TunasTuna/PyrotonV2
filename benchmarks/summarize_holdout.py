"""Summarize complete, like-for-like Step 1 results and paired changes."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generation_index(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    index = {row["id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate generation IDs in {path}")
    return index


def score_index(path):
    rows = read_json(path)
    index = {row["id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate score IDs in {path}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "audit", "reference-scores", "stock-generations", "stock-scores",
                 "adapter-generations", "adapter-scores", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    dataset = read_json(args.dataset)
    audit = read_json(args.audit)
    refs = score_index(args.reference_scores)
    stock = generation_index(args.stock_generations)
    adapter = generation_index(args.adapter_generations)
    stock_scores = score_index(args.stock_scores)
    adapter_scores = score_index(args.adapter_scores)
    ids = {row["id"] for row in dataset}
    if len(ids) != len(dataset) or any(set(group) != ids for group in (refs, stock, adapter, stock_scores, adapter_scores)):
        raise ValueError("Evaluation, reference, generation and score IDs must agree exactly.")

    config_stock = next(iter(stock.values()))["configuration"]
    config_adapter = next(iter(adapter.values()))["configuration"]
    if config_stock["adapter_sha256"] is not None or not config_adapter["adapter_sha256"]:
        raise ValueError("Stock must run without an adapter and Pyroton must run with an adapter.")
    if {k: v for k, v in config_stock.items() if k != "adapter_sha256"} != {k: v for k, v in config_adapter.items() if k != "adapter_sha256"}:
        raise ValueError("Benchmark generation settings differ between the two models.")
    if config_stock["dataset_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
        raise ValueError("Generation dataset mismatch.")
    if any(row["configuration"] != config_stock for row in stock.values()) or any(row["configuration"] != config_adapter for row in adapter.values()):
        raise ValueError("Mixed generation configurations.")

    excluded = {i: row["error"] for i, row in refs.items() if not row["passed"]}
    accepted = [row for row in dataset if row["id"] not in excluded]
    if len(accepted) < 150:
        raise ValueError(f"Only {len(accepted)} tasks passed reference preflight.")
    results = []
    for row in accepted:
        i = row["id"]
        results.append({
            "id": i, "source": row["source"], "prompt": row["prompt"],
            "cases": refs[i]["suite_cases"],
            "stock": {**stock_scores[i], "answer": stock[i]["answer"], "tokens": stock[i]["tokens"]},
            "adapter": {**adapter_scores[i], "answer": adapter[i]["answer"], "tokens": adapter[i]["tokens"]},
        })
    recoveries = [row["id"] for row in results if row["adapter"]["passed"] and not row["stock"]["passed"]]
    regressions = [row["id"] for row in results if row["stock"]["passed"] and not row["adapter"]["passed"]]

    def totals(subset):
        return {
            "tasks": len(subset),
            "stock_pass": sum(row["stock"]["passed"] for row in subset),
            "adapter_pass": sum(row["adapter"]["passed"] for row in subset),
            "stock_median_tokens": median(row["stock"]["tokens"] for row in subset),
            "adapter_median_tokens": median(row["adapter"]["tokens"] for row in subset),
        }

    summary = {
        "all": totals(results),
        "by_source": {source: totals([row for row in results if row["source"] == source])
                      for source in dict.fromkeys(row["source"] for row in results)},
        "test_cases": sum(row["cases"] for row in results),
        "reference_preflight_exclusions": excluded,
        "adapter_recovers": recoveries, "adapter_regresses": regressions,
        "training_overlap_audit": audit,
        "stock_configuration": config_stock, "adapter_configuration": config_adapter,
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "all", "by_source", "test_cases", "reference_preflight_exclusions", "adapter_recovers", "adapter_regresses"
    )}, indent=2))


if __name__ == "__main__":
    main()
