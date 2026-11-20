"""Build the pinned Pyroton V2 Step 1 holdout and write an overlap audit."""

import argparse
import ast
import difflib
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from validation_tasks import TASKS


EXCLUSIONS = {
    3: "Related to old ledger task; conservative exclusion.",
    13: "Old gcd task.",
    31: "Old is_prime task.",
    32: "Multiple valid polynomial roots; requires a special oracle.",
    34: "Related to old dedupe task; conservative exclusion.",
    48: "Old palindrome task without normalization.",
    56: "Old balanced_brackets with a reduced alphabet.",
    58: "Old intersection task.",
    61: "Old balanced_brackets with a reduced alphabet.",
    76: "Related to old is_power_two task; conservative exclusion.",
    84: "Old digit_sum plus a binary representation.",
    90: "Related to old second_largest task; conservative exclusion.",
    119: "Related to old balanced_brackets task; conservative exclusion.",
    125: "Related to old collapse_ws task; conservative exclusion.",
    132: "Related to old balanced_brackets task; conservative exclusion.",
    140: "Related to MBPP 631 and old collapse_ws; conservative exclusion.",
}
CUSTOM_EXCLUSIONS = {
    "replace_unique": "Related to pilot replace_once.",
    "run_window_counts": "Related to old moving_sum.",
    "clip_span": "Related to old merge_intervals and HumanEval/127.",
}


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class NormalizeCode(ast.NodeTransformer):
    """Normalize local identifiers and ignore docstrings/type annotations."""
    def __init__(self):
        self.names = {}

    def name(self, name):
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_FunctionDef(self, node):
        node.name = self.name(node.name)
        node.returns = None
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return self.generic_visit(node)

    def visit_arg(self, node):
        return ast.arg(arg=self.name(node.arg))

    def visit_Name(self, node):
        builtins = {"len", "min", "max", "sum", "range", "sorted", "list", "set",
                    "dict", "int", "str", "bool", "type", "isinstance", "enumerate", "zip"}
        if node.id not in builtins:
            node.id = self.name(node.id)
        return node


def code_key(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    return ast.dump(NormalizeCode().visit(tree), include_attributes=False)


def load_original_examples(notebook):
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        source = "".join(cell["source"])
        if source.startswith("original_examples = "):
            statement = ast.parse(source).body[0]
            return ast.literal_eval(statement.value)
    raise ValueError("Original examples cell not found")


def custom_rows():
    for name, prompt, reference, cases in TASKS:
        checks = []
        for args, expected in cases:
            call = args if isinstance(args, str) else f"{name}(*{args!r})"
            if expected is ValueError:
                checks.append(
                    f"try:\n    {call}\nexcept ValueError:\n    pass\n"
                    "else:\n    raise AssertionError('Expected ValueError')"
                )
            else:
                checks.append(f"assert equal({call}, {expected!r}, 0), {call!r}")
        if name == "multimap_remove":
            checks.append(
                "before={'x':[1,2],'y':[0]}\nafter=multimap_remove(before,'x',1)\n"
                "after['y'].append(3)\nassert before=={'x':[1,2],'y':[0]}"
            )
        if name == "replace_unique":
            checks.append("items=[0,1]\nassert replace_unique(items,1,2)==[0,2] and items==[0,1]")
        yield {
            "id": f"PyrotonStrict/{name}", "entry_point": name,
            "description": prompt,
            "prompt": f"Write a Python function named {name}. {prompt} Return only Python code.",
            "reference": reference, "checks": checks, "source": "original",
            "excluded": CUSTOM_EXCLUSIONS.get(name),
        }


def public_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            number = int(source["task_id"].split("/")[-1])
            prompt = source["prompt"]
            entry = source["entry_point"]
            reference = prompt + source["canonical_solution"]
            implementation = next(node for node in ast.parse(reference).body
                                  if isinstance(node, ast.FunctionDef) and node.name == entry)
            extra = source["plus_input"]
            selected = sorted(random.Random(4100 + number).sample(range(len(extra)), min(64, len(extra))))
            yield {
                "id": source["task_id"], "entry_point": entry, "description": ast.get_docstring(implementation),
                "prompt": "Implement the function specified below. Return only complete Python code, including imports.\n\n" + prompt,
                "reference": reference, "cases": source["base_input"] + [extra[index] for index in selected],
                "atol": source["atol"], "source": "HumanEval+ v0.1.10", "excluded": EXCLUSIONS.get(number),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("humaneval", "mbpp", "training-metadata", "notebook", "old-report", "output", "audit"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_metadata.read_text(encoding="utf-8"))
    dataset = training["dataset"]
    raw_mbpp = args.mbpp.read_bytes()
    if hashlib.sha256(raw_mbpp).hexdigest() != dataset["source_sha256"]:
        raise ValueError("MBPP source does not match the exported training metadata")
    selected_ids = set(dataset["task_ids"])
    known = [
        (f"MBPP/{row['task_id']}", row["prompt"], row["code"])
        for row in json.loads(raw_mbpp) if row["task_id"] in selected_ids
    ]
    if len(known) != len(selected_ids):
        raise ValueError("Missing training examples")
    pilot = load_original_examples(args.notebook)
    known.extend((f"Pilot/{index}", row[0], row[1]) for index, row in enumerate(pilot, 1))
    assert len(known) == training["examples"]
    old = json.loads(args.old_report.read_text(encoding="utf-8"))[:40]
    known.extend((f"Old40/{row['name']}", row["prompt"], row["answer"] if row["passed"] else "") for row in old)
    indexed = [(label, words(text), code_key(code)) for label, text, code in known]

    audit, holdout = [], []
    for row in [*public_rows(args.humaneval), *custom_rows()]:
        text = words(row["description"])
        key = code_key(row["reference"])
        overlaps = []
        for label, old_text, old_code in indexed:
            exact_code = bool(key and old_code and key == old_code)
            text_score = difflib.SequenceMatcher(None, text, old_text, autojunk=False).ratio()
            code_score = difflib.SequenceMatcher(None, key.split(", "), old_code.split(", "), autojunk=False).ratio() if key and old_code else 0
            if exact_code or text_score >= 0.55 or code_score >= 0.85:
                overlaps.append({"against": label, "text_similarity": round(text_score, 3),
                                 "code_similarity": round(code_score, 3), "same_normalized_code": exact_code})
        reason = row.pop("excluded")
        if not reason and any(hit["same_normalized_code"] or hit["text_similarity"] >= 0.70 for hit in overlaps):
            reason = "Conservative automated exclusion for a strong text or exact normalized-code match."
        audit.append({"id": row["id"], "entry_point": row["entry_point"], "exclusion": reason, "flags": overlaps})
        if not reason:
            holdout.append(row)
    if len(holdout) < 150:
        raise ValueError(f"Only {len(holdout)} tasks survived audit")
    args.output.write_text(json.dumps(holdout, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    report = {"known_training": len(known)-len(old), "old_tasks": len(old),
              "human_eval_source_sha256": hashlib.file_digest(args.humaneval.open("rb"), "sha256").hexdigest(),
              "held_out": len(holdout), "audit": audit}
    args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(holdout)} tasks kept; {len(audit)-len(holdout)} excluded")


if __name__ == "__main__":
    main()
