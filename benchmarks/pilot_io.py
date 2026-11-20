"""Small persistence helpers for interruption-safe benchmark journals."""

import json
from pathlib import Path


def read_jsonl_repair(path):
    """Keep valid records and remove only an unterminated, malformed last line."""
    rows = []
    with Path(path).open("rb+") as stream:
        while line := stream.readline():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                stream.seek(-len(line), 1)
                stream.truncate()
                break
            rows.append(row)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                break
    return rows


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
