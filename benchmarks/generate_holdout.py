"""Generate first attempts with llama.cpp using identical CPU and decoding settings."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from pilot_io import read_jsonl_repair

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def request(url, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.gpu_layers < 0 or args.slots < 1 or args.threads < 1:
        parser.error("Use positive slot/thread counts and a nonnegative GPU layer count.")
    server_path = args.server if args.server.is_file() else shutil.which(str(args.server))
    if not server_path:
        parser.error(f"llama-server executable not found: {args.server}")
    server = Path(server_path).resolve()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    configuration = {
        "base_sha256": digest(args.model),
        "adapter_sha256": digest(args.adapter) if args.adapter else None,
        "dataset_sha256": digest(args.dataset),
        "server_sha256": digest(server), "new_tokens": args.tokens,
        "slots": args.slots, "threads": args.threads, "temperature": 0,
    }
    if args.gpu_layers:
        configuration["gpu_layers"] = args.gpu_layers
    complete = set()
    if args.output.exists():
        for row in read_jsonl_repair(args.output):
            if row["configuration"] != configuration:
                raise ValueError("Existing generations belong to a different run")
            if row["id"] in complete:
                raise ValueError("Existing generations contain a duplicate task ID.")
            complete.add(row["id"])
    if not complete <= {row["id"] for row in rows}:
        raise ValueError("Existing generations contain unknown task IDs.")
    rows = [row for row in rows[:args.limit] if row["id"] not in complete]
    if not rows:
        return

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    adapter = ["--lora", str(args.adapter.resolve())] if args.adapter else []
    command = [
        str(server), "--host", "127.0.0.1", "--port", str(port),
        "-m", str(args.model.resolve()), "-c", str(2048 * args.slots),
        "-t", str(args.threads), "-tb", str(args.threads), "-np", str(args.slots),
        "-ngl", str(args.gpu_layers), "--no-webui", *adapter,
    ]

    def generate(row):
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n")
        result = request(url + "/completion", {
            "prompt": prompt, "temperature": 0, "repeat_penalty": 1,
            "seed": 42, "n_predict": args.tokens, "cache_prompt": False,
            "stream": False, "stop": ["<|im_end|>", "<|endoftext|>"],
        })
        timing = result["timings"]
        return {"id": row["id"], "answer": result["content"], "tokens": timing["predicted_n"],
                "decode_seconds": timing["predicted_ms"] / 1000, "configuration": configuration}

    log_path = args.output.with_suffix(".server.log")
    with log_path.open("w", encoding="utf-8") as log:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"llama.cpp failed to start. See {log_path}.")
                try:
                    if request(url + "/health", timeout=3).get("status") == "ok":
                        break
                except OSError:
                    pass
                time.sleep(0.3)
            if args.gpu_layers:
                log_contents = log_path.read_text(encoding="utf-8", errors="replace")
                if "CUDA0" not in log_contents or "offloaded 0/" in log_contents:
                    raise RuntimeError(f"GPU offload could not be verified. See {log_path}.")
            with args.output.open("a", encoding="utf-8") as result_file:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as pool:
                    pending = [pool.submit(generate, row) for row in rows]
                    for future in concurrent.futures.as_completed(pending):
                        record = future.result()
                        result_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                        result_file.flush()
                        print(record["id"], record["tokens"], flush=True)
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
