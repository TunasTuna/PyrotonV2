"""Run the fixed eight-sample Pyroton pilot on a local llama.cpp server."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.error
from pathlib import Path

from generate_holdout import SYSTEM, digest, request
from score_holdout import extract


STYLES = ("mbpp", "humaneval")
PER_STYLE = 4
TEMPERATURE = 0.8
MAX_TOKENS = 512
TOP_P = 0.95
TOP_K = 40
SEED = "pyroton-step2-pilot-v1"


def identity(record):
    return record["problem_id"], record["style"], record["sample_index"]


def sample_seed(problem_id, style, index):
    value = f"{SEED}|{problem_id}|{style}|{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFF_FFFF


def chat_prompt(messages):
    return "".join(
        f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
    ) + "<|im_start|>assistant\n"


def jobs_for(tasks):
    ids = set()
    for row in tasks:
        problem = row["id"]
        if problem in ids:
            raise ValueError(f"Duplicate verified problem ID: {problem}")
        ids.add(problem)
        if not row.get("test_suite_sha256"):
            raise ValueError(f"Problem has no verified test suite: {problem}")
        for style in STYLES:
            for index in range(PER_STYLE):
                yield {
                    "problem_id": problem, "style": style, "sample_index": index,
                    "seed": sample_seed(problem, style, index),
                    "prompt": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": row["prompts"][style]}],
                    "test_suite_sha256": row["test_suite_sha256"],
                }


def read_journal(path, expected, config):
    completed = {}
    if path.exists():
        with path.open("rb+") as stream:
            while line := stream.readline():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if line.endswith(b"\n"):
                        raise
                    stream.seek(-len(line), 1)
                    stream.truncate()
                    break
                key = identity(record)
                if not line.endswith(b"\n"):
                    stream.write(b"\n")
                if (key in completed or key not in expected or record.get("configuration") != config
                        or any(record.get(field) != value for field, value in expected[key].items())):
                    raise ValueError(f"Duplicate or incompatible sample in {path}: {key}")
                completed[key] = record
    return completed


def healthy_server(url, process, log):
    end = time.monotonic() + 120
    while time.monotonic() < end and process.poll() is None:
        try:
            if request(url + "/health", timeout=3).get("status") == "ok":
                return
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"llama.cpp server did not become ready. See {log}.")


def make_sample(url, job, config):
    request_body = {
        "prompt": chat_prompt(job["prompt"]),
        "temperature": TEMPERATURE, "seed": job["seed"],
        "repeat_penalty": 1, "n_predict": MAX_TOKENS,
        "top_p": TOP_P, "top_k": TOP_K, "min_p": 0.0,
        "cache_prompt": False, "stream": False,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    for attempt in range(3):
        try:
            result = request(url + "/completion", request_body, timeout=300)
            answer = result["content"]
            code = extract(answer)
            solution_tokens = len(request(url + "/tokenize", {"content": code, "add_special": False})["tokens"])
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    predicted = result["timings"]["predicted_n"]
    return {
        **job, "answer": answer, "code": code, "tokens": predicted, "solution_tokens": solution_tokens,
        "stop_reason": result.get("stop_type") or ("limit" if predicted >= MAX_TOKENS else "completed"),
        "decode_seconds": result["timings"]["predicted_ms"] / 1000,
        "configuration": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "verified-corpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--limit-problems", type=int, help="Smoke test only. Writes to a separate journal.")
    args = parser.parse_args()
    if args.slots < 1 or args.threads < 1 or args.gpu_layers < 0:
        parser.error("Use positive slot/thread counts and a nonnegative GPU layer count.")
    if args.limit_problems is not None and args.limit_problems <= 0:
        parser.error("--limit-problems must be positive.")
    executable = args.server if args.server.is_file() else shutil.which(str(args.server))
    if not executable:
        parser.error("Cannot find the pinned llama-server executable.")
    server = Path(executable).resolve()
    tasks = json.loads(args.verified_corpus.read_bytes())
    expected = {identity(job): job for job in jobs_for(tasks[:args.limit_problems])}
    if not expected:
        parser.error("The verified corpus is empty.")
    config = {
        "protocol": SEED, "model_sha256": digest(args.model), "server_sha256": digest(server),
        "verified_corpus_sha256": digest(args.verified_corpus), "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "min_p": 0.0, "per_style": PER_STYLE,
        "slots": args.slots, "threads": args.threads, "gpu_layers": args.gpu_layers,
        "limit_problems": args.limit_problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    finished = read_journal(args.output, expected, config)
    remaining = [value for key, value in expected.items() if key not in finished]
    print(f"{len(finished)}/{len(expected)} samples already saved.", flush=True)
    if not remaining:
        return

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        str(server), "--host", "127.0.0.1", "--port", str(port), "-m", str(args.model.resolve()),
        "-c", str(2048 * args.slots), "-t", str(args.threads), "-tb", str(args.threads),
        "-np", str(args.slots), "-ngl", str(args.gpu_layers), "--no-webui",
    ]
    log_path = args.output.with_suffix(".server.log")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            healthy_server(url, process, log_path)
            if args.gpu_layers > 0:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if "CUDA0" not in log_text or "offloaded 0/" in log_text:
                    raise RuntimeError(f"GPU offload was requested but could not be verified in {log_path}.")
            completed_count = len(finished)
            with args.output.open("a", encoding="utf-8") as journal:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as pool:
                    iterator = iter(remaining)
                    pending = {pool.submit(make_sample, url, job, config)
                               for _, job in zip(range(args.slots), iterator)}
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            result = future.result()
                            journal.write(json.dumps(result, ensure_ascii=True) + "\n")
                            journal.flush()
                            completed_count += 1
                            if completed_count % 32 == 0 or completed_count == len(expected):
                                print(f"{completed_count}/{len(expected)} samples saved.", flush=True)
                            if (next_job := next(iterator, None)) is not None:
                                pending.add(pool.submit(make_sample, url, next_job, config))
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
