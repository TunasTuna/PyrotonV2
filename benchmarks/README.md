# Pyroton V2 Step 1 holdout

The holdout combines HumanEval+ v0.1.10 and original Python tasks with explicit
input validation. Every public task uses all base cases and up to 64 deterministically
selected additional cases. This is a Pyroton benchmark using EvalPlus data. Its
scores are specific to this test harness and sample of cases.

The builder compares normalized task descriptions and identifier-normalized Python
ASTs with the 82 MBPP training references and 12 original training examples from
the MBPP v1 export. It compares the same signals with the old 40-task prompts
and uses passing PrimeFix completions as code proxies where available. The old
benchmark has no canonical solutions. The audit also records manually excluded
semantic neighbors. Similarity checks are heuristic and cannot establish what
was in Qwen pretraining.

Keep the frozen evaluation JSON and its prompts, reference solutions, tests, and
outputs out of future SFT, rejection sampling and ORPO. Audit each new training
source before emitting training examples. Publish or share benchmark data as
evaluation data only.

## Inputs

- [HumanEval+ v0.1.10 dataset](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10), from [EvalPlus](https://github.com/evalplus/evalplus).
- [Pinned MBPP training source](https://raw.githubusercontent.com/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json), Google Research, Austin et al. (2021), CC BY 4.0.
- The MBPP v1 export's `run.json`, the MBPP training notebook, and the old PrimeFix benchmark report.
- The same Qwen2.5-Coder-1.5B-Instruct Q4_K_M base file for both models, with the Pyroton MBPP v1 LoRA converted to f16 GGUF for the adapted run.

## Run

Use Python 3.12 or later and llama.cpp b12521 or a compatible build. The scorer
requires `wasi-python==0.1.5` in an isolated environment. Its worker runs in a
subprocess, with a per-task WASI deadline, memory cap, and an outer watchdog.
The guest sees no host files or network. Executable reference code is checked
before accepting a task into the scored denominator.

```sh
python benchmarks/build_holdout.py --humaneval HumanEvalPlus.jsonl.gz --mbpp sanitized-mbpp.json --training-metadata run.json --notebook pyroton-1.5b-mbpp-training.ipynb --old-report primefix-benchmark.json --output holdout.json --audit overlap-audit.json
python benchmarks/generate_holdout.py --server llama-server --model qwen-1.5b-q4.gguf --dataset holdout.json --output stock.jsonl
python benchmarks/generate_holdout.py --server llama-server --model qwen-1.5b-q4.gguf --adapter mbpp-v1-adapter.gguf --dataset holdout.json --output adapter.jsonl
python benchmarks/score_holdout.py --dataset holdout.json --output reference-scores.json
python benchmarks/score_holdout.py --dataset holdout.json --generations stock.jsonl --output stock-scores.json
python benchmarks/score_holdout.py --dataset holdout.json --generations adapter.jsonl --output adapter-scores.json
python benchmarks/summarize_holdout.py --dataset holdout.json --audit overlap-audit.json --reference-scores reference-scores.json --stock-generations stock.jsonl --stock-scores stock-scores.json --adapter-generations adapter.jsonl --adapter-scores adapter-scores.json --output summary.json
```

Each model receives the same prompts and token cap with one greedy attempt per
task. Four parallel slots reduce wall time. Use single-request decode throughput
for later speed comparisons. A fence may be unwrapped, and any remaining code
must execute as returned. The evaluator uses tolerant numeric comparisons for
public floating-point tasks and checks return types on original tasks.
