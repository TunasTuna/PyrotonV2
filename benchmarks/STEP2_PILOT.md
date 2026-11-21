# Step 2 pilot only

This pilot keeps stock Qwen2.5-Coder-1.5B-Instruct Q4_K_M as the generator.
It takes four samples per problem for each of two equivalent prompt styles,
using temperature 0.8, top-p 0.95, top-k 40, min-p 0 and a cap of 512 new
tokens. Seeds are stable by problem, style and draw. The sampling journal
records raw generation length, extracted Python length and stop reason.
No training or quantization step is part of this pilot.

## Run on Colab

Open the numbered `notebooks/step2_pilot.ipynb` in a T4 runtime. Upload the
companion `pyroton-v2-step2-pilot-inputs.zip` in cell 2. Complete cells 1?9.
After a disconnect, use the same Drive account and run directory. Sampling,
preflight and grading resume from validated journals. The GPU notebook uses
the pinned llama.cpp b12521 CUDA build, the same stock model hash as Step 1,
and pinned WASI packages. The runtime can require roughly 2 GB of Google
Drive cache for the model and backend archives, in addition to result files.
It does not install or change torch or torchao.

When cell 9 completes, the run's `results` directory contains the 512-token
baseline report, verifier report, sample and grading journals, per-problem
pilot report, a refreshed overlap audit, `sft.jsonl`, and `orpo.jsonl`. A full
report requires eight judgments for every verified problem. Filenames in a
smoke-run directory remain separate and progress mode emits only a report.
Stop at data export.

The 60-task dev result is a subset of the same generations used for the
full 160-task result. Colab runs a 256-token CUDA control on both suites
to compare caps on the same backend. The historical Windows CPU baseline
is also recorded for the primary suite. The baseline file preserves the
exact settings for all runs.

For a CPU comparison that reuses the primary Step 1 results, pass the
MBPP+ control with `--matched256-mbppplus-generations` and
`--matched256-mbppplus-scores`. The reporter can pair either suite
independently and checks that only the output cap changed in paired runs.

## Verification and selection

The corpus builder loads original MBPP training IDs 601?974 and validation
IDs 511?600, then excludes known matches and close variants of the old 40,
the full dev/final union and all official MBPP+ test IDs. It also excludes
previous near matches found by the evaluation audits. The selected subset
may be smaller than the original estimate. Evaluation records and outputs
are only used by the audit and baseline scripts.

The verifier runs the reference in a subprocess sandbox against the supplied
assertions and additional candidate inputs derived from the source tests.
It retains reference-supported edge cases and compares candidate output or
a specified exception on the frozen cases. Out-of-contract behavior of an
old reference does not become a new requirement. Wrong-type checks require
evidence of an explicit validation contract. The candidate namespace sees
no imports from the tests or the reference. The entry point and signature
are checked before the original and edge tests, and execution is timed.

Final selection first requires an execution pass and a refreshed overlap
audit. It prefers avoiding a copied prompt docstring, then shorter complete
Python. ORPO pairs use the same exact prompt and style, with chosen passing
and rejected failing a behavioral check. Primary rejected solutions retain
their callable signature, so the comparison teaches behavior. Model output
and candidate duplicate code overlap are checked again before writing the
two data files.

## Sources

- [Google Research MBPP and original split](https://github.com/google-research/google-research/tree/master/mbpp)
- [EvalPlus MBPP+ release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0)
- [Qwen 1.5B GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF)
- [llama.cpp b12521](https://github.com/ggml-org/llama.cpp/releases/tag/b12521)
