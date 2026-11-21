# Step 2b and Step 3 pilot

The Step 2b verifier reuses original stage evidence and the sealed overlap
audit. It reruns edge failures, unresolved sandbox failures, and previous
passes affected by a corrected expected result. The report separates compile
or load failures from signature mismatches and records unresolved sandbox
failures separately. The original checks include the published challenge
assertions. An edge stays in the verifier only if its domain and expected
behavior have an explicit contract review. Tasks without retained reviewed
edges use their original assertions. The report preserves the original
verdicts and includes a reproducible sample of 30 original edge-only failures.

The rebuilt ORPO pairs use the same prompt for both answers and require the
longer completion to have at most 125% of the shorter completion's token
count. The exporter preserves the pilot's holdout exclusions. The pilot
trains directly from the stock Qwen2.5-Coder-1.5B-Instruct base with LoRA
rank 16, learning rate 0.00004, beta 0.1, and two epochs.

## Colab run

Open the numbered `notebooks/step3_pilot.ipynb` in a T4 runtime. Complete
Drive authorization in cell 1 and upload the `pyroton-v2-step3-pilot-inputs.zip`
bundle in cell 2. The bundle contains the completed Step 2b data-quality and
CPU stock-repeat reports. The notebook repeats both stock runs on T4 for
paired checkpoint comparisons and decode-speed measurements on the same
hardware. It installs pinned libraries without changing the existing torch
or torchao versions.

After a disconnect, reconnect a T4 to the same Drive account and rerun the
cells in order. The notebook resumes from the latest complete epoch and
reuses validated generation journals and model exports. Each completed epoch
is merged into the pinned stock Hugging Face revision, converted by llama.cpp
`b12521`, and quantized to Q4_K_M. Official stock uses the same base and
quantization class. Its original conversion predates the pilot converter.

Evaluation uses dev and MBPP+. The 32-draw diagnostic samples the original
167 zero-pass problems using stock on the T4, with 16 draws per prompt style.
The diagnostic reports both the full cohort and the subset still unsolved
after verifier correction. These draws stay outside training. Drive holds
both epoch checkpoints, the merged GGUF exports, and the report archive.

- [Stock Qwen base](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [TRL ORPO documentation](https://huggingface.co/docs/trl/en/orpo_trainer)
- [Pinned llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12521)
