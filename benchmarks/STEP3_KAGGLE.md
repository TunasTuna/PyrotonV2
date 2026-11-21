# Step 3 Kaggle pilot

The numbered `notebooks/step3_kaggle.ipynb` uses the private
`pyroton-v2-inputs` Kaggle dataset and runs one visible device in FP16.
It trains stock Qwen2.5-Coder-1.5B-Instruct on the 89 audited preference
pairs with LoRA 16/32, beta 0.1, learning rate 0.00004 and two epochs.
The Kaggle output contains completed epoch checkpoints, adapter ZIPs and
per-step ORPO loss and reward margins. The trainer flags non-finite
metrics or missing or non-increasing mean reward margins.

The full checkpoint directory can resume a fresh Kaggle session once
restored as described in cell 0. The two small adapter ZIPs contain the
weights and epoch provenance for Windows inference. No Hub upload runs.
When using an interactive session, use Quick Save with outputs after the
export cell finishes to persist the completed files without a second GPU run.

## Windows evaluation

With the pinned b12521 llama.cpp checkout and CPU server, run the evaluator
with `--model` set to the exact stock Q4_K_M file and `--base-config` set to
the directory holding the pinned Hugging Face `config.json`. Pass the two
Kaggle adapter ZIPs together with `--adapters`, the known original stock
journals with `--stock-dir`, and the frozen datasets with `--dev` and
`--mbppplus`. `--output-dir` accepts an empty directory or the same directory
on resume. `python benchmarks/step3_cpu_eval.py --help` lists the flags.

The evaluator applies each converted F16 LoRA adapter directly to the
original Q4_K_M base. It keeps the prior 512 cap, deterministic request
seed 42, temperature zero, one slot and four CPU threads. It reports
exact two-sided McNemar p-values and both median and mean output lengths.
The length gate flags drops greater than 25% relative to stock, and a
checkpoint must improve correctness on both suites without such a drop to
pass. The 25% threshold is an evaluation rule and does not affect training.
Regression labels use the first scored diagnostic and known fixture
boundaries. The CLI rejects any dataset other than the frozen dev and
MBPP+ snapshots.

- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Pinned llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12521)
