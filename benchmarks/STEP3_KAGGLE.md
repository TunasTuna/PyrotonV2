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

## Pilot results

Neither checkpoint passed the two-suite gate. The frozen stock model
passed 37/60 on dev and 96/162 on MBPP+. The fixed stock repeats produced
identical answers and verdicts on all 222 tasks.

- Epoch 1 passed 35/60 on dev (0 wins, 2 losses, exact McNemar p = 0.5000)
  and 98/162 on MBPP+ (6 wins, 4 losses, p = 0.7539).
- Epoch 2 passed 34/60 on dev (1 win, 4 losses, p = 0.3750) and 99/162
  on MBPP+ (11 wins, 8 losses, p = 0.6476).
- Median MBPP+ output fell from 62.5 tokens for stock to 36.5 and 27.5.
  Median dev output was 105, 100 and 127 tokens, respectively.
- Manual review found 15 logic regressions, one zero-input regression,
  and two missing-import regressions across both checkpoints. This counts
  a task again when both checkpoints regressed on it.
- Training completed 23 optimizer steps per epoch, resuming from the
  saved optimizer state after an adapter export callback error. Mean
  logged loss was 2.0813 and 1.7116. Mean reward margin rose from 0.004239
  to 0.004382, with no non-finite metrics recorded.

- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Pinned llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12521)
