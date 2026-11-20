# Pyroton V2 evaluation split and Step 2 data contract

## Evaluation split

The 160 scored tasks from Step 1 are partitioned into 60 development and 100
final-test tasks. Dev has 54 HumanEval+ and six original validation tasks.
Final test has 91 HumanEval+ and nine original validation tasks. Assignment
uses only IDs, data sources and fixed groups of related prompts. It uses no
model outcomes. The manifest records the source and split hashes.

Use dev for model selection, prompt and decoding choices, early stopping and
hyperparameter decisions. Reserve final test and the audited MBPP+ test
partition for evaluation after those decisions are fixed. The original 40
tasks remain a separate regression suite. Every evaluation partition is
evaluation-only. The 160-task parent set and its baseline outputs were
already inspected before this split. Its new partitions are prospectively
reserved, and scores on them do not represent a previously unseen evaluation.

To recreate the split, extract `evaluation/scored-holdout.json` from the Step 1
archive and run:

```sh
python benchmarks/split_holdout.py --source scored-holdout.json --output-dir heldout
```

The separate MBPP+ audit documents upstream IDs, source versions, withheld
splits, training overlap, and retained or excluded test items. Before sampling
or training, check every prospective training task against dev, final test,
the original 40, and the upstream MBPP+ test pool. Block exact and known near
matches on task text, identifiers and reference behavior. Version and archive
the audit with the training data. Keep evaluation cases, references, and
baseline generations out of prompts, SFT completions, and preference pairs.

### MBPP+ second holdout

Use `evaluation/v2-mbppplus-test.json` for generation and scoring. It keeps
the pinned MBPP+ v0.2.0 base and augmented inputs for the retained task IDs.
MBPP+ v0.2.0 contains multiple splits. The candidate pool is the intersection
of that release and Google's original MBPP test IDs 11 through 510.

The build compares candidates with all 94 recorded MBPP v1 training examples,
the repository training JSONL files, the original 40 prompts, and the 160
primary holdout tasks. It uses normalized text, normalized code, and a
conservative manual review of near variants. An audit entry records each
exclusion. A reference preflight excludes any additional tasks where the
answer key fails this sandbox or exceeds its resource limits.

`score_holdout.py` restores MBPP+ tuple, set, complex, and non-finite numeric
fixtures and applies the comparators needed by the retained IDs. It uses the
same WASI subprocess and watchdog as the primary suite. Scores are for this
audited subset and this sandbox. The raw EvalPlus rows are also included for
reproducibility.

- [MBPP+ v0.2.0 release and data](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0)
- [Google Research MBPP split definition](https://github.com/google-research/google-research/tree/master/mbpp#evaluation-details)

Rebuild with the original MBPP sources and provenance from Step 1, then
preflight the references and freeze the passing subset:

```sh
python benchmarks/build_mbppplus_holdout.py --release MbppPlus.jsonl.gz --mbpp sanitized-mbpp.json --training-metadata mbpp-v1-training-metadata.json --notebook pyroton-1.5b-mbpp-training.ipynb --primary scored-holdout.json --old-report primefix-benchmark.json --output-dir mbppplus-audited
python benchmarks/score_holdout.py --dataset mbppplus-audited/v2-mbppplus-test.json --output mbppplus-reference-preflight.json
python benchmarks/build_mbppplus_holdout.py --release MbppPlus.jsonl.gz --mbpp sanitized-mbpp.json --training-metadata mbpp-v1-training-metadata.json --notebook pyroton-1.5b-mbpp-training.ipynb --primary scored-holdout.json --old-report primefix-benchmark.json --reference-scores mbppplus-reference-preflight.json --output-dir heldout
```

## Step 2 training records

Use the same system and user prompt for both files. Include the exact function
name and signature, return rules and any required validation in the user
message, followed by a request for complete Python code and imports. Record the
prompt as a list of system and user messages. The training adapter can format
those messages using the pinned model chat template. When preparing a
conversational trainer row, wrap each stored response string as an assistant
message. A preformatted trainer row instead renders both prompt and responses
as strings with the same template.

- **`sft.jsonl`** has one record per distinct passing solution. Store
  `problem_id`, `source`, `source_split`, `test_suite_sha256`, `prompt`,
  `completion`, and `completion_tokens`. The completion is plain executable
  Python. Train loss on the assistant completion. Deduplicate and cap the
  number of near-identical passes per problem.
- **`orpo.jsonl`** has `problem_id`, `source`, `source_split`,
  `test_suite_sha256`, `prompt`, `chosen`, `rejected`, `failure_category`,
  `first_failed_check`, `chosen_tokens`, and `rejected_tokens`. Chosen and
  rejected are assistant-only strings for the same exact prompt, signature
  and tests. Pass only prompt and response content to the trainer. Keep
  provenance and diagnostics in metadata.
- Record sampling seed, raw token count, stop reason, raw response, extraction
  result and sandbox status in the sampling audit. Save the per-problem
  distribution including zero-pass problems. Failure labels should describe
  the first verified cause, with execution timeouts and malformed syntax
  recorded as secondary effects when another cause is known.

Extract only a clearly delimited Python block or a plain Python response.
Verify the exact normalized code destined for the data file. Use a clean
sandbox namespace and check the expected entry point and signature before
behavioral tests. Supply no implicit imports to candidates. Test ordinary
cases plus meaningful empty input, negatives, type traps and boundaries
where the prompt defines what should happen. An invalid input check should
assert the specified exception, rather than assuming all Python functions
must reject out-of-contract inputs.

An execution-passing completion is eligible for SFT. Prefer compact passing
code with all necessary imports and validation. For ORPO, prioritize
syntactically valid, near-length pairs with the signature intact where the
rejected solution fails a logic or edge-case test. Keep caps and extraction
failures as separate diagnostic buckets. Reserve a small, separate pairing
quota for them only if explicitly targeting output format. A short answer
cannot compensate for omitted behavior. Track solution and generation token
lengths separately so a change in style does not mask correctness.

The 40 changed Step 1 outcomes support these choices. MBPP v1's 26
regressions were logic or edge/validation failures. Five of its 14 wins
came from stock Qwen running out of tokens before completing the code.
These 40 evaluation texts and their generations stay out of Step 2.
