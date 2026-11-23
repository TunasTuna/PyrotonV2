# PyrotonV2 Step 3c

Run A used one end-of-turn marker with the original ORPO chosen NLL.
Run B also masked prompt targets in chosen NLL. Both runs used completion
labels for the odds-ratio term and the same 89 pairs, base revision, seed,
and training settings. The fixed dev and MBPP+ comparisons used llama.cpp
b12521 with 4 threads, one slot, seed 42, temperature 0, and a 512-token cap.
The 100-task final holdout was kept sealed.

| Checkpoint | Dev | MBPP+ | Combined | Dev median tokens | MBPP+ median tokens | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Stock | 37/60 | 96/162 | 133/222 | 105 | 62.5 | Baseline |
| Old epoch 1 | 35/60 | 98/162 | 133/222 | 100 | 36.5 | Missed promotion |
| Old epoch 2 | 34/60 | 99/162 | 133/222 | 127 | 27.5 | Missed promotion |
| A-1 | 36/60 | 95/162 | 131/222 | 97.5 | 36 | b: output length stayed short on at least one suite (misses promotion) |
| A-2 | 31/60 | 99/162 | 130/222 | 120 | 27 | b: output length stayed short on at least one suite (misses promotion) |
| B-1 | 32/60 | 98/162 | 130/222 | 60 | 27 | b: output length stayed short on at least one suite (misses promotion) |
| B-2 | 33/60 | 97/162 | 130/222 | 58 | 29 | b: output length stayed short on at least one suite (misses promotion) |

Promotion requires higher correctness on both suites and a median output
length no more than 25% below stock on either suite.

## Paired results

Each cell gives wins–losses and the exact two-sided McNemar p-value.

| Checkpoint | Reference | Dev | MBPP+ | Combined |
|---|---|---|---|---|
| A-1 | stock | 1–2 (p = 1.00000) | 6–7 (p = 1.00000) | 7–9 (p = 0.80362) |
| A-1 | old-1 | 1–0 (p = 1.00000) | 0–3 (p = 0.25000) | 1–3 (p = 0.62500) |
| A-1 | old-2 | 4–2 (p = 0.68750) | 6–10 (p = 0.45450) | 10–12 (p = 0.83181) |
| A-2 | stock | 1–7 (p = 0.07031) | 11–8 (p = 0.64761) | 12–15 (p = 0.70111) |
| A-2 | old-1 | 2–6 (p = 0.28906) | 7–6 (p = 1.00000) | 9–12 (p = 0.66362) |
| A-2 | old-2 | 0–3 (p = 0.25000) | 2–2 (p = 1.00000) | 2–5 (p = 0.45312) |
| B-1 | stock | 1–6 (p = 0.12500) | 10–8 (p = 0.81453) | 11–14 (p = 0.69004) |
| B-1 | old-1 | 1–4 (p = 0.37500) | 6–6 (p = 1.00000) | 7–10 (p = 0.62906) |
| B-1 | old-2 | 1–3 (p = 0.62500) | 5–6 (p = 1.00000) | 6–9 (p = 0.60724) |
| B-2 | stock | 1–5 (p = 0.21875) | 9–8 (p = 1.00000) | 10–13 (p = 0.67764) |
| B-2 | old-1 | 1–3 (p = 0.62500) | 6–7 (p = 1.00000) | 7–10 (p = 0.62906) |
| B-2 | old-2 | 3–4 (p = 1.00000) | 5–7 (p = 0.77441) | 8–11 (p = 0.64761) |

## Training metrics

| Run | First-step loss | First-step chosen NLL | First-step reward margin | Mean margin epoch 1 | Mean margin epoch 2 |
|---|---:|---:|---:|---:|---:|
| Original | 2.6268 | 2.557067 | 0.00051342 | 0.00423944 | 0.00438212 |
| A | 2.5753 | 2.506674 | 0.00142743 | 0.00459018 | 0.00448020 |
| B | 0.3857 | 0.317092 | 0.00142743 | 0.00465933 | 0.00498365 |

The CPU preflight replayed the first three sealed pairs through the pinned
stock Q4_K_M model. Run B's chosen NLL was 0.293168 nats/token across 357
completion targets. The weighted odds penalty was bitwise identical in A
and B at 0.0585845410823822.

The [detailed report](step3c-report.json) includes every paired comparison
against stock and both original epochs, exact two-sided McNemar p-values,
wins and losses grouped by the failing implementation's first diagnostic,
median and mean output lengths by category, and all logged reward margins.

[Run A on Kaggle](https://www.kaggle.com/code/tunastuna/pyroton-v2-single-eot-run-a)
 · [Run B on Kaggle](https://www.kaggle.com/code/tunastuna/pyroton-v2-single-eot-run-b)
