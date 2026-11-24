# Experiment reproducibility contract

The authoritative scientific direction is the 10-page approved medical LLM thesis proposal. This repository currently contains **infrastructure scaffolding and a synthetic CPU pilot**, not research results.

## Run metadata to freeze for each real experiment

- Unique experiment/run ID; exact Git commit and script revision
- Data name, access approval, **exact version**, immutable split IDs and hashes
- Model architecture/checkpoint/revision, tokenizer revision, license, inference precision
- Python and package versions, execution environment, CPU/GPU hardware
- Random seed and sampler/progress/RNG state for training; completed task IDs for independent evaluation
- Prompt templates, claim extraction versions, retrieval index & corpus document IDs, retrieval top-k and scores
- NLI and LLM judge versions, sufficiency thresholds, calibration fit split and parameters, risk thresholds
- Per-task outputs where permitted, runtime, errors, and integrity-checked checkpoints
- 95% bootstrap confidence intervals, appropriate paired tests, and no gold-test tuning

## Dataset boundary

MedHallu = development; MedVAL-Bench v1.0.1 = restricted physician gold-test; BioASQ 14b = external test.
MedVAL physician error/risk grades are not automatically the three claim-level evidence labels.
Plan a separate annotation/mapping protocol and freeze it before claim-level gold reporting.
Keep the MedVAL and BioASQ final test sets untouched during threshold/calibration tuning.

## Checkpoint invariants

The synthetic pilot writes an atomically replaced, SHA256-verified JSON checkpoint. Completed IDs and result hashes are validated, duplicates are rejected, and the exact source code commit is embedded in the Kaggle script.
This **does not** prove durability on unplanned Kaggle shutdown or exact optimizer/RNG training resume. Persistent output requires a successful Kaggle version and a correctly attached input source.

## Budget and readiness

Free-tier availability and accelerator quotas can change. Benchmark fit/throughput before full execution. There are no paid API requirements in the synthetic CPU pilot.
