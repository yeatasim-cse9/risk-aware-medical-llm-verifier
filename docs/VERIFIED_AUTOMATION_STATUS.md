# Verified live infrastructure milestones

This file records **observed** cloud events, not claims of completed thesis experiments.

| Verification | GitHub Actions run | Observed outcome |
| --- | --- | --- |
| CPU smoke test | 45460024701 | Success |
| Kaggle API auth | 45467404272 | CLI authenticated successfully |
| Kaggle read-only preflight | 45471346800 | Remote slot A reported missing; success |
| First private CPU-only script submission | 45471589845 | Kernel version 1 successfully pushed |
| Kaggle remote execution and saved output | 45471757482 | Remote status COMPLETE; checkpoint downloaded and validated |

Pilot evidence:
- Notebook: `yeatasimbillah/medical-llm-cpu-auto-a`, version 1
- Frozen source commit: `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`
- Task count: **40 / 100** synthetic tasks
- Verified saved status: `in_progress` (because more synthetic tasks remain)
- Synthetic checkpoint path: `checkpoints/cpu-pilot-automation.json`
- No restricted medical data, patient text, GPU, or paid LLM API used

Still open:
1. Scheduled polling, quota-aware pause/backoff, safe automatic relaunch and error recovery
2. Telegram secret setup, notification delivery and monitoring
3. Real proposal datasets, evidence sources, E1–E7 models and statistical metrics

The test does not guarantee every future Kaggle run will be available, free or uninterrupted.

## Verified Kaggle slot B checkpoint (2026-11-25 Bangladesh local time)

- GitHub Actions A→B submission: run `45472493840`, success; slot B version 1 pushed after validating source A's 40/100 checkpoint.
- Independent remote-output verification: run `45473523954`, success; Kaggle status `complete`, checkpoint downloaded from B saved output.
- Validated **80/100** unique task IDs exactly `0..79`, no repeat of prior 40 tasks, status `in_progress` and source commit `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`.
- The next B→A resume has subsequently been completed and independently verified; see the final section.

## Final synthetic three-session proof (2026-11-25 Bangladesh local time)

- Run `45473921059`: GitHub Actions submitted Slot A version 2 after checking Slot B `complete` and validating Slot B `80/100` checkpoint.
- Run `45474169382`: independently checked Slot A version 2 `complete`, downloaded its persisted Kaggle output, verified checksum, provenance, and `status=completed`.
- Verified completed task IDs precisely `0..99`, with 100 result hashes and no duplicates.
- Stored a synthetic, time-limited GitHub Actions artifact named `final-cpu-pilot-100-task-checkpoint`.
- Final frozen source SHA: `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`.
- **Confirmed:** one remote CPU-only pilot processed 100 deterministic tasks across three separate Kaggle versions (A v1 → B v1 → A v2), using persistent checkpoints.
- **Not confirmed:** unattended/automatic next-session dispatch, quota-aware rescheduling, Telegram notifications, or any E1–E7 scientific evaluation.

## Kaggle → Telegram and safe-monitor proof

- GitHub Actions run `45477514616`: Telegram private notification smoke test succeeded.
- Run `45478249012`: Kaggle Slot A saved output independently downloaded and verified, with **100/100** synthetic tasks; validated operational status message delivered to Telegram.
- Run `45478609612`: Read-only safe-monitor dry run succeeded, producing `SAFE_MONITOR: STOP ALL_TASKS_COMPLETE_NO_RESTART` for the completed, checksum-verified checkpoint.
- Source for the monitor: `scripts/safe_monitor_decision.py`, with standalone unit tests. It cannot dispatch Kaggle versions or send unsolicited status notifications.

**Outstanding:** automatic scheduled status polling, durable progress state/deduplication, quota-aware resubmission and pause/failure recovery. The safe-monitor dry run is not unattended continuous monitoring.

## Scheduled synthetic monitoring and deduplicated Telegram status (2026-11-25 Bangladesh time)

- Workflow: `.github/workflows/synthetic-kaggle-telegram-monitor.yml` with UTC cron at 03:17, 09:17, 15:17 and 21:17, plus manual dispatch.
- Historical message `100/100 completed` from GitHub run `45478249012` was recorded in non-sensitive `monitoring/synthetic-pilot-telegram-ledger.json` to prevent a repeated first message.
- Live GitHub Actions run `45479386760` succeeded: downloaded Kaggle A output, validated `100/100`, reported `SCHEDULED_MONITOR: DUPLICATE_PROGRESS_SKIPPED` and `SCHEDULED_MONITOR: EXPERIMENT_COMPLETE_NO_RESTART`. No Telegram message or Kaggle submission was triggered in this run.
- A completed and already-notified experiment bypasses the Kaggle API on regular scheduled checks; the explicit manual workflow can still verify Kaggle state.
- All Python CI/unit tests passed in `45479386537`.

Still **not** proven: the first actual future cron event execution (GitHub schedules are best-effort), a real increasing checkpoint transition persisted to the public synthetic ledger, quota-aware automatic session relaunch, failure recovery or actual medical experiments. Exactly-once Telegram delivery cannot be guaranteed if delivery succeeds but ledger persistence fails.

## Step 10.4A — read-only two-slot automatic-resume safety preflight

- Run `45480261444`: succeeded with exact private synthetic Kaggle A/B status read and independently downloaded checkpoints.
- Slot A `complete`, validated `100/100`; Slot B `complete`, validated `80/100` against the same frozen source SHA.
- Observed controller decision: `AUTO_RESUME_PREFLIGHT: STOP ALL_100_TASKS_COMPLETE_NO_RESTART`.
- Repository CI and offline safety unit tests: run `45480261149` succeeded.
- No Kaggle submission, Telegram message, medical data processing, or actual quota consumption measurement occurred.
- Policy simulation covers running/queued/failed, missing or corrupt checkpoints, equal-progress ambiguity, unknown quota and insufficient quota. **Real quota monitoring and automated, idempotent resume remain unimplemented.**

## Step 10.4B — CPU quota, cooldown and retry guard

- Live read-only GitHub Actions run `45480973446` succeeded against private Kaggle A/B notebooks.
- Verified Kaggle Slot A `complete` with integrity-checked `100/100` and Slot B `complete` with integrity-checked `80/100`; same frozen SHA.
- Controller correctly output `QUOTA_COOLDOWN_GUARD: STOP ALL_100_TASKS_COMPLETE_NO_RESTART` and `QUOTA_COOLDOWN_GUARD: NO_KAGGLE_SUBMISSION`.
- Deterministic Python unit tests passed for unknown quota, 6-hour cooldown, 12-hour post-failure backoff, 2 consecutive failures requiring review, rolling 24-hour 2-attempt local cap, stale attestation, missing/untrusted history, active runs and 100/100 stop.
- Conservative thresholds are **local safeguards only**, not Kaggle-declared quota limits. No CPU remaining quota was measured or inferred from GPU/TPU reporting.
- No Kaggle job submitted or Telegram message sent. This **does not** enable automatic resume; an independently verified quota/attempt record, durable launch locking and failure recovery remain future work.

## Step 10.4C — idempotent resume-intent and failure-recovery checks

- GitHub Actions run `45481990075` **success**; source patch CI `45481902118` and workflow-change CI `45481989881` both **success**.
- Checked private Kaggle A and B remotely, with SHA-verified snapshots **A 100/100 completed** and **B 80/100 in_progress**. Observed `IDEMPOTENT_RESUME_CONTROLLER: STOP ALREADY_COMPLETE_DUPLICATE_SUBMISSION_BLOCKED`.
- Pure state-machine tests cover unique resume intent, repeated reservation, stale checkpoints, competing claims, CAS conflicts, an attempted-submit marker before pushing, recovery requiring manual review, and verified 40→80→100 reconciliation.
- Tested `kaggle quota` read-only: output `GPU_TPU_QUOTA_API: READ_ONLY_QUERY_SUCCESS`. This confirms query access but **does not prove how many GPU hours remain**. No GPU session was launched.
- A simulated quota-unknown/retry case cannot initiate a new Kaggle job. The production CLI is **read-only**: no live durable claim API integration, no automatic external submission, and no global exactly-once delivery guarantee. Existing manual controller is separate and must not be mistaken for idempotent unattended dispatch.
- The first two development commits exposed test failures; they were fixed and superseded by the successful runs above.

## Step 10.5A — Private CUDA GPU smoke test ready, read-only preflight verified

- GitHub Actions run `45483180421` succeeded: tested private offline GPU metadata, verified pinned Kaggle CLI authentication, queried accelerator quota endpoint successfully and confirmed the separate GPU notebook `yeatasimbillah/medical-llm-gpu-smoke` is **missing**, therefore no duplicate exists.
- GitHub CI `45483180468` succeeded, including GPU package compilation and synthetic integrity unit tests.
- Private metadata uses `enable_gpu=true`, `is_private=true`, `enable_internet=false` and no Kaggle dataset/kernel/model inputs.
- GPU computation, CUDA availability and remote saved GPU checkpoint **have not yet been verified**; actual one-time Kaggle GPU submission requires explicit user review and `confirm_gpu_quota=true` in the manual controller. No GPU quota was spent by read-only preflight and no exact remaining quota has been claimed.
- Original CPU pilot and restricted MedVAL-Bench remain untouched.

## Step 10.5A — First real private GPU CUDA execution confirmed

- GitHub workflow run `45483951593`: one-time `kaggle kernels push` successfully submitted private notebook `yeatasimbillah/medical-llm-gpu-smoke` version 1 with frozen Git SHA `d92b5d6054b6f3802d146e5a1662127e61223993`. Explicit manual GPU approval and duplicate-slug gate passed.
- Independent read-only verification run `45484283016`: Kaggle reported remote notebook `complete`. Downloaded its saved synthetic checkpoint, verified SHA-256 integrity and frozen source SHA, and validated CUDA matrix calculation. Logs contain `VERIFIED_PRIVATE_GPU_SMOKE: CUDA_MATRIX_OK`, `VERIFIED_PRIVATE_GPU_SMOKE: SOURCE_SHA_MATCHED` and `GPU_SMOKE_REMOTE_CHECKPOINT_VERIFIED`.
- A non-sensitive 7-day GitHub Actions artifact `verified-gpu-smoke-45484283016` was saved. The private GPU notebook is separate from CPU A/B. No medical dataset, MedVAL-Bench, real LLM or model weights were involved.
- **Confirmed:** GPU-enabled private Kaggle notebook can execute PyTorch CUDA computation and return an integrity-validated synthetic checkpoint to GitHub. **Not confirmed:** inference/training on a medical model, accurate remaining weekly GPU quota, GPU resume across sessions, or full scientific E1–E7 evaluation.

## Step 10.5B — Tiny Transformer GPU inference prepared and safely preflighted

- GitHub Actions run `45485423732` **success**: eight offline unit tests passed; Kaggle GPU quota endpoint accessible; new private notebook `yeatasimbillah/medical-llm-gpu-inference-pilot` reported `missing` (not yet submitted).
- The notebook source builds a tiny randomly initialized one-layer Transformer Encoder with a synthetic 3-class prediction head. It performs actual CUDA forward inference on four synthetic token sequences, saves model weights, reloads them for comparison and persists a source-pinned SHA256 result manifest. It has **no pretrained medical weights or medically meaningful labels**.
- Metadata requires private GPU notebook, internet disabled and no input datasets/models/kernels/competitions. Original CPU A/B and GPU smoke notebooks remain unchanged.
- **No GPU time consumed by this preflight.** Next step is a distinct, explicitly approved one-time user-initiated Kaggle GPU submission; any claim of successful inference awaits downloaded Kaggle output and independent verification.
- Expected fixed documentation and workflow: `docs/TINY_GPU_INFERENCE_RUNBOOK.md`, `.github/workflows/kaggle-tiny-gpu-inference.yml`.

## Step 10.5B — First private GPU Transformer inference and checkpoint restored

- Kaggle submit workflow run `45485785310` completed successfully: version 1 of private `yeatasimbillah/medical-llm-gpu-inference-pilot` pushed under frozen source SHA `184a1270aa6aaef8bc0e82c142e3285d0ea82084` with explicit user approval.
- Independent GitHub verification run `45485964322` succeeded: Kaggle remote status `complete`; stored inference outputs and weights independently downloaded; verifier confirmed correct manifest, softmax probabilities, predicted classes, saved weights SHA-256, source SHA and recorded restore comparison. Log: `TINY_GPU_MODEL_REMOTE_INFERENCE_VERIFIED`, `TINY_GPU_MODEL_CHECKPOINT_AND_RESTORE_VERIFIED`, `TINY_GPU_MODEL_SOURCE_COMMIT_MATCHED`, `TINY_GPU_MODEL_REMOTE_CHECKPOINT_VERIFIED`.
- GitHub Actions artifact `verified-tiny-gpu-inference-45485964322` retained for seven days, containing **only synthetic** inference manifest and untrained randomly initialized weights. Original CPU pilot and GPU CUDA smoke notebook were not overwritten.
- Relevant CI `45485964364` passed. No second GPU job was launched by independent verification.
- **Scope limit:** This is a toy one-layer PyTorch Transformer with 4 synthetic token sequences and 3 synthetic output classes; it is not a pretrained medical model, nor a hallucination detection, clinical or E1–E7 experiment. Scientific quality and actual remaining weekly accelerator quota are unknown.

## Step 10.5C — Pretrained biomedical-model selection and declared-license audit

- Official upstream Hugging Face cards inspected on November 25, 2026: `sentence-transformers/all-MiniLM-L6-v2` (embedding baseline, card declares Apache-2.0); `cross-encoder/nli-MiniLM2-L6-H768` (SNLI/MultiNLI three-label NLI baseline, card declares Apache-2.0); `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (biomedical masked-language model, card and LICENSE.md indicate MIT).
- Saved source links and limitations in `docs/PRETRAINED_MODEL_LICENSE_AUDIT.md`, with strict metadata-only registry at `configs/pretrained-model-audit.json`.
- GitHub Actions `45486737054` **success**; 8 unit tests plus policy validation passed. Log: `PRETRAINED_MODEL_AUDIT: MODEL_CARD_METADATA_VERIFIED 3_CANDIDATES` and `PRETRAINED_MODEL_AUDIT: NO_GPU_EXECUTION_NO_MODEL_DOWNLOAD`.
- **All deployment gates remain closed:** upstream model/tokenizer revision and SHA256 not yet frozen; actual weight license notices and downstream distribution rights not fully reviewed; no actual pretrained model VRAM measurement or inference; no authorized model bundle upload and no real dataset use.
- Earlier Kaggle GPU smoke and tiny untrained Transformer validation are infrastructure evidence only, not medical evidence-grounded evaluation. Restricted MedVAL-Bench has not been uploaded or processed.

## Step 10.5D — First pretrained MiniLM immutable metadata snapshot

- GitHub Actions `45487590017` **success**: 8 provenance/safety tests passed; Hugging Face live public API supplied immutable `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, model safetensors file size `90,868,376` bytes and metadata LFS SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
- Hub reports 22,713,728 parameters, giving theoretical parameter-only storage 90,854,912 bytes FP32 or 45,427,456 bytes FP16. These do **not** prove peak GPU memory or actual feasibility.
- Recorded exact evidence in `configs/model-pins/all-minilm-l6-v2.json` and `docs/MINILM_PROVENANCE_FEASIBILITY.md`. Independent byte-level file hash verification, tokenizer hashes, actual weight/model license-text review, permissible transfer route, GPU memory benchmark and separate approval remain pending.
- No model or tokenizer bytes were downloaded; no Kaggle GPU session was run; no actual medical datasets or E1–E7 experiments were processed.
