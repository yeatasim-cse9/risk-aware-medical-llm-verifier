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
