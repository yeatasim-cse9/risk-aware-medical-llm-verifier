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
