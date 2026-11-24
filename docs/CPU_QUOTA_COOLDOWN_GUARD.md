# Step 10.4B — CPU Quota & Cooldown Guard

This phase is **read-only** and intentionally cannot submit or restart Kaggle notebooks.

Verified Kaggle CLI documentation describes `kaggle quota` as a GPU/TPU weekly accelerator quota report, not a proven source of CPU notebook allowance. The CPU-only pilot therefore uses `quota_signal=unknown` in the live workflow, which blocks any incomplete-run resume. **No remaining CPU hours or Kaggle account quota have been measured or claimed.** Never invent an exact weekly quota.

## Policies tested with offline synthetic fixtures

- A completed checkpoint wins: `STOP` for the verified 100/100 synthetic pilot regardless of unknown quota.
- Any running/queued notebook produces `WAIT`; failed/unknown state produces `PAUSE`.
- Partial 40/100 or 80/100 plus unknown/insufficient CPU quota produces `PAUSE`.
- Even if a human attests enough quota and records a fresh timestamp (within 30 minutes), result is at most `READY_FOR_MANUAL_REVIEW`, **never permission to launch**.
- Absent, incomplete, spoofed, duplicate, unsorted or future-dated attempt history produces `PAUSE`. There is no assumption that an empty or missing file means zero previous submissions.
- Local conservative guard policy: minimum **6-hour cooldown** after a completed submission; **12-hour cooldown** after the latest failed attempt; at **2 consecutive failures**, require manual diagnosis. Maximum **2 submission attempts within a rolling 24 hours**. These are our self-imposed research safety thresholds, **not documented Kaggle quotas**.
- Checkpoints use the same fixed synthetic experiment ID and frozen Git source SHA validated by Step 10.4A. Code prohibits arbitrary benchmark data and contains no Telegram messages or notebook-submission calls.

Workflow: `.github/workflows/cpu-quota-cooldown-guard.yml`. It checks the current private Kaggle slot A/B notebook statuses, downloads completed saved checkpoints, and runs the above guard with unknown quota and no operator history. Expected actual outcome: `QUOTA_COOLDOWN_GUARD: STOP ALL_100_TASKS_COMPLETE_NO_RESTART`.

**Still not implemented:** live CPU quota measurement, a persisted trustworthy launch-attempt record, failure notifications, or automatic idempotent relaunch. Cooldown/backoff tests are deterministic simulations, not actual quota depletion or real Kaggle failures. Step 10.4C must resolve those before an automatic submit path is ever enabled. The runner uses `contents: read` and never runs `kaggle kernels push`.
