# Step 10.3 — scheduled, deduplicated monitoring of the synthetic CPU pilot

**Scope:** Synthetic Kaggle CPU-pilot only. No restricted data or patient text is processed or transmitted. All scientific E1–E7 evaluations remain separate future work.

## What is deployed

- Workflow `.github/workflows/synthetic-kaggle-telegram-monitor.yml` runs at UTC 03:17, 09:17, 15:17 and 21:17 (four checks/day), subject to GitHub Actions scheduling availability.
- The workflow is also manually runnable. A one-time `push` trigger performs an independent remote check when first added.
- A versioned *public synthetic-only* ledger at `monitoring/synthetic-pilot-telegram-ledger.json` stores the last successfully notified **task count and status**, not any secret or clinical content.
- The ledger is bootstrapped at **100/100, completed**, because GitHub run `45478249012` had already delivered that progress to Telegram. This intentionally prevents the duplicate first notification.
- On **scheduled** runs, if the immutable pilot is already terminal and notified, the controller skips the Kaggle API entirely, conserving quota and Actions runtime.
- On **manual and initial push** runs, it checks Kaggle Slot A's exact completed kernel, downloads the checkpoint and validates its checksum, task outputs and source SHA. It suppresses repeated progress notifications using the persisted ledger.
- On a future legitimate **increasing task count**, it sends a bounded synthetic-only Telegram message and commits the small updated ledger file to the default branch. A `GITHUB_TOKEN`-authenticated update does not create a new push workflow run.
- The workflow never uses `kaggle kernels push` or starts GPU work.

## Safety and limits

- `contents: write` is required **only** to persist the tiny public, synthetic metadata ledger using GitHub's Contents API. Do not put credentials, Chat IDs, patient data or real experiment outputs in the ledger.
- GitHub Actions concurrency serializes instances of this monitor workflow. If someone manually edits the ledger or another separate workflow posts, duplicates are possible; this is not a global lock.
- Delivery-first ledger-second provides at-least-once semantics. If Telegram accepts the message but GitHub state update fails, a later run may repeat the message. **Exactly-once delivery is not guaranteed.**
- A missing or corrupt ledger, changed checkpoint commit, unrecognized task IDs, regression, status mismatch or Telegram/GitHub API failures stop delivery (fail-closed). A notebook that is incomplete/failed also causes a failed poll, with no false completion notification.
- Public GitHub repository scheduled workflows can be delayed, skipped under load or disabled after 60 days without repository activity. Do not treat the schedule as an always-on service.
- **No automatic Kaggle session restart or quota-aware relaunch is implemented.** Those belong to Step 10.4 and must be tested separately.

## Manual validation

Actions → Scheduled Synthetic Kaggle Telegram Monitor → Run workflow → `main`.

Expected on the already completed pilot:
`SCHEDULED_MONITOR: DUPLICATE_PROGRESS_SKIPPED`
`SCHEDULED_MONITOR: EXPERIMENT_COMPLETE_NO_RESTART`

Expected on a scheduled poll, because 100/100 is already recorded:
`SCHEDULED_MONITOR: TERMINAL_ALREADY_REPORTED_SKIP_KAGGLE`

The initial scheduled monitor run uses only the saved Kaggle output, not new compute sessions.
