# Step 10.5F — Automatic research completion/failure Telegram alerts

Scope: **allowlisted GitHub Actions workflows, four historical synthetic Kaggle notebooks, and one approved public MedHallu metadata-only notebook.** Real E1–E7 are NOT running. This is not a generic subscriber to arbitrary programs, datasets, private accounts or future medical experiments.

## Two event sources

1. **GitHub Actions `workflow_run` events (prompt):** a separate notification workflow reacts when an allowlisted workflow starts or finishes on this repository's `main` branch. It reads GitHub-supplied run status, ID, attempt and a strict workflow-name allowlist; it does NOT execute or download artifacts from the triggering run. It notifies privately about success/failure, timeout or cancellation. A successful GitHub **submission workflow** means the controller succeeded, NOT that a Kaggle notebook has finished.
2. **Kaggle notebook polling (best effort):** an hourly GitHub cron at UTC minute `:37` queries the private Kaggle status API ONLY for newly registered, previously unnotified synthetic jobs. Remote status `complete` is never sufficient by itself: the saved output must pass the exact checkpoint/hash/source-revision verifier before a completion message. Terminal Kaggle errors generate a sanitized failure alert. Unknown/active/queued statuses are not claimed as success.

## Registered notebook list

- `yeatasimbillah/medical-llm-cpu-auto-a`: CPU synthetic A; 100/100 verified terminal.
- `yeatasimbillah/medical-llm-cpu-auto-b`: CPU synthetic B; saved output 80/100, despite remote kernel version `complete` (not the whole experiment).
- `yeatasimbillah/medical-llm-gpu-smoke`: private CUDA smoke checkpoint verified.
- `yeatasimbillah/medical-llm-gpu-inference-pilot`: private untrained Transformer inference and saved weights verified.

These FOUR are historic, already completed notebooks. Their notifications are initialized as `historical_suppressed` to avoid retroactive spam and redundant GPU/Kaggle queries. The CPU A 100/100 had previously been delivered to Telegram via a separate workflow; suppression does not claim messages were sent for the others.

**Before a future Kaggle study can receive automatic terminal notifications:** the developer must add a new, distinct private notebook slug and immutable source SHA to the reviewed `WATCHED` allowlist in `scripts/universal_telegram_alerts.py`, provide its task-specific saved checkpoint verifier, and atomically update the state ledger with an unnotified `last_event=null, delivery=none` entry. Do not silently register clinical content or unrestricted/third-party notebooks. Reusing an already-completed slug without a new version-identity audit is not supported.

## Deduplication and security

- New non-sensitive ledger: `monitoring/universal-telegram-alert-ledger.json`, modified with GitHub's expected blob SHA (compare-and-swap). Global GitHub Actions concurrency serializes this workflow's live writers; a write conflict fails closed.
- A completed GitHub workflow is identified by `workflow_run.id`, `run_attempt` and conclusion. A registered Kaggle outcome is identified by its trusted job identifier and terminal state.
- Telegram delivery occurs before ledger persistence; if a bot send succeeds but a GitHub ledger update fails, a duplicate is possible. **This is best-effort at-least-once, not an exactly-once guarantee.**
- Repeated ordinary CI successes on `push` and successful periodic CPU monitor heartbeats are intentionally muted to avoid noise/loops; CI failures on allowed main-branch runs still alert. For GitHub Actions, both success and failure of manual workflows on the allowlist alert.
- Workflow checks `github.repository`, `head_repository.full_name` and `head_branch=main`; private Telegram secrets stay GitHub Actions secrets, never in logs, Git commits, checkpoint artifacts or messages.
- The workflow has `contents:write` solely for its versioned sanitized ledger; it never calls `kaggle kernels push`, enables GPU or handles patient/MedVAL records. Telegram messages are fixed operational status texts with no raw logs or arbitrary model outputs.
- GitHub schedules are not exact and can be delayed/skipped. Kaggle completion notifications can therefore arrive later than the actual finish; no message is guaranteed if GitHub/Telegram APIs are unavailable. GitHub workflow_run events should be near-immediate but may also be delayed.
- This step does NOT provide universal failure catch-all for all possible workflows or medical-data safety clearance. A new research workflow must be reviewed and added to the allowlist before alerts will trigger.

## Operations and files

- Workflow: `.github/workflows/universal-research-telegram-alerts.yml`.
- Program: `scripts/universal_telegram_alerts.py`.
- Ledger: `monitoring/universal-telegram-alert-ledger.json`.
- Tests: `tests/test_universal_telegram_alerts.py`.

On first push of the workflow, the job runs offline safety tests and sends **one** private, non-medical Telegram setup message, which is recorded in the ledger and cannot deliberately replay through normal workflow use. It does not backfill old Kaggle results. A `workflow_dispatch` with `operation=poll` checks pending jobs only; `operation=smoke` tries the already-deduplicated one-time setup message.

Existing CPU pilot's older `synthetic-kaggle-telegram-monitor.yml` remains in place and independently suppresses duplicate 100/100 status messages. It should not be reconfigured to send clinical contents. No E1–E7 datasets have been acquired or evaluated.

## First live verification — November 25, 2026 (Bangladesh)

- `45544039917`: successful setup bootstrapping; exactly one operational Telegram confirmation was sent and `bootstrap_sent=true` persisted to the GitHub non-sensitive ledger.
- `45544218761`: successful safe re-test of the same `push` event: `BOOTSTRAP_ALREADY_RECORDED`, then `HISTORICAL_KAGGLE_ALERTS_SUPPRESSED`; **no repeat setup alert and no retroactive Kaggle messages**.
- 16 alert-specific Python unit tests passed in the final live repeat run; repository CI `45544218842` also succeeded.
- Real `workflow_run` handling was exercised by `45544022815` and `45544184671`. Both correctly skipped routine GitHub CI push successes (not a test of sending an actual job-failure alert).
- **Not yet independently observed**: the first hourly GitHub `schedule` event, an actual future workflow failure notification, and a newly registered Kaggle terminal-state message. The unit tests simulate those situations but do not replace a live failure/end-to-end alert observation.
- Full unattended Kaggle relaunch, exact-once Telegram delivery, restricted medical dataset use, and scientific E1–E7 work remain out of scope.

## Start/finish notification expansion — 2026-11-25

- Allowlisted workflow start (`in_progress`) and finish (`completed`) events now have separate deduplicated Telegram messages. GH workflow start/end **are not remote Kaggle experiment start/end**.
- The existing MedHallu public metadata-only Kaggle notebook (Version 2, exact code SHA `9fcfaad9ec13c4fd524e14ae51d85092eca6fe46`) is newly registered. Scheduled read-only polling and the first monitor push will send a status notification if running/queued, verified complete, or failed. If the first poll occurs after terminal status, an earlier start cannot be inferred.
- The metadata manifest must pass independent verifier `verify_medhallu_audit.py` before a Kaggle success message is allowed. No raw text, full logs or restricted data ever go to Telegram.
- Future E1–E7 workflow names are reserved in allowlists, but **no scientific experiment workflows are active yet**. Every later Kaggle experimental notebook must be approved and registered with source revision and independent checkpoint verification before remote completion notifications can be claimed.
- At-least-once delivery: failures in Telegram or ledger persistence may delay alerts or cause duplicates; hourly GitHub cron is not an exact real-time guarantee. Never equate GitHub controller success with verified Kaggle completion.
