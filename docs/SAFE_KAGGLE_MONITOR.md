# Safe Kaggle Monitoring — Dry Run

This step is deliberately **read-only**: it never creates or updates Kaggle notebooks, touches dataset access, or sends Telegram messages. It checks the previously verified *synthetic* 100-task CPU pilot.

- GitHub workflow: `.github/workflows/kaggle-safe-monitor-dryrun.yml`
- Kaggle input: private CPU-only `yeatasimbillah/medical-llm-cpu-auto-a` saved checkpoint (A v2)
- Exact source SHA: `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`
- Safety: validate checkpoint file checksum, 100 task outputs, canonical ordered task IDs, experiment ID and source SHA
- Expected final decision: `SAFE_MONITOR: STOP ALL_TASKS_COMPLETE_NO_RESTART`

Safety decision table:
| Remote status | Verified checkpoint | Decision |
| --- | --- | --- |
| complete | completed 100/100 | stop; never resubmit |
| complete | in_progress 40/100 or 80/100 | ready_for_review; no automatic submission |
| running or queued | not required | wait; do not submit |
| failed, cancelled, missing, or unknown | any | pause; no submission |
| complete | missing, corrupt or wrong version | pause; no submission |

This dry run tests only the decision to stop a completed experiment and the controller's fail-closed logic. **It does NOT implement automatic monitoring, scheduled polling, deduplicated alerts, safe automatic relaunch, or quota-aware backoff.** Those must be implemented and tested separately, using allowed free-tier resources and no restricted medical data.
