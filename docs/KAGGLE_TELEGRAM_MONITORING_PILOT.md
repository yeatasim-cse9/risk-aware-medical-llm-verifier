# Integrated Kaggle → Telegram Monitoring Smoke Test

This is a **manual-only**, single-shot, synthetic CPU infrastructure test. It does not start or restart a Kaggle notebook and does not contain any real medical data.

Workflow: `.github/workflows/kaggle-telegram-status-test.yml`

1. GitHub Actions uses the repository's `KAGGLE_API_TOKEN` secret to check the completed private Kaggle notebook `yeatasimbillah/medical-llm-cpu-auto-a`.
2. It downloads the saved Kaggle checkpoint from the latest notebook output, requiring `complete` status.
3. `scripts/verify_checkpoint.py` validates checkpoint SHA256, task-result integrity, expected Git commit, and record metadata.
4. `scripts/notify_synthetic_pilot_status.py` accepts only experiment `cpu-pilot-automation-001`, commit `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`, 100 expected tasks and valid consecutive IDs. It sends the verified progress to the configured private chat.
5. `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are read from GitHub Actions Secrets and never printed. Failed verification sends **no** completion message.

**Not provided yet:** periodic polling, change-detection and deduplicated notifications, session submit/restart, quota-aware backoff, failure recovery, model-training checkpoint logic. This workflow should not be scheduled: it sends one progress message each time it is manually started and the verified checkpoint is available.

**Privacy boundary:** Only the hard-coded synthetic experiment ID and bounded task counts are sent to Telegram. Never adapt this flow to send private medical claims, patient text, restricted benchmark labels or evaluation predictions.
