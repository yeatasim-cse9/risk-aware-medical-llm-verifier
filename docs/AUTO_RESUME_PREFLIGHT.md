# Step 10.4A — Safe Automatic Resume Preflight (read-only)

This phase **does not automatically relaunch** any Kaggle notebook.

The standalone GitHub Actions workflow `.github/workflows/auto-resume-safety-preflight.yml` checks the actual private synthetic CPU-only notebook slots A/B and downloads every completed slot's saved output. `verify_checkpoint.py` verifies task hashes, checkpoint checksum, immutable source commit and record schema. The decision layer chooses the highest verified checkpoint count as the latest progression.

Decision rules:

- **100/100 verified:** `STOP ALL_100_TASKS_COMPLETE_NO_RESTART`. This is the expected actual live result.
- **Any active remote Kaggle notebook:** `WAIT`, never start another.
- **Failed/unknown/missing checkpoint, checksum or source-commit mismatch, two equally progressed snapshots:** `PAUSE` and manual investigation.
- **Valid partial checkpoint (40/100 or 80/100):** never authorize automatic submission when Kaggle quota is unknown; `PAUSE QUOTA_NOT_INDEPENDENTLY_VERIFIED`.
- Even if an operator explicitly attests adequate quota during separate offline testing, output is only `READY_FOR_MANUAL_REVIEW`, not approval to submit.

**Quota limitation:** This preflight does not measure Kaggle's current free-tier remaining CPU/GPU hours, nor assume an unpublished or unstable quota API. Unknown quota is intentionally a hard block to resume. A quota-aware and duplicate-proof submit controller remains for a later stage after independently confirming safe usage information and testing limits.

The workflow uses read-only `contents: read` GitHub permission and read-only Kaggle operations, requires no new secret and sends no Telegram messages or any data. Never use this path with restricted MedVAL-Bench or patient data.
