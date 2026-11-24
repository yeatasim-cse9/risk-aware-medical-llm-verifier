# Step 10.5E — Read-only end-to-end synthetic infrastructure validation

Purpose: recheck all completed historic synthetic Kaggle infrastructure experiments and connect their saved evidence to persistent status and duplicate-notification ledger controls. **No new Kaggle job, GPU session, pretrained model download, clinical dataset or Telegram message is triggered.**

## Evidence path

1. GitHub Actions checks out fixed source, runs the full Python repository test suite and installs the previously validated pinned Kaggle CLI.
2. Using existing `KAGGLE_API_TOKEN`, the runner checks exact status of CPU A, CPU B, CUDA GPU smoke and the tiny untrained Transformer GPU notebook. **All must be `complete`** before downloading any output.
3. Independently downloads each notebook's **saved** outputs with `kaggle kernels output`. All Kaggle references are fixed in code and the only subprocess command is `kaggle kernels output`.
4. Validates CPU A 100/100 and B 80/100 against original immutable CPU experiment SHA; checks CPU `STOP` and double-submission-blocked control ledger.
5. Compares validated A 100/100 progress against the **existing** Telegram dedup ledger and validates the sanitized notification template without sending a message.
6. Validates previously executed CUDA matrix test and toy untrained Transformer inference, weight-file SHA-256, checkpoint-reload evidence and exact SHA of their separately frozen GPU submissions.
7. Confirms the pinned `all-MiniLM-L6-v2` repository metadata record still has **closed** deployment, pretrained weight, MedVAL transfer and VRAM-fit flags. Validates source-linked three-model license metadata registry as well.
8. Creates a small sanitized JSON report with SHA-256 manifest digest as a 7-day GitHub Actions artifact. Does not archive any downloaded Kaggle checkpoint, weights, patient data, credentials, Telegram IDs or user secrets.

## Key distinctions

- This is a **historical synthetic evidence revalidation**. Its success means all four already completed Kaggle notebooks can still be queried, their saved checkpoint outputs are internally consistent, and the existing completion/dedup ledger agrees.
- It **does not** prove GitHub `schedule` cron firing. That requires observing an actual `schedule` event, not a manual or `push` test.
- It **does not** resend a Telegram message or prove current Telegram deliverability. Delivery was previously confirmed in historic run 45478249012, while today's step verifies only the ledger and sanitized message construction.
- It **does not** prove unattended Kaggle submission/retry, global exactly-once protection, live CPU quota, current GPU hours, full MiniLM license clearance, actual pretrained model GPU fit, dataset rights, or thesis experiments E1–E7.
- GitHub remote source SHA checks and synthetic output hashes do not establish medical correctness or scientific validation.
- A future license-audited pretrained pilot requires independent binary hash verification and separate GPU user consent. Restricted MedVAL-Bench remains in a separately approved environment.

Workflow: `.github/workflows/e2e-readonly-infrastructure-audit.yml`. Script: `scripts/e2e_infrastructure_audit.py`. Tests: `tests/test_e2e_infrastructure_audit.py`.

**Safe operation:** Manual or one-time matching workflow push only. No schedule and no new paid/accelerator compute. The workflow permission is `contents: read`; Telegram secrets are not mounted.
