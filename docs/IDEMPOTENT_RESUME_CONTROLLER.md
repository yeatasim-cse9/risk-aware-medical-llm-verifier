# Step 10.4C — Duplicate-proof resume intents and failure recovery

Only a synthetic CPU pilot is in scope. **This stage is read-only in production.**
No workflow in this stage calls `kaggle kernels push`, creates jobs, sends notifications or edits medical data.

## What has been implemented

- `scripts/idempotent_resume_controller.py`: pure state machine with a strict, frozen experiment ID and code commit. It refuses checkpoint regression and unrecognized resume edges; a reservation is uniquely keyed by source SHA, task count and A/B transition.
- `monitoring/synthetic-resume-control.json`: small, non-sensitive baseline ledger set to `completed`, **100/100** in slot A, reflecting already observed completion. There are no secrets or patient records.
- `tests/test_idempotent_resume_controller.py`: mock simulations for duplicate claims, competing intent keys, stale checkpoints, active/failed runs, corrupt ledgers, ambiguous delivery, successful 40→80 and 80→100 reconciliation, and GitHub CAS write conflicts.
- `.github/workflows/kaggle-idempotent-resume-controller.yml`: live read-only GitHub workflow. It checks Kaggle A and B, verifies saved checkpoints, compares the ledger against remote state and must output `IDEMPOTENT_RESUME_CONTROLLER: STOP ALREADY_COMPLETE_DUPLICATE_SUBMISSION_BLOCKED`.
- Separate read-only `kaggle quota` probe tests that account accelerator quota introspection is available; **it does not spend GPU quota** and does not claim remaining CPU availability.

## How durable idempotency would work if a future submitter were enabled

1. Recheck remote Kaggle status, validated checkpoint, source commit, target notebook and available resources. Use the 10.4B cooldown and failure thresholds and an independently trustworthy attempt history.
2. Persist a unique `reserved` intent to GitHub using a compare-and-swap ledger update. A competing writer conflict stops the launch.
3. Persist `submitted` (submission attempted) **before** invoking the irreversible Kaggle push, to make any crash default to no retry.
4. Reconcile remote terminal status and independently verified checkpoint before advancing the ledger. A crashed or ambiguous push becomes `needs_review`, **never an unattended retry**.
5. Only one writer may own the claim. GitHub Actions `concurrency` protects this workflow, but not external manual submissions or other workflows. Completely global exactly-once submission is **not guaranteed** by Kaggle and must not be advertised as such.

This document describes the safe state-machine protocol and future requirements. **The controller does not currently perform steps 2–4 in production**, because actual trusted CPU quota/attempt history and a single centralized dispatch service are not yet deployed. The code is deliberately incapable of submitting jobs, even if a caller toggles a GitHub Actions input.

## GPU vs CPU

Our current CPU-only synthetic checkpoint pilot computes deterministic SHA-256 hashes. A GPU would waste accelerator quota without improving the validity of this automation test.

Kaggle supports GPU notebooks by selecting an accelerator and has an accelerator quota query (`kaggle quota`). GPU capacity is useful later for medical transformer inference, embeddings and possibly parameter-efficient fine-tuning, subject to current entitlement, licenses, VRAM and model availability. Do not retrofit a GPU into the completed A/B pilot. Instead, create a separate private GPU pilot in the next infrastructure verification stage.

Never send restricted MedVAL-Bench data to Kaggle or public GitHub without explicit authorization under its terms.
