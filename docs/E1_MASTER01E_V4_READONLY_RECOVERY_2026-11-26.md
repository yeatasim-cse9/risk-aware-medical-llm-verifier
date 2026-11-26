# MASTER 01E V4 — completed runtime, read-only recovery (2026-11-26)

Original run 45797892397, remote job 139833296005, submission commit
03b691dde1081d925bf73b705a91b1557a6ebbf7. Kaggle API status complete
twice in controller log; then MASTER01EV4_CONTROLLER_FAILED=monitor |
RECOVERY_SCHEMA. User-supplied runtime log reached all 32 stages,
including train/dev input audits, inference, head fit, metrics,
bootstrap, aggregate validation/write, and COMPLETE at 1476.6 s.
The outer writer omitted input_budget_audit_passed and
actual_budget_reconciled, while the verifier required both. The original
wrapper success regression checked the inner fixture only; it did not
round-trip the emitted outer envelope into the verifier. This is a
controller schema defect, not a demonstrated failure of neural execution.
Original aggregate contents still require independent validation.

Read-only recovery workflow has no submission or GPU path. It checks
out the exact original producer, downloads only the original aggregate
from the completed V4 notebook, validates its exact six-key writer
schema, strict JSON, original envelope digest and original full inner
validator against pinned source, historical checkpoint and model
manifest. It preserves original bytes unchanged; the separate receipt
records the original digest and absent flags, not fabricated flags.
The runtime stages support the audit/reconciliation inference, but the
receipt does not claim those fields existed in the original result.
No scientific approval or test-set evaluation. Original notebooks,
controller and failed GitHub run remain untouched. The old writer and
verifier mismatch remains historical; never rerun this submission.

After CI passes and merge, use Actions > MASTER 01E V4 Read-only Result
Recovery > Run workflow > main. No inputs. This retrieves and verifies
the existing aggregate only. Expected marker:
MASTER01EV4_READONLY_ORIGINAL_AGGREGATE_VERIFIED.
Artifact master01ev4-readonly-verified-original contains original
master01ev4_private_aggregate.json and verification_receipt.json.
If verification fails, do not rerun GPU; inspect only sanitized evidence.
