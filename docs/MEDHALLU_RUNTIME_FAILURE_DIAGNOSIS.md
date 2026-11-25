# Private MedHallu audit failure: safe diagnostics and controlled retry

## Observed historical failure (25 Nov 2026)
The private Kaggle notebook `yeatasimbillah/medical-llm-medhallu-native-rows-v1` version 1 reported `error` in the live read-only GitHub check. The notebook log contained only `MEDHALLU_NATIVE_PRIVATE_AUDIT_FAILED_SAFE`, followed by unrelated `mistune` / `nbconvert` syntax warnings. **These warnings do not identify the scientific audit failure.** The original exception was intentionally hidden for privacy; **root cause was not recoverable from that log alone**.

## Reviewed defensive change
The runner now checks that Kaggle's persistent working directory exists and creates an ephemeral, non-symlink `/kaggle/temp` directory if it is not already present. A missing scratch directory is a plausible (not confirmed) contributor to rapid failure. No source dataset bytes may be written to persistent `/kaggle/working`.

The diagnostic runner prints **constant, non-sensitive stage names**:
- `PRIVATE_ENVIRONMENT_CHECK`, `PRIVATE_ENVIRONMENT_OK`
- `DEPENDENCIES`, `HUB_SNAPSHOT`
- `FILE_LAYOUT_AND_SCHEMA`, `NATIVE_ROW_AUDIT`
- `AGGREGATE_INTEGRITY`, `AGGREGATE_SAVE`

On error it prints only `MEDHALLU_NATIVE_AUDIT_FAILED_STAGE=<constant>` and `MEDHALLU_NATIVE_AUDIT_FAILED_CODE=<sanitized>`. Exceptions, source text, file contents, HTTP response bodies, remote error messages, tokens and traceback data are **never printed** by this handler. If the environment or remote network fails again, the stage narrows diagnosis without revealing private material.

## Retry procedure (never automatic)
1. Merge the reviewed fix only after GitHub CI passes. The push trigger executes **synthetic tests and a code-only build**, not Kaggle.
2. Retain the first error version. Existing `operation=submit` continues to **block** because the target slug is present.
3. Only with renewed explicit authorization for MedHallu private Kaggle processing and CPU use, run **MedHallu Private Native Row Audit Controller** on `main` with `operation=retry_failed`, `confirm_private_dataset_rights=true`, and `confirm_cpu_quota=true`. All those booleans must reflect a real rights review; checking the boxes is not a substitute for licensing review.
4. The GitHub workflow reads the live Kaggle status immediately before push and rejects anything other than **exactly `error`**. It also checks access to the Kaggle quota API, but cannot claim a precise remaining CPU quota. It then pushes one new private version of the same reviewed notebook; never auto-retries on another failure.
5. After completion, run `status`; only when Kaggle state is `complete`, run `verify` using the **new, exact GitHub source SHA printed by retry_failed**, not the failed-version SHA.
6. If Kaggle fails again, read only the constant `FAILED_STAGE`/`FAILED_CODE` (or a stage code from Kaggle logs); **do not copy raw data, actual notebook source rows, API keys or traceback** into chat.

This remains a private, CPU-only, rights-gated dataset structural audit, not the E1 hallucination detection baseline. Both scientific model execution and MedVAL transfers remain blocked.
