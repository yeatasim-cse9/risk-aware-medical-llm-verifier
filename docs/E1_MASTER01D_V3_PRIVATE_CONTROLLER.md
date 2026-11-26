# MASTER 01D V3 — private reconstruction and calibration inventory controller
2026-11-26. Base PR #24: 6cf66d731fb9e3852e72ea1e1c2a623d2b04299c.
Engineering preparation only; no new real-data submission by this change.

## Scope
Reconstruct the pinned historical MedHallu/PubMedQA audit in a NEW private
Kaggle GPU script using unchanged 01C source routines. Verify the historical
canonical checkpoint, package source hashes, previous encoder aggregates,
quarantine counts, provisional mapping, distribution balance and official
500 holdout identity. Check exact runtime dependency versions, GPU presence
and synthetic model/window smoke before medical data acquisition. A version
change, numerical drift, extra exact proxy conflict or mismatch blocks the
audit rather than silently changing the protocol. The GPU requirement and
quota are distinct: read-only preflight checks the quota endpoint, not the
numeric remaining GPU hours. Actual GPU/model smoke occurs in the separately
authorized job before data access. No new human annotation or reviewer
access attestation. No E1 model training, NLI inference, or clinical claim.

The original 9474 candidate parents and 500 holdout covariates are
reconstructed; 270 quarantined remain excluded, retained 7427/890/887.
Source-group counts are proxy components, not proven independent groups.
Train/dev-only answer diagnostics use native benchmark roles, not gold
clinical truth. The pinned native parquet contains test answer fields
and the historical reconstruction validates nonempty paired role schema;
no test answer shortcut scoring or tuning. The official ground-truth file
is read for hash/ID checks; no holdout answers are used for tuning.

PubMedQA inventory: use the already pinned labeled 1000 source rows,
exclude the official 500, and inspect 500 nonholdout candidates for
native reference availability and source/evidence metadata. Protect all
01C originals (including quarantine) plus holdout; apply exact connected
source/question/context proxies and deterministic group representative
allocation. This is an exact-proxy feasibility inventory, not a full new
semantic screen or an independent calibration cohort. It may yield zero
eligible groups. No data acquisition from BioASQ, MedQA, MedMCQA, MEDEC,
or restricted MedVAL. BioASQ needs a separately authorized release,
rights and a future adapter. No new mappings or identifiers exported.
Report hypothetical power only at observed candidate calibration count.
No policy choice, clinical approval, split freeze or E1 gate modification.

## Manual operation (only main, correct repo)
Workflow: E1 MASTER 01D V3 Private GPU Controller.
1. Dispatch preflight with both confirmations false, expected commit blank.
   No data download or GPU submission. Verify marker
   MASTER01DV3_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED.
2. Independently confirm permission to process pinned MedHallu/PubMedQA,
   inherited abstracts and native labels privately on Kaggle; check
   actual GPU quota; explicitly authorize one new audit. Only then
   dispatch submit once with both boxes true. Do NOT reuse prior consent.
   No rerun of a submit workflow; run attempt must be 1. Concurrency and
   exact new slug absence are checked immediately before push. These
   guards are not a distributed transaction against manual external pushes.
3. Expected Telegram messages, with prefix E1 MASTER 01D V3:
   controller started; submission accepted; Kaggle runtime RUNNING observed
   (if observed during polling); Kaggle COMPLETE | aggregate independently
   verified | scientific approval pending. Telegram failure does not imply
   failed computation. Completion message follows aggregate upload.
4. If submission fails/uncertain or monitoring times out, do not retry,
   delete, replace or rename the slug to bypass duplicate checks. Use
   status or verify; verify requires exact original submission commit.
   No automatic retries. Controller timeout 240 minutes, workflow 270.
   New slug: yeatasimbillah/medical-llm-e1-master01d-v3-gpu-v1.
5. Send only completion marker or sanitized aggregate to research assistant.
   Independently check artifact and hashes, interpretation and next gates.

Code-only package (prepare) is explicitly non-executable. Authorized package
is built only after fresh checkboxes and first-attempt checks; never uploaded
to public artifacts. No raw logs, private rows, source IDs, review packets,
answers, credentials or mappings are exported. Private code/data reside
in temporary /kaggle/temp directory and cleanup is attempted on normal
exit or exception; forced termination may prevent cleanup. Runtime raw
stdout/stderr is suppressed during private processing; only fixed safe
error codes/markers emitted. Public Actions artifact includes only validated
aggregate, with strict schemas, bounded counts, pinned digests, no arbitrary
strings, and independently recomputed hypothetical power. Hashes validate
integrity and expected source, not scientific truth or authorization.
Synthetic tests are not evidence of live Kaggle compatibility, GPU
availability or model performance. Existing E1 readiness remains blocked.
