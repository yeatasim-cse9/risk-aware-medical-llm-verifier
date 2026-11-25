# E1 research readiness — first actionable gate

**State:** BLOCKED until independently approved evidence exists. This stage does not launch a scientific experiment.

## Why E1 is not launched yet
- The E1 registry still says `planned`; no execution authorization exists.
- MedHallu's recorded version is `revision-not-yet-pinned`; raw downloaded bytes have no independently confirmed SHA-256.
- The public dataset card describes **two dataset configurations** (`pqa_labeled` and `pqa_artificial`), each containing a `train` split. These are **not** independently validated train/dev/test partitions. The public card advertises 1,000 and 9,000 rows, which is background metadata, not a locally verified dataset.
- Native column types, real label values, question/PubMedQA lineage IDs, class semantics and potential leakage have not been audited from approved data rows.
- No approved model/evaluator protocol, seed-based split list, or reproducible baseline execution artifact exists.
- Restricted MedVAL data must remain outside GitHub Actions and Kaggle.

## Current safety action
Run **Actions → E1 MedHallu Readiness Audit - No Dataset or Kaggle Launch → Run workflow**. The workflow only reads the repository config and runs offline tests. Expected output is `E1_PREFLIGHT_BLOCKED_NO_SUBMISSION_NO_MEDICAL_DATA` and the named missing gates; GitHub job success does **not** mean E1 is ready or evaluated.

Code: `scripts/e1_readiness.py`; tests: `tests/test_e1_readiness.py`. The absence of `configs/e1-approved-protocol.json` is deliberate. Do **not** fabricate the approval file or flip flags to bypass the gate; the gate performs structural checks, not independent authentication of approvals.

## Scientific approvals needed before actual Kaggle E1
1. Inspect original MedHallu license and PubMedQA lineage/terms, obtain the exact official commit and verify downloaded bytes in an approved private environment.
2. Examine native row schema, label meanings, prevalence and provenance; design and document binary/multiclass mapping with adjudication.
3. Freeze non-overlapping train/dev/test IDs with lineage-aware group controls. Do not treat `pqa_labeled` versus `pqa_artificial` as train versus independent test.
4. Pin model/tokenizer revisions, downloaded-byte hashes and legal terms; define a genuine detection baseline, evaluation, seeds, macro F1/precision/recall and uncertainty methods. Ensure no gold-test tuning.
5. Have the actual dataset audit, license review and experiment protocol independently approved before writing the authorization config and running E1.

All future Kaggle outputs remain private and Telegram alerts contain only non-sensitive status. No efficacy or clinical-safety claim is supported by this preparation.
