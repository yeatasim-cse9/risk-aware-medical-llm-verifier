# E1 Source-Lineage Verification and Advanced Leakage Audit — Hybrid Kaggle v1

**Status: code and synthetic-only tests. Real Kaggle source matching NOT YET EXECUTED. E1 scientific experiment BLOCKED.**

## Why this is the next audit

MedHallu's native six fields have no PMID/article identifier. The source paper and dataset card say it is derived from PubMedQA, which *does* contain `pubid`, `question` and structured abstract `context` fields. Therefore an independent direct look-up by normalized exact question can establish **candidate** PubMedQA article links. Article provenance is not automatically verified by question equality, and all rows that match zero or multiple source PMIDs must be reported as unresolved.

Official references:
- MedHallu card: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- PubMedQA card: https://huggingface.co/datasets/qiaojin/PubMedQA
- Official PubMedQA repository: https://github.com/pubmedqa/pubmedqa
- Original 500 labeled test set: https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/data/test_ground_truth.json

## Immutable data and source anchors

The manual runner, `scripts/e1_pubmedqa_lineage_kaggle.py`, uses two HF repos:

1. **MedHallu** at SHA `515060458a945c633debc6fd5baac7764416b724` with both *already independently verified* Parquet SHA256s and byte lengths:
   - `pqa_labeled`: 1,000 rows; SHA256 `ac963f9aaeb163f494edd4dbfec04086c3e35c7cc55e3f9c8b1d7b31188fa34c`
   - `pqa_artificial`: 9,000 rows; SHA256 `70de8ce7f89a38314fa231a12e45168f3ee1aa591e8a38f6556b89aa8b1ca513`
2. **PubMedQA** at immutable Hugging Face conversion commit `323752ad243ce751ba75c9222d34a3130d080030`:
   - `pqa_labeled`: 1,000 rows; published Parquet SHA256 `3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c`
   - `pqa_artificial`: 211,269 rows; published Parquet SHA256 `cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05`
   - These two PubMedQA hashes are from the public HF pointer metadata, **not yet independently checked in Kaggle**. The runtime will fail closed if hashes/bytes or row counts differ.
3. **Original official labeled 500-test IDs** from upstream Git commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`, Git blob hash `825bf8f262c3cca0e3bd1c6f38eecf5e581b0659`. The file is checked for 500 numeric PMID keys and `yes/no/maybe` values; none of the IDs or decisions is exported.

This adds a sizable **233 MB PubMedQA source Parquet download**, requiring an approved private Kaggle CPU environment with adequate temporary storage and internet. No GPU or paid API use.

## Procedure after reviewer sign-off

1. Confirm the existing MedHallu authorization *also covers* accessing and using PubMedQA's abstract-derived contents on private Kaggle; MIT code/dataset tags do not settle third-party abstract copyright or benchmark restrictions.
2. Import the generated one-cell `E1_PubMedQA_Advanced_Lineage_Audit.ipynb` notebook into Kaggle. Set **Private**, Accelerator **None**, Internet **On**.
3. The cell must fetch the runner at the exact *merged* Git commit and verify its raw UTF-8 **SHA256** before executing (no mutable `main` URL).
4. Execute once. It downloads only pinned Parquet files under `/kaggle/temp`, recomputes the previously verified MedHallu candidate assignment SHA256, loads the original official PubMedQA 500 test-ID file, scans PubMedQA question/PMID/context, checks exact candidate question-to-PMID matches, detects source PMID crossing provisional partitions, and checks where original PubMedQA held-out IDs fall within our candidate train/dev/test.
5. A second bounded word-trigram and short-question char-fivegram screen checks suspicious cross-split near-text overlap at explicitly defined thresholds. It does not compute semantic embeddings or all-pairs comparisons. It reports a candidate-cap/blocked-common-gram flag if incomplete.
6. The only saved output is the aggregate `/kaggle/working/e1_pubmedqa_lineage_advanced_leakage_v1.json`. No PMID, biomedical text, question/answer fingerprints, internal matched rows, assignment indices or official gold test IDs may leave the private runtime.
7. Share **only** the safe stage/counter lines or the aggregate report for independent *consistency* verification. Never send raw Kaggle error tracebacks, token values, source passages or individual identifiers.

## Interpretation gates

- **Unique exact PubMedQA question → PMID** matching is evidence of a source link, *not* independent attestation that MedHallu copied the article context correctly.
- Exact context agreement is a stronger check, while `context_not_exact` is unresolved, not automatically wrong.
- Any source PMID appearing in two candidate splits is a direct **source-group leakage concern**. Do not proceed to E1 until grouped by real PMID and independently retested. Rows with unmatched/ambiguous PubMedQA provenance remain non-certified.
- Any originally held-out PubMedQA 500-test article landing in MedHallu `train` is flagged for benchmark-contamination review; the two tasks differ, so this observation is not by itself proof of an invalid MedHallu split, but it matters for any external PubMedQA evaluation.
- Even if there are zero exact source conflicts and zero n-gram flags, this is **not a complete semantic duplicate, paraphrase, or LLM pretraining contamination audit**.
- Prior provisional train/dev/test sizes and role labels are *not* altered by this diagnostic; no final manifest is created, and no E1 model training or inference takes place.

## Fail-closed status markers

On success:
```
E1_LINEAGE_AGGREGATE_SAVED_PRIVATE_CPU_ONLY
E1_LINEAGE_PUBMEDQA_MATCHED_PARENT_ROWS=<nonnegative integer>
E1_LINEAGE_OFFICIAL_TEST_TRAIN_OVERLAP=<nonnegative integer>
E1_LINEAGE_SOURCE_CROSS_SPLIT_PUBID_CONFLICTS=<nonnegative integer>
E1_LINEAGE_ADVANCED_NEAR_FLAGS=<nonnegative integer>
E1_LINEAGE_FINAL_E1_STATUS=BLOCKED_PENDING_INDEPENDENT_APPROVAL
```

On failure send **only**:
```
E1_LINEAGE_FAILED_STAGE=<safe stage>
E1_LINEAGE_FAILED_CODE=<safe code>
```

**Results Section note:** this is a dataset provenance/quality audit, not E1 predictive performance. No macro-F1, AUROC, AUPRC, calibration, clinical performance or hallucination-reduction metric can be claimed from it.
