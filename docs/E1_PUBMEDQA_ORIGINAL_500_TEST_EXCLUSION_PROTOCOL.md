# E1 next gate — prevent original PubMedQA test contamination

**Date: 2026-11-25. State: remediation algorithm prepared, NOT run on real data in this version. E1–E7 SCIENTIFIC MODEL EXPERIMENTS REMAIN UNEXECUTED.**

## Verified-consistency input audit

The user provided the sanitized private-Kaggle `e1_pubmedqa_lineage_advanced_leakage_v1` JSON. Its canonical JSON SHA-256 (all fields other than `integrity_sha256`) recalculates to `da1ba059f139fecf8330bd6f43a4f3c55e52146ec9186e70f8ac086d3be2cc75`. This checks internal report consistency, **not independent retrieval of the original manual Kaggle output or cryptographic attestation**. Source byte/revision anchors are frozen in `configs/e1-pubmedqa-source-lineage-audit-evidence-2026-11-25.json`.

The report found 9,974 exact-one PMID matches with exact context agreement among 9,997 candidate MedHallu parent records, **23 ambiguous multiple-PMID matches**, and no unmatched questions. It showed **all 500 official original PubMedQA test IDs inside the prior provisional MedHallu candidate partitions**: **401 old train, 61 old dev, 38 old test**. These are **potential contamination for independently evaluating PubMedQA**; they do **not** prove actual model training, since no E1 model experiment has been executed. Source-PMID overlap across prior splits was zero for uniquely resolved matches, but 23 ambiguous cases remain.

A bounded advanced cross-partition near-text screen compared 23,949 candidate pairs, found zero high similarity flags, skipped six high-frequency token buckets, and did not perform complete pairwise or semantic-paraphrase screening. Therefore `zero flags` **is not** evidence of a fully leakage-free dataset.

## Remediation policy (provisional, private CPU)

The reviewed pure module `src/research_data/e1_pubmedqa_test_exclusion.py`:

1. Starts from the previous candidate split mapping with frozen seed `20261125`, rather than silently modifying the original source rows.
2. Reconstructs PubMedQA exact-question `pubid` matches from the two immutable PubMedQA source configurations. Any zero, multiple or context-inconsistent candidates are **quarantined**.
3. Keeps the official upstream **500 labeled test PMIDs completely out of every new E1 development partition**, protecting against later external PubMedQA evaluation contamination. The source IDs remain private.
4. Among remaining eligible rows, builds connected groups by **actual matched source PMID**, normalized exact question and normalized exact Knowledge to avoid source/article and exact-text proxies crossing candidate partitions.
5. Proposes a deterministic new 80/10/10 split **by source-group count**, not necessarily by row count. It returns the private assignments *in memory only* and an allowlisted aggregate with source/configuration balance and sorted category frequencies *without category names*.
6. Rejects any report that pretends medical binary labels, complete semantic nonoverlap or final split approval were scientifically established.
7. Preserves a reproducible aggregate assignment hash, but does not publish source PMIDs, medical text, or row-level split IDs.

**Derived expectation, NOT an observed result of this new code yet:** based on the prior report, excluding 500 uniquely matched original test rows and the 23 ambiguous rows from the 9,997 prior candidate parent rows should leave **9,474** parents before any further anomalies. Exact count and source-group split composition MUST be recomputed on private real rows, and the runner fails closed if expectations differ.

## Private Kaggle manual execution (Hybrid Workflow)

Use **PRIVATE** Kaggle notebook, **CPU**, **Internet On**, and confirmed rights to process MedHallu/PubMedQA material and PubMed abstracts on Kaggle. No local Python and no GPU are needed.

- Execute one SHA256-verified, exact-Git-commit-pinned one-cell loader for `scripts/e1_pubmedqa_test_exclusion_kaggle.py`.
- The runner retrieves the existing reviewed data source loader from the exact same commit; checks both MedHallu and PubMedQA pinned Parquet source hashes and the original PubMedQA 500 test-file SHA256; recomputes prior candidate assignment and source audit counts.
- It loads `src/research_data/e1_pubmedqa_test_exclusion.py` and verifies that exactly 500 official original test rows and 23 ambiguous rows are excluded and 9,474 candidate parents remain. If these checks fail, it publishes **no success artifact**.
- On success, it writes **ONLY** `/kaggle/working/e1_official_test_exclusion_candidate_splits_v1.json`. The real medical texts and private assignment stay in temporary storage and Python RAM, discarded after the run. No row-level artifacts go to GitHub, Telegram, or shared chat.
- To check the outcome, share only safe lines:
```
E1_REMEDIATION_SAFE_AGGREGATE_SAVED
E1_REMEDIATION_PREVIOUS_500_TEST_EXCLUDED=...
E1_REMEDIATION_AMBIGUOUS_PMIDS_QUARANTINED=...
E1_REMEDIATION_REMAINING_PARENTS=...
E1_REMEDIATION_FINAL_E1_STATUS=BLOCKED_PENDING_SCIENTIFIC_APPROVAL
```
On failure share only:
```
E1_REMEDIATION_FAILED_STAGE=...
E1_REMEDIATION_FAILED_CODE=...
```

**The saved aggregate is not a frozen private final split manifest.** Before E1 training/inference, an authorized reviewer must assess category/difficulty/configuration shift; possible paraphrase/source overlap beyond the exact questions; native `Ground Truth`/ `Hallucinated Answer` semantics; errors in source PubMedQA question matching; and legal rights. After these checks, an approved protocol must freeze source-group-aware IDs *privately* and preregister model/checkpoint/evaluation metrics.

## Research significance for thesis Results section

This observation belongs in **Dataset Validation/Threats to Validity**, not predictive performance tables. Avoid claiming leakage-free labels, any measured Macro-F1/AUROC or hallucination reduction while E1 inference/training has not run. The 401 overlap should be described specifically as **original PubMedQA heldout set overlap with the previous provisional MedHallu train partition**, not as an E1 model having seen those records.

Sources:
- https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- https://huggingface.co/datasets/qiaojin/PubMedQA
- https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/data/test_ground_truth.json
