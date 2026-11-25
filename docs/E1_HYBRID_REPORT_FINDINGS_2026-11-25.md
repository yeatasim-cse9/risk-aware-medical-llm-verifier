# E1 Hybrid Kaggle Manual Audit — Verified-Consistency Findings and Scientific Gates

**Date:** 2026-11-25. **Status:** manual private-data diagnostic report checked for internal integrity; **NO E1 model experiment or approved test partition**.

## Evidence provenance and what was actually checked

The user supplied the sanitized JSON written by `scripts/e1_hybrid_private_label_lineage.py` from a private Kaggle manual run. Recomputing SHA256 of its canonical JSON *without* the `integrity_sha256` field yields the reported hash:

`b060a7a3f7bc8c449f36961bbdee10b12545feaf9f502b89edf0dc49369b7c3e`

Exact Git source commit `5e5b08d8f3818d4557051db9f5badadebd69ea78` exists and the two reviewed repository modules' UTF-8 byte SHA256s match those in the report. Dataset revision, both frozen Parquet SHA256s, candidate seed, previous assignment digest and all structural counts match the earlier independently verified aggregate records.

**Boundary:** This verifies report *internal integrity and consistency* with reviewed source code and frozen audit metadata. The manual Kaggle output was supplied as user text, not independently retrieved from Kaggle; an unkeyed JSON SHA256 is not a trusted signature proving execution provenance. Do not label this independent Kaggle result verification or scientific quality assurance.

## Observed aggregate counts (source text and IDs not exported)

- Native MedHallu: 10,000 rows (9,000 `pqa_artificial`, 1,000 `pqa_labeled`).
- Candidate design: 9,997 eligible parent rows and 19,994 answer-role examples, 3 quarantined parent rows (2 repeated question-group rows and 1 equal-reference/hallucinated-answer role collision).
- Provisional parent split: 7,998 train, 1,000 dev, 999 test. **Candidate only**, not source-lineage-approved final splits.
- Category values: 4 distinct values reported within **each configuration**, but raw category names are deliberately not in the JSON. Sorted frequencies artificial = [6848, 1096, 1010, 46], labeled = [752, 212, 33, 3]. These frequencies show strong imbalance and must be interpreted separately by configuration. **Cannot assume highest-frequency category in both configurations has the same text or clinical meaning**.
- Difficulty values: 3 distinct per configuration, artificial = [3182, 2926, 2892], labeled = [408, 318, 274]. These values are frequency totals, not verified disease severity or risk tiers.

## Bounded near-duplicate screening result

- 9,997 eligible parent question rows screened.
- 1,223 **cross-partition candidate pairs compared** by rare five-word-gram indexing and token-trigram similarity threshold at least 0.80.
- **0 pairs flagged** among that searched candidate set.
- 3 parent questions were too short for the five-word candidate indexing.
- 0 high-frequency buckets excluded, and candidate-pair cap not reached.
- The search **does not** inspect all possible pairs; it cannot establish no semantic duplicates, no shared article lineage, no pretraining contamination or leakage-free splits. The native six columns do **not** directly include a PubMedQA PMID/article ID.

## Prioritized next research steps

1. **Source-lineage evidence gate.** Investigate original PubMedQA article/PMID linkage inside an authorized private workspace, with a documented resolution rate and exact/near matching uncertainty. If original source IDs cannot be verified, explicitly state the limitation and avoid an independently leakage-free scientific claim.
2. **Private anomaly adjudication.** Review the duplicate-question group and same-answer collision privately. Do not export those question/answer texts to GitHub or Telegram. Document how many rows stay excluded vs are corrected under a reviewed protocol.
3. **Category and role semantics gate.** Inspect the four native categories per configuration privately; verify which category counts correspond to which meaning and whether answer roles are clinically adjudicated or only benchmark-provided. Avoid treating `Category of Hallucination` as the study's three claim-level evidence labels.
4. **Improved split leakage/bias assessment.** Evaluate source/configuration class prevalence by candidate split, detect exact shared evidence/article fingerprints and near/semantic duplicate risks more comprehensively; check source/configuration shift. Do not inspect gold-test performance during design.
5. **Approved split freeze and E1 protocol.** Only after lineage, rights and labeling review, freeze *private* split IDs/manifests, pin a model checkpoint/tokenizer/environment, predefine baseline metrics and confidence intervals, then seek independent approval to launch E1 inference/training.

## No invalid performance claim

This file does not report macro-F1, AUROC, hallucination reduction, severity-weighted errors or any other model performance. **E1–E7 scientific experiments remain unexecuted.** The label/lineage screen is a research-data audit, not a baseline experimental result.

For a sanitized machine-readable evidence manifest, see `configs/e1-hybrid-audit-evidence-2026-11-25.json`.
