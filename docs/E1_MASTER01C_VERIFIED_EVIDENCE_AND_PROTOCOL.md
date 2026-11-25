# MASTER 01B V2 evidence and MASTER 01C prospective protocol
Checkpoint: 2026-11-26. No new medical computation or scientific approval.

## Verified evidence
- Live GitHub run [45634101734](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45634101734): both jobs successful, including synthetic compatibility tests, runtime monitoring, aggregate verification and upload. Notification step success does not establish delivery.
- [PR #17](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/pull/17) merged; main at inspection 0b3785b8149c71feb8adabca0533f1c2fdbfd336.
- Artifact 15038313281, master01b-verified-aggregate, 4380 bytes, expires 2026-12-09 15:47:38 UTC.
- Attached archive independently hashed: eb9bb06d27458dfd3d56008e3c0c2c5cfdbac58fd918f7f7385c05cbf1df97d4; matches live GitHub digest.
- Canonical JSON SHA256 cac57aa852abc685f4125b276f6080cb5019bee73b5c5e97f93934c104ab321a; matches embedded integrity (remove integrity_sha256, sort keys, compact separators, ensure_ascii=True, UTF-8).
- Remote report digest declared by controller 69f78f506a6bc6c9684727dbd3e794aa0b640469709e635e269cc950d6825b3c; original remote report not independently retrieved in this inspection.
- MASTER 01 attached ZIP SHA256 cdd2e3efd74108b011f46bfee870fa62201253675eceb6a16cbea9fea220736e and canonical digest 362e75f2b7abdd3fa61ac4c7b1c9d0ea46afb713989c604f2d277154ee6bbcd8 both matched.
- Runner commit bdf92f6c940c8f40064d1f44f6bff5dda04dfd5e; recorded runner SHA256 89bdecce4109aa0cc4dc90f1f8a537ce085f13a707c9c0df9258d36625f62e2f; source commit c6e0f3637c9e7f753091e874f53a22ed2cc45f26; mapping SHA256 9fa4a8d68d695c20447f25cb26c246f9b35ff355955850f9b26acb79197e1e2c.
- Tesla T4; torch 2.11.0+cu130, transformers 5.20.2, huggingface_hub 1.33.0, numpy 2.1.3, pyarrow 23.0.1. Seed 20261125; 256 tokens including BERT framing, overlap 64.
- MiniLM revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf, weight SHA256 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db.
- PubMedBERT revision 27cf1ec4e51deabc22b419742453d5bf4923e38c, weight SHA256 929ddc16369bb4ff6f8e92a6ee9f5e0748f2bef33989b5617533dd50beebe4ef.

## Observed aggregate
9474 candidate parents, 18948 paired answer-role examples; separately reserved official 500. Zero exact normalized question/context cross-partition conflicts. Full supplied tokenizer coverage, not full articles. 166697875 window-pair comparisons, not independent observations.

| Encoder/view | Window comparisons | >=.80 | >=.85 | >=.90 | >=.95 |
|---|---:|---:|---:|---:|---:|
| MiniLM question | 19996961 | 29 | 9 | 2 | 0 |
| MiniLM context | 72065915 | 52 | 5 | 0 | 0 |
| PubMedBERT question | 19996961 | 26 | 5 | 0 | 0 |
| PubMedBERT context | 54638038 | 118 | 23 | 2 | 0 |

Union: train/dev 60, train/test 73, train/holdout 28, dev/test 3, dev/holdout 5, test/holdout 2 = 171. Affected train 152/7579, dev 57/947, test 61/948, holdout 27/500. All 171 flags and 600 unflagged sample pairs received automated lexical diagnostics, not human review. Matching aggregate counts cannot authenticate original membership. Max-window similarity has length and multiplicity bias; heuristic thresholds do not certify equivalence or complete semantic recall. No full article provenance, clinical gold labels, final approved split or E1-E7 effectiveness results.

## MASTER 01C prospective decisions, subject to scientific review
Reproduce unchanged pinned screening, mapping and asset hashes. Require exact integer coverage/count agreement and prospective absolute maxima tolerance 1e-5; block on discrepancies rather than tune thresholds. Record environment differences. Reconstructed membership is not independently authenticated.

Review all reconstructed flags and 100 uniformly sampled unflagged pairs per six strata, without replacement, separate seed 20261125+1000+stratum index. Expected queue if counts match: 771. New sample not asserted identical to prior automated lexical sample. Record population sizes and selection probabilities; small negative sample cannot rule out rare misses. Report stratum-specific uncertainty with source-group dependence, not naive pooling.

Two independent qualified reviewers blinded to flags, proposed splits, similarity scores and each other where feasible. Use question and evidence, not official holdout answers. Rubric: same_source (confirmed article identity); near_duplicate (equivalent information need/evidence); topic_only (related topic but distinct need); clinically_distinct (material PICO or context difference); unresolved (insufficient evidence). Third qualified adjudicator resolves disagreements and unresolved judgments, which may remain unresolved. Private rationale/source evidence. Report raw agreement, Cohen kappa and design/source-group-aware uncertainty; do not fabricate review.

Conservative option: remove all flagged candidate endpoints and connected PMID/question/context proxies; never change official 500. Keep paired roles together, no repartition or tuning. Before extra closure expected retained 7427/890/887 = 9204 parents, 18408 examples, not observed. Audit retained counts, residual flags, proxy independence, config/category/difficulty distributions and total variation by consistent private category identity. Sparse cells and exclusion bias remain. Sorted frequency ranks do not identify categories.

Separate clinical label review: prespecify paired candidate answer sampling stratified by configuration/category/difficulty/split, allocation and qualifications before review. Reviewers blinded to role and each other assess claims against supplied evidence: supported, contradicted, insufficient evidence, ambiguous, not applicable. Record evidence/rationale and clinical relevance; third adjudicator, agreement and uncertainty. No official holdout answers for tuning. Benchmark role prediction is distinct from clinical factuality.

Freeze gates: documented rights, source provenance, semantic disposition, clinical label review, independent scientific approval, locked analysis protocol, private manifest review. Structural validation cannot authenticate approvals. No automatic final freeze or training. No public row IDs, text, PMIDs, embeddings, review pair IDs or secrets. Agree authorized reviewer environment, retention and access controls before writing private review materials. Private Kaggle alone is not rights proof. Never transfer restricted MedVAL data.

This initial branch implements in-memory primitives and synthetic tests. Private reconstruction/encoder adapter, reviewer-facing access, strict aggregate validator, pinned packaging and Kaggle lifecycle controller are still required. No real-data submission, automatic retry or checkpoint/resume.
