# E1 MASTER 01 verified result and next scientific gate

Observation date: 2026-11-25. Status: real-data screening complete; scientific approval pending; E1 model experiment not executed.

## Evidence chain
- [GitHub run 45606355921](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45606355921): completed, success; remote monitoring/verification and sanitized artifact upload succeeded.
- Artifact ID 15018285961, name master01-verified-aggregate, archive 2433 bytes, available through 2026-12-09 11:42:17 UTC according to GitHub metadata.
- Downloaded ZIP SHA256: cdd2e3efd74108b011f46bfee870fa62201253675eceb6a16cbea9fea220736e (matches GitHub artifact digest).
- Sole member: master01_verified_aggregate.json. Independently recomputed canonical JSON integrity SHA256 (sorted keys, compact separators, ensure_ascii=True, omitting integrity_sha256): 362e75f2b7abdd3fa61ac4c7b1c9d0ea46afb713989c604f2d277154ee6bbcd8, matches embedded digest.
- The reduced artifact reports remote report SHA256 dfbad95bea3c4173a0e945f5637b36593855ba419abad65674662f0d66ccd7d2; full remote original was checked by controller, not independently downloaded in this review.
- Master runner commit 40f96e132d81a86e61978a29c8a7a3d5aa444fc7; source remediation commit c6e0f3637c9e7f753091e874f53a22ed2cc45f26; mapping digest 9fa4a8d68d695c20447f25cb26c246f9b35ff355955850f9b26acb79197e1e2c.
- Encoder sentence-transformers/all-MiniLM-L6-v2, revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf; seed 20261125. Generic encoder; not clinical equivalence model. Artifact does not preserve actual GPU model/runtime library versions; do not infer these.
- No medical text, row assignments or article IDs in this evidence record.

## Observed aggregate results
9,474 candidate parents: train 7,579, dev 947, test 948; 18,948 paired benchmark answer-role examples, not independently clinical labels. Original independent 500-record PubMedQA test is separately reserved. Exact normalized question and full context matches: zero in all six cross-partition comparisons.

| Partition pair | Pairs per view | Q >=.80 | Q >=.85 | Q >=.90 | C >=.80 | C >=.85 | C >=.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train/dev | 7177313 | 12 | 4 | 1 | 17 | 1 | 0 |
| train/test | 7184892 | 11 | 4 | 1 | 18 | 3 | 0 |
| train/official_holdout | 3789500 | 6 | 1 | 0 | 8 | 0 | 0 |
| dev/test | 897756 | 0 | 0 | 0 | 1 | 0 | 0 |
| dev/official_holdout | 473500 | 0 | 0 | 0 | 4 | 0 | 0 |
| test/official_holdout | 474000 | 0 | 0 | 0 | 1 | 0 | 0 |
| total | 19996961 | 29 | 9 | 2 | 49 | 4 | 0 |

Both views had zero pairs >=.95. Total 39,993,922 pair-view comparisons, not independent unique biomedical observations. Max question similarity train/test 0.9268395; max train/dev 0.9094676; train/official_holdout 0.8618726. Context maxima train/test 0.8781398, train/dev 0.8503482, train/official_holdout 0.8388023. Candidate pair overlap between the question and context views is unknown; do not add as unique cases.
Context inputs exceeding 256 tokens: train 6058/7579; dev 758/947; test 773/948; official holdout 358/500; combined 7947/9974. Zero question truncation.
Configuration counts: pqa_artificial train 7185/dev 890/test 899; pqa_labeled train 394/dev 57/test 49. Native category sorted count vectors (no category identity): artificial train [5472,879,793,41], dev [675,115,98,2], test [682,117,97,3]; labeled train [300,84,10], dev [47,8,2], test [33,14,2]. The fourth labeled native category is not observed in these revised partitions; sorted counts cannot establish its identity. Do not compare ranks as named categories across splits.

## Decision: additional MASTER 01B gate, not MASTER 02
The current candidate split is NOT approved. Similarity thresholds are screening heuristics, not evidence of duplicated medical facts or labels. No automatic split approval or training on provisional data.

Proposed private, GPU-assisted next master audit (requires reviewed code, rights and one-run authorization before compute):
1. Reconstruct and verify exact prior mapping and all immutable source pins. Keep original official holdout fixed; use only text/source identities for contamination screening, never its answers, labels or model scores for threshold tuning.
2. Cover entire abstract with token-aware overlapping windows, explicitly count windows and coverage and report any failure to cover text; use max window similarity for candidate screening with length/window-count bias disclosed. Keep fixed baseline encoder and independently pin/validate a domain-relevant second encoder if available and licensed; do not claim full semantic recall. Preserve all six cross-partition comparisons. Assess lexical duplicate evidence and source/article identity in the authorized environment. If using approximate retrieval, predefine and assess recall against exact checks. Record actual device, package versions, weights provenance and hashes.
3. Pre-register fixed screening and review sampling policy before further inspection; review union of flagged question/context pairs privately, plus prespecified sample of unflagged pairs to assess misses (report uncertainty). Distinguish same article/near duplicate from topic similarity and clinically different populations, interventions/outcomes. Blind qualified reviewers to model results and proposed split where feasible. Two independent reviewers and adjudication where feasible; report agreement with denominators and uncertainty. No row content/PMIDs/row assignments exported. Thresholds for additional windows are not clinical cutoffs; preserve uncertainty.
4. For adjudicated duplicates, use connected components; preserve original holdout, quarantine linked candidate records rather than alter holdout; resolve internal split conflicts by predeclared group policy, not model performance. Unresolved provenance/semantic candidates stay quarantined or keep approval blocked. Reassess split balance, group independence, and report disposition counts and new mapping digest only. Full article provenance and dataset rights require separate approval.
5. Define benchmark answer-role prediction as a distinct task from clinical correctness and evidence-grounded claim verification. A physician/qualified reviewer protocol is needed for any medical factuality ground-truth claim; assess label noise with blinded prespecified review, named native category definitions and strata in private, difficulty semantics, configuration-specific reporting, and sparse-category uncertainty. No fabricated expert labels or claim of physician adjudication. Do not merge categories silently.
6. Freeze private manifests and preregister metrics, model baselines, seeds, evaluation, patient/source-group clustered uncertainty where relevant, and test-use restrictions only after independent scientific review. E1 scientific training/inference requires explicit subsequent authorization; no MedVAL restricted dataset access in this workflow.

This is a proposed protocol for scientific review, not an implemented new audit or clinical approval. No new Kaggle execution authorized or submitted by this documentation change. Publication acceptance/rank is not guaranteed.
