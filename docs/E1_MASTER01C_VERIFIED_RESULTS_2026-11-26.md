# MASTER 01C verified automated audit — 2026-11-26

Status: PROVISIONAL_REMEDIATION_REVIEW_REQUIRED. Automated audit completed;
human scientific adjudication, clinical labels, final split freeze and E1
effectiveness evaluation remain unapproved/not performed.

## Evidence
- [GitHub Actions run 45766728678 (#9)](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45766728678) completed successfully; synthetic-and-package and remote jobs passed. Fresh authorization step, one submission, running/complete remote observations, strict aggregate validation, artifact upload and four operational Telegram delivery steps succeeded.
- Source Git commit b2dab6ab6cce514c59c5d5df98a83bc70f7355c4, merged PR #18.
- Artifact ID 15092732808, master01c-verified-aggregate, 3289 bytes, expiry 2026-12-10T06:30:05Z.
- Downloaded ZIP SHA256 0e7fdaad8cf81c38a2fe19c782790c7b726fee0965edaf96e375d343b50a5736 matches live GitHub metadata. Sole member master01c_verified_aggregate.json.
- Canonical JSON SHA256 83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d independently recomputed and matched embedded digest (omit integrity_sha256, sort keys, compact separators, ensure_ascii=True).
- Independently compared attached verified 01B artifact against new aggregate: previous evidence and mapping digests, all six union counts and four endpoint counts, model asset set digests, encoder aggregate digest after removing rounded maxima, device and dependency versions all match; all 24 reported rounded maxima identical (maximum absolute reported difference 0). This does not authenticate original pair membership or independently repeat inference.
- Source package hashes: core a30e9116430a568e8fdb3d2aaa06dd791225246fe8b73b7f580b9c010d7a9602; private adapter 6b3f764fa79c6f839cc6705c4cc05c3178d3c4432d52c762ecffa25a8c100cbc. Prior runner hashes unchanged. Dataset pins and source-module pins recorded in aggregate and checked by controller.
- Tesla T4; torch 2.11.0+cu130, transformers 5.20.2, huggingface_hub 1.33.0, numpy 2.1.3, pyarrow 23.0.1. Seed 20261125.
- Previous private mapping digest 9fa4a8d68d695c20447f25cb26c246f9b35ff355955850f9b26acb79197e1e2c. Provisional new mapping digest 025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb. No private manifest was exported; new digest is reported by the runtime, not independently recomputed from private rows by this review.

## Observed results
| Partition | Original parents | Quarantined | Retained | Retained paired examples |
|---|---:|---:|---:|---:|
| train | 7579 | 152 | 7427 | 14854 |
| dev | 947 | 57 | 890 | 1780 |
| test | 948 | 61 | 887 | 1774 |
| candidate total | 9474 | 270 | 9204 | 18408 |
| official holdout | 500 | 0 | 500 | not used |

270 candidate parents (2.85%) and 540 paired answer-role examples removed provisionally. Connected source-proxy closure required no additional removals beyond flagged endpoints. Reported residual screened flagged pairs and retained cross-partition source-proxy conflicts: zero. Holdout unchanged, including its 27 affected endpoints. No claim of complete semantic leakage clearance or confirmed leakage. Reconstructed review queue: 171 flagged + 600 uniform unflagged (100 per partition pair) = 771; queue was generated in memory and discarded, not retained for reviewers. No human judgment recorded.

| Partition | Configuration TV | Category TV | Difficulty TV |
|---|---:|---:|---:|
| train | .00066000 | .00196434 | .00060941 |
| dev | .00176311 | .00285704 | .00072138 |
| test | .00129983 | .00489608 | .00543243 |

TV = total variation distance using consistent private category identities. Overall source configuration totals: artificial 8974 -> 8714 (260 removed); labeled 500 -> 490 (10 removed). Removal rates differ by partition (train 2.01%, dev 6.02%, test 6.43%). Aggregate shifts are small but do not establish representativeness or absence of exclusion bias; rare category cells remain as small as two parents. Labeled test difficulty TV .02692141. The review sample is not proof of absence of rare false-negative semantic matches.

## Next gate
Arrange an institutionally authorized private reviewer environment and secure access/retention/custodian procedures before reconstructing private packets again. Use existing pinned adapter, verify new mapping and count/asset pins, preserve holdout and blind reviewers. Obtain two independent qualified reviewers and a qualified adjudicator; preregister clinical sampling allocations and uncertainty/analysis procedures before viewing labels or model results. Document independent approval of rights, source provenance, semantic disposition, clinical labels, locked analysis protocol and manifest. No automated freeze or E1 training. The workflow report lists all seven scientific gates as missing: execution attestation is not documented scientific rights approval. No additional GPU run authorized by this result; no restricted MedVAL data access. Do not resubmit completed slug. Evidence archive expires: retain approved sanitized copy separately.
