# MASTER 01D V3 verified private audit — 2026-11-26

Status: PRIVATE_BENCHMARK_AUDIT_ONLY. The single authorized GPU audit completed;
no clinical approval, new annotation, final freeze or E1 effectiveness run.

## Evidence
- [Run 45777027665 (#4)](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45777027665), source c63169fc6e597b80d48fbcdaa01102c97abcddc8 (PR #25), both jobs successful; one submit, observed running and complete, strict validation, sanitized aggregate upload and Telegram delivery.
- Artifact 15096991851, master01dv3-verified-aggregate, 2111 bytes, expiry 2026-12-10T07:48:58Z.
- Downloaded ZIP SHA256 6ae3d0e47326dc52498e9b43b22fd1a96eb49aa15e45643b0da213db6fd64af5 independently recomputed and matched GitHub metadata. Sole member master01dv3_verified_aggregate.json.
- Canonical JSON SHA256 a8e159311f8829a5b5ece7e12e0ffd3a76812f20dd3f066c9d508bf23304ace8 independently recomputed (omit integrity_sha256, sorted keys, compact separators, ensure_ascii=True). Archive canonical-equivalent aggregate in configs; original ZIP remains distinct.
- Independently compared original downloaded MASTER 01C aggregate: all 24 rounded maxima identical, four historical package source hashes, five dependency versions, mapping hash and retained counts match. Tesla T4. This does not independently reperform private inference or establish identity of all original pair memberships.
- Historical mapping 025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb, retained 7427/890/887 candidate parents and official holdout 500. 18408 paired roles. All reported residual screened flags, source proxy conflicts and supplemental exact conflicts zero. Source-component counts equal retained counts per partition; independence and comprehensive leakage clearance not established. Prior 270 quarantine removals remain excluded.

## Actual train/dev shortcut diagnostics
| Partition | Parents | Ground-truth role longer | Hallucinated role longer | Length tie | Length-only paired ranking, half-credit ties |
|---|---:|---:|---:|---:|---:|
| Train | 7427 | 5465 | 1723 | 239 | 0.7519186751 |
| Dev | 890 | 667 | 206 | 17 | 0.7589887640 |

Zero identical normalized paired roles and zero exact context-copy matches in each of train/dev. Normalized whitespace token count is the length metric, not model tokenizer length. These descriptive paired ranking rates are not trained model accuracy, unpaired classification performance, clinical labels or test results. No test answer shortcut scoring. Substantial length signal means E1 must compare length-only, answer-only, style and evidence-removal controls, with locked test protocol and paired source-group inference. Do not remove examples after seeing model/test outcomes or claim evidence grounding merely from role discrimination.

## Actual calibration inventory
PubMedQA labeled nonholdout 500 candidates: all 500 excluded for connection to protected prior 01C originals/holdout under exact source/question/context proxies. Zero eligible representatives in train/tune/calibration/test; no hypothetical power scenario at n=0. This does not mean every pair is an exact duplicate or all other PubMedQA populations are unavailable. Under this prospective cohort rule, the particular nonholdout population is unusable. BioASQ not acquired. No borrowing official holdout, relaxing protected overlap, or pooling incompatible tasks.

## Next decision
Develop separate BioASQ release-specific rights and cohort inventory pathway (public metadata only until fresh authorization). Determine whether a genuinely disjoint reference-scored cohort is available, and whether a small fixed policy family is prospectively viable. In parallel prepare E1 matched-budget shortcut/evidence baselines and a benchmark-only locked analysis protocol. E1 detection and closed-QA intervention remain separate scientific tasks; calibration infeasibility blocks confirmatory intervention, not necessarily future separately approved detection. No new GPU job authorized. Existing gates, legacy human-review protocol, final freeze and E1 authorization unchanged. No physician certification, actual effectiveness evidence, or novelty certification.
