# MedHallu / PubMedQA licensing and private-Kaggle audit review (2026-11-25)

## Official source evidence checked

| Source | Review finding | Source |
| --- | --- | --- |
| MedHallu dataset | Author-maintained Hugging Face card declares MIT and acknowledges PubMedQA derivation; structured Hub license field absent from our metadata endpoint response | https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu |
| MedHallu authors' code | Official MedHallu GitHub repository has a standalone MIT LICENSE | https://github.com/MedHallu/MedHallu/blob/main/LICENSE |
| PubMedQA original | Original PubMedQA repository contains a MIT LICENSE (copyright 2019 pubmedqa) | https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE |
| PubMed abstracts | NLM warns that journal publishers or authors may hold copyright in abstracts and that the user must determine copyright and permission terms | https://www.ncbi.nlm.nih.gov/home/about/policies/ |
| Specific MedHallu dataset revision | Public metadata-only live audit previously returned `515060458a945c633debc6fd5baac7764416b724`. This is a *repository revision SHA*, not downloaded-file SHA256 | https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu |

**Review decision:** The MedHallu and PubMedQA project-level MIT licenses are confirmed by their official public repositories. This does **not** independently establish universal rights to reproduce or redistribute every underlying PubMed abstract outside its original context. Treat transfer of underlying abstract text to a third-party cloud (including private Kaggle) as **conditional on the researcher's/institution's explicit rights review**. This document is not an authorization, legal opinion or research result.

## A private, limited, no-redistribution audit is prepared

The new code-only Kaggle workflow is:
[MedHallu Private Native Row Audit Controller](../.github/workflows/medhallu-private-native-row-audit.yml).

- On main-branch push, **only offline code-only preparation and fake-data tests run**, with no Kaggle API secret or dataset download.
- For a real private CPU row audit, manually run `preflight` (read-only). If allowed after rights review, `submit` requires BOTH `confirm_private_dataset_rights=true` and `confirm_cpu_quota=true`. The workflow refuses reusing an existing notebook slug.
- The Kaggle script is **private, CPU-only, GPU/TPU disabled**; internet is enabled only because the specific public Hugging Face dataset must be retrieved. No MedVAL, BioASQ, external LLM API, patient dataset, medical model or training runs are involved.
- The script uses the single frozen MedHallu revision above and downloads only the two `pqa_*/` Parquet configurations into **ephemeral `/kaggle/temp`**, not persistent `/kaggle/working`.
- Each downloaded Parquet file is byte-hashed with SHA-256 in Kaggle before reading; one shard per config, exact column names and declared 1000/9000 row counts are checked. Real source rows are never copied into GitHub Actions or GitHub artifacts.
- A strict private Python audit examines native column types, repeated question hashes, duplicate question-answer pairs and ground truth vs hallucinated answer equality. A sanitized **counts-only manifest** is the only output in `/kaggle/working`.
- After a Kaggle `complete` state, `verify` requires the **exact submitted 40-character Git SHA**. It independently checks aggregate integrity, source SHA, dataset revision, source file sizes/byte SHA256 records and all negative scientific-approval flags. Only the single sanitized aggregate file is allowed into the optional short-lived GitHub artifact.

## Important limitations

- The fact that an upstream dataset card says MIT does not eliminate inherited copyright or publisher permissions. The owner should obtain supervisor/institutional advice on *private cloud processing*, and retain needed copyright notices/attribution. Do not tick the rights attestation without this review.
- Downloaded-file hashes are verified *locally* against the actual Kaggle bytes but no independent publisher-provided SHA256 baseline is yet established. They are for repeatability and later cross-run checking, not proof of a publisher's original file authenticity.
- The original `pqa_labeled` and `pqa_artificial` are two **source configurations with only train splits**, not a scientifically frozen train/dev/test partition.
- This structural audit does not evaluate native hallucination category semantics, patient risk, clinical accuracy or independent PubMedQA source lineage; it cannot establish that claims are correct or that 3-way hallucination gold labels exist.
- Github/GitHub Actions workflow START or END is never a claim of actual Kaggle notebook completion. The separate `verify` step checks Kaggle output; completion notifications are not yet continuous for this research-specific job.
- Kaggle notebook storage/privacy terms, environment image and quota availability may change. A one-time permitted analysis is not an approval for future repeated automatic processing.
- Do not alter `configs/dataset-sources.json` or `configs/experiment-registry.json` from `not verified`/`planned` simply because the limited structural audit succeeded.

## Minimal operation sequence

1. **Prepare:** automatic on reviewed main push. Confirm tests and generated code-only package.
2. **Preflight:** GitHub Actions manual `preflight` checks Kaggle authentication, notebook existence and quota endpoint; no dataset access.
3. **Rights + CPU confirmation:** only when independently authorized, manual `submit` with both safeguards checked. Note the complete source Git SHA in logs.
4. **Status:** manual `status` if required. It never resubmits.
5. **Verify:** manual `verify`, supply exact submitted source Git SHA. It downloads only the aggregate output for validation; only then are audited counts reportable.

None of the five steps requires copying a script into Kaggle or running Python locally. This run is a prerequisite for, **not** the E1 baseline experiment.
