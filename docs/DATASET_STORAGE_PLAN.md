# Dataset storage and acquisition policy

**Official thesis:** Risk-Aware Evidence-Grounded Verification for Detecting and Reducing Hallucinations in Medical LLM Responses.

**This repo stores acquisition code, explicit dataset manifests, provenance and allowed aggregate metrics — NOT raw datasets.** Never create a public mirror of benchmark data merely because a dataset is downloadable.

## Storage decisions

| Resource | Role | Storage | Acquisition | Hard boundary |
|---|---|---|---|---|
| MedHallu | E1 train/development | Kaggle private dataset (after licensing review), optional local cache | Pin the UTAustin-AIHealth/MedHallu Hugging Face revision, audit schema and splits | Keep final held-out split frozen; do not treat answer-level labels as claim-level NLI gold |
| BioASQ 14b | E5/E6 external generalization | Kaggle private dataset only if competition/redistribution rules permit; otherwise approved local or authenticated retrieval | Register for official BioASQ Participant Area and record exact 14b files | Freeze test; do not tune on final gold examples or publicly redistribute without checking |
| MedVAL-Bench v1.0.1 | E2/E3/E4 physician-gold evaluation | Approved restricted environment ONLY, separately encrypted where required | User-completed PhysioNet credentialing, required CITI training and DUA | No GitHub, Kaggle, cloud API, Telegram, or public outputs unless explicit approved authorization; physician error/risk labels are not automatic three-class claim gold |
| PubMed 2026 | Biomedical evidence corpus | Filtered/versioned local or permitted private Kaggle indices; metadata in Git | Ingest selected baseline XML files from NLM, validate MD5, extract relevant PMID / abstracts; update only under frozen corpus version | PubMed baseline = citation/abstract records, not all article full text; respect NLM/third-party copyright |
| DailyMed | Drug evidence | Selected/versioned SPL label cache in permitted private Kaggle inputs; metadata in Git | Use DailyMed API for specific drug labels first, or partial bulk SPL release after capacity test | Keep SETID, SPL version, effective/update date, source URL and checksums; respect download policy |

## Fastest safe approach

1. **Do not download every dataset yet.** Record access rights, schema, size, source URL, revision and expected manifest fields.
2. Run a tiny **MedHallu metadata/audit pilot** on a fixed, version-pinned source; derive reproducible development splits without inspecting final gold.
3. Download source data directly from its authorized endpoint **into Kaggle/cloud** when permitted, avoiding Windows download/re-upload. Use a Kaggle **private dataset** for repeatable inputs.
4. For retrieval, start with **small, declared and frozen** PubMed and DailyMed evidence subsets; index BM25 on CPU before dense/hybrid indexing.
5. Persist acquired file hashes, source IDs, license/access notes, split-hash, and document IDs. Pin private Kaggle dataset version, not just an editable slug.
6. Use **separate stable inputs** and **short-lived notebook outputs**: outputs contain checkpoints and permitted predictions, never restricted MedVAL.
7. Never upload raw protected content to Telegram or GitHub Actions artifacts. Use status counts and non-sensitive aggregate metrics only.

## Size, quota and licensing checks

- 2026 PubMed baseline: NLM published files pubmed26n0001–pubmed26n1334, with companion MD5 checksums. It is not a suitable first full-cache Kaggle job.
- DailyMed prescription/OTC bulk releases are split into multiple parts, rather than one conveniently small ZIP.
- Kaggle per-account quotas and private dataset limits vary. Verify **current account** quotas, storage consumed, notebook output constraints and source terms before ingestion.
- Git LFS does not solve the restricted-data license problem and is not the default storage for these corpora.
- GitHub Actions artifacts are time-limited build outputs, not a research dataset registry.

## Reproducible manifest requirements

Every acquired dataset/index must register:
`source_url`, `source_revision`, `license_review_status`, `access_scope`,
`file_sha256`, `downloaded_at_utc`, `preprocessing_git_sha`,
`split_manifest_sha256`, `evidence_corpus_version`, and `kaggle_dataset_version`
(where relevant). Unknown values remain null until verified — never fabricate hashes.

## Verified official starting points

- MedHallu canonical dataset: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- BioASQ participant registration: https://participants-area.bioasq.org/general_information/general_information_registration/
- BioASQ 14b: https://participants-area.bioasq.org/general_information/Task14b/
- MedVAL 1.0.1: https://physionet.org/content/medval-bench/1.0.1/
- PubMed 2026: https://www.nlm.nih.gov/pubs/techbull/jf26/jf26_PubMed_2026_BaselineRelease.html
- DailyMed resources: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
- Kaggle private datasets: https://www.kaggle.com/docs/datasets

**Current status:** Dataset downloads, permissions, license compatibility, final frozen splits, and all real model evaluations remain NOT STARTED. This is a planning and safety document, not evidence of acquired data.
