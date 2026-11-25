# MASTER 01B — full-context private GPU leakage screen, protocol v1

Status: prospective automated screening protocol. No medical label adjudication,
final split approval, training, or scientific E1 effectiveness evaluation.

## Fixed inputs and compute
- MASTER 01 verified aggregate and decision gate: docs/E1_MASTER01_VERIFIED_RESULTS_AND_NEXT_GATE_2026-11-25.md.
- Reconstruct the exact 9,474-parent provisional mapping with source pins and prior digest. Official PubMedQA 500 remains outside all candidate partitions.
- Source inputs are the supplied questions and Knowledge/context passages (not necessarily entire articles). No holdout answer text or ground-truth labels are used in embedding or decisions.
- Fixed encoders: all-MiniLM-L6-v2 (prior revision) and NeuML/pubmedbert-base-embeddings (27cf1ec4e51deabc22b419742453d5bf4923e38c). Both publisher pages declare Apache 2.0; upstream rights must be reviewed independently. Pinned safetensors SHA256 values are verified before loading; config/tokenizer file hashes are recorded. No remote code or remote inference.
- Mean pooling with attention mask, L2 normalization, float32, deterministic PyTorch operations and fixed seed. One GPU, encoder loaded one at a time.
- Every tokenizer token in each question and context is covered by 256-token windows including model-specific special tokens; 64 payload-token overlap. Fail closed on empty text or >64 windows per parent, never silently truncate. This is coverage of the supplied context, not full article certification.
- Exhaustively compare every cross-parent pair in all six partition combinations, for both views and both encoders. Each parent-pair score is max cosine over all window pairs. Track exact parent and window comparisons, coverage counts and actual package versions/device. No ANN candidate cap. Max-window similarity has length and multiple-comparison bias; 0.80/0.85/0.90/0.95 are screening thresholds, not clinical equivalence.
- Union flags at >=0.80 across both encoders and views; count unique flagged pairs and affected parents (no row identifiers exported). The independent holdout is never changed. No automatic quarantine, relabeling or split changes. Reproducible sampling of up to 100 flagged and 100 unflagged pairs per partition pair, without replacement, for lexical Jaccard diagnostics. Samples are automated diagnostics only, not human adjudication or estimates of missed semantic duplicates. Sampling can be reconstructed privately; no persistent row-level review queue exported.

## Human scientific gate (separate)
Review the candidate union in an authorized private environment, including unflagged review sampling, with two independent qualified reviewers where feasible; predefine adjudication criteria and report agreement and uncertainty. External article provenance and clinical label adjudication remain separate. Benchmark role counts are not clinical correctness. Resolve true source conflicts with a predeclared group policy; keep the official holdout unchanged. Freeze manifests only after review. No E1 model training without later approval.

## Workflow
GitHub Actions: E1 MASTER 01B Full Context Private GPU Controller.
Push/PR executes synthetic tests only (CPU torch reference comparisons); push also builds code-only private package. Manual dispatch on main: preflight (read-only), prepare, submit, status, verify. Submit requires BOTH private dataset rights and one new GPU run confirmation. A distinct immutable slug prevents duplicate submission. GitHub secrets send controller/start/accepted/runtime-observed/verified-completion alerts, not Kaggle credentials. Completion is sent after the sanitized artifact is uploaded. The four-hour polling window is not an execution-time promise; if it expires use status or verify without resubmission. No automatic retry or checkpoint resume in this version. Actual quota balance and GPU availability must be checked in Kaggle. Only the fixed JSON aggregate is downloaded, validated and retained. No raw text, IDs, embeddings, assignments or logs uploaded. Runtime success is not scientific approval.

Model metadata sources:
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors
- https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/27cf1ec4e51deabc22b419742453d5bf4923e38c/model.safetensors
