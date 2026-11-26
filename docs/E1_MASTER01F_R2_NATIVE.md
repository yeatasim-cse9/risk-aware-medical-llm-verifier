# MASTER 01F-R2 — native split metadata and private CPU transfer gate

Status: source candidate and infrastructure only. The metadata probe observed\nrevision 3500188ec0691c1d83e2738592ec01a16e7dbcb5 with five train, one val,\nand one test Parquet object (public CI run 45819728567). This observation\nwas initially rejected solely by the old filename allowlist. Supervisor/rights approval,
actual source-file bytes, cross-source group separation, row audit and
scientific execution remain unapproved.

## Source documentation

Lamarche et al., MedHal: A Synthetic Dataset for Medical Hallucination
Detection, AACL–IJCNLP 2026, pp. 4599–4619
https://aclanthology.org/2026.aacl-long.208/ (sections 3.1–3.4).
This supersedes the earlier author shorthand in the 01F-R proposal.
Dataset candidate: https://huggingface.co/datasets/LAMA-WeST/medhal-lf_qafil_bal
Publisher declares Apache-2.0. The paper describes binary factuality,
optional context, and explanations for some negative examples. A factual
statement may be supported by supplied context OR general knowledge.
Underlying labels are not clinical safety gold. Publisher expert checks
of a subset do not validate every row. The paper describes retaining
original train/test splits and deriving validation using stratification;
that is not a certificate of article/source-group separation.
The older author's documentation describes True=factual, False=nonfactual
(https://github.com/GM07/clinical-analysis); confirm mapping from the exact
release's authoritative documentation before fitting. Never infer polarity
from a few observed rows or use test labels to establish semantics.
The paper and live card may describe different variants/counts; do not copy
paper totals into a current revision manifest.

Prioritize SumPubMed-derived evidence-bearing summarization for primary
feasibility, subject to prospective supervisor approval. Exclude QA without
independently supplied evidence, and do not use target explanations as
features. Hold clinical-note-derived sources and unknown rights. SumPubMed
project declares MIT (https://github.com/vgupta123/sumpubmed); this does not
override individual article terms (https://pmc.ncbi.nlm.nih.gov/tools/textmining/).
Native train shards can contain mixed source families. Rights must cover
EVERY downloaded byte, including sources later excluded. If whole-shard
rights cannot be established, require a publisher-provided or otherwise
authorized physically isolated subset and a new reviewed acquisition plan.
Do not download full train and filter restricted rows afterward.
Source overlap with historical MedHallu/PubMedQA remains unqualified.

## Metadata probe

Public metadata only: GET fixed /api/datasets/{candidate}/revision/main then
GET /api/datasets/{candidate}/tree/{exact SHA}?recursive=true. No viewer,
rows API, dataset loading, file download, credentials or redirect. Require
public ungated candidate, declared Apache-2.0, complete conventional native
split filenames and LFS content hash/size metadata. The observed physical\nvalidation spelling is val (not validation). Only this exact physical convention\nis accepted; internal logical aliasing never changes the reviewed metadata\npin or any download path. Reject unknown files,
paths, pagination and ambiguous/aliased shards. If layout changes, review
and version before allowing. Only normalized public metadata and digest
may appear in CI logs. The probe may fail due to network/API/schema drift;
that is a BLOCKED metadata observation, not a failed scientific experiment.
Metadata matching names cannot establish actual row contents, publisher
honesty, source separation, processing rights or independent provenance.
Do not autoapprove the latest revision or use the digest printed by the
probe as its own independent review. Preserve the actual observed public
metadata in a reviewed pinned manifest before any transfer.

## Private CPU acquisition (not enabled in CI)

Separate operation acquire on an authenticated immutable source checkout,
trusted interpreter, private Kaggle CPU runtime, no GPU allocated.
No notebook submission controller is provided in this phase.
Operator must verify actual privacy, permissions, CPU quota, rights,
revision, native shard inventory and independent trusted review.
Use a fresh owner-only subdirectory of /kaggle/temp, not /kaggle/working.
The launcher checks CPU visibility, not actual accelerator allocation;
the operator must set no accelerator. Use exclusive control of the
filesystem and no concurrent writers. No shared cache, automatic retry,
snapshot_download, load_dataset, row loading, or validation/test resolve.

External review supplies normalized metadata JSON and its canonical SHA256
plus exact approval JSON and independently communicated canonical SHA256.
Approval exact fields: schema=1, repository, revision, metadata_sha256,
supervisor_approved=true, all_train_file_contents_rights_verified=true,
native_train_paths_reviewed=true, private_cpu_processing_approved=true,
protected_splits_excluded=true. A hash binds bytes; it does not attest the
truth of a claim. No actual approval/key is generated in code or tests.
If any prerequisite is missing, do not invoke acquire.

Future command shape:
CUDA_VISIBLE_DEVICES="" python scripts/e1_master01f_r2_native_v1.py acquire
  --metadata METADATA --revision REVIEWED_REV --metadata-sha256 REVIEWED_PIN
  --approval APPROVAL --approval-sha256 REVIEWED_APPROVAL_PIN
  --out /kaggle/temp/OWNER_ONLY_PARENT/train --authorized-private-cpu

Do not run this placeholder command. Exact train paths at pinned revision
only, HTTPS with bounded CDN redirects, no credentials, byte cap, streamed
size/SHA256 verification and local readback, exclusive atomic directory
publication, failure cleanup. No test or validation downloads. Unexpected
redirect hosts fail closed; CDN storage behavior and actual runtime have
not been qualified. Environment checks do not provide an OS sandbox.
Actual downloaded bytes are not decoded by this step. Separate train-only
row/schema/source/rights audit and grouping are needed before model use.
No raw rows, file paths containing identifiers or private results in public
GitHub artifacts. Preserve historical evidence, no GPU runs, no test-answer
access, no calibration or scientific claims in this phase.

## Observed immutable public metadata (not approval)

The successful strict observation at run 45819880190 / job 139909084221
recorded revision 3500188ec0691c1d83e2738592ec01a16e7dbcb5 and canonical digest
`1eb70ba0c1c1d06549d37c41596e5bdd19d0f2c1953493ebc9de2bb26c36f17a`. The exact normalized publisher inventory is archived in
`configs/master01fr2-medhal-public-metadata.json` for reproducible review.
These are publisher metadata hashes, not independently computed dataset
byte hashes; neither the archived manifest nor a regression test is a
supervisor authorization. Revalidate actual metadata and rights before
any acquisition. A changed release requires prospective review.
