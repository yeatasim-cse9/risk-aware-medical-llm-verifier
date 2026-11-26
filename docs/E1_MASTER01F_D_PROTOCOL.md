# MASTER 01F-D v1 — exploratory projected private pilot candidate

GitHub-only archive policy: public repository is code, documentation and invented
fixtures only. Original verified ZIPs, hashes, private aggregates and manifests
belong in a dedicated access-controlled private GitHub repository, not public
branches or public Actions artifacts. No Google Drive. Do not add raw medical rows
to Git unless rights and private storage policy separately permit it. A private
archive repository must be provisioned and access verified before upload.

## Execution boundary

This is a private Kaggle entrypoint, not a remote submission workflow. No new
dataset acquisition, model download, test or calibration access, network push,
automatic retry, or public artifact upload. The historical mixed-parquet loader
must NOT be reused. An independently authorized data custodian must supply a
projected input bundle containing ONLY the selected fit/selection answer pairs
in individually pinned JSON files, plus metadata-only retained train/dev
inventory. The custodian must authenticate the original source release, retained
mapping, historical checkpoint, joint source components and full exposure history
and independently communicate the manifest SHA256. Self-generated digest or
matching historical digest strings alone cannot establish true provenance.
This controller checks the authenticated manifest, not the truth of its claims.
No complete historical reconstruction in the pilot; no test/holdout answers may
be mounted. A safe producer for the projected bundle is still required.

Manifest schema: schema=1, mapping_sha256 and checkpoint_sha256 from prior
integrity module, exposure=ALL_RETAINED_TRAIN_DEV_PREVIOUSLY_LABEL_EXPOSED,
metadata={fit:[original train metadata],selection:[original dev metadata]},
selected_files={selected parent SHA256 + ".json": exact file SHA256}.
Each metadata row contains only parent, source_group and original partition.
Parent and source-group IDs follow the prior V4 SHA256 representation. Fixed
label-blind original seed/hash order, one parent per group, first 256 fit and
128 selection. Each selected JSON has exact prior pair schema with partition
renamed fit or selection. No extra file allowed. Metadata preflight opens
only manifest, stats selected files and hashes separate public model assets.
Actual answer bytes are opened only after explicit authorization and GPU smoke.
A malicious custodian could mislabel rows; provenance and segregation must
be enforced before mounting. No test answer access by this phase.

Private Kaggle requirements: immutable full git checkout of reviewed source
commit, independent manifest pin, read-only projected bundle and local six-file
pinned public model assets, Internet OFF, T4 x2, cuda:0, fresh session, private
visibility verified externally, external maximum 90-minute runtime. The entrypoint
checks source bytes against git, original dependencies against base commit,
exact runtime versions, GPU hardware, asset bytes, public synthetic GPU smoke,
both partition input audits before private inference, fixed 36,864 private
forward examples (18,874,368 padded positions), and cost reconciliation.
Python deadline and exclusive local attempt lock do not enforce global one-run
authorization or preempt all native calls. No new run instructions until the
bundle, trust anchor, private output retention and actual preflight are ready.

Preflight (future private CPU session, no GPU or answer reads):
set MASTER01FD_PRIVATE_ENVIRONMENT=YES, MASTER01FD_SOURCE_COMMIT to reviewed
immutable commit, MASTER01FD_CUSTODIAN_MANIFEST_SHA256 to independently
authenticated pin. Run scripts/e1_master01f_d_controller_v1.py preflight
--bundle <private projected directory> --assets <pinned model directory>.
Expected fixed marker MASTER01FD_READONLY_PREFLIGHT_PASSED_METADATA_ONLY.
Do not set authorization or use run before fresh explicit approval.
Public CI executes invented fixtures only, never this private preflight.

## Scientific design

All retained train/dev groups were previously label-exposed. This is explicitly
exploratory, not a clean holdout or calibration experiment. Use same 384 parents
for all comparable methods. 12 model configurations, full/removed/permuted
conditions where applicable, three removed-refitted controls, two raw NLI scores.
16 paired source-group contrasts, 1000 common bootstrap draws; AP, ranking,
Brier; unadjusted descriptive intervals. Exact normalized word-count tie subset
only if >=10 groups. No winner selection, clinical approval, or calibration.
Fixed schema validated on serialized output, no rows, heads, identifiers or raw
scores; private aggregate only, no public export. No final test.

## Independent cohort feasibility (public paper metadata only)

MedHallu derives from PubMedQA, so a mirror or repackaging does not establish
fresh provenance (https://aclanthology.org/2025.emnlp-main.143/).
MedHal is a candidate for a separate source/task audit; its heterogeneous
intrinsic/extrinsic synthetic detection tasks may not match paired answer-role
endpoint (https://aclanthology.org/2026.aacl-long.208/).
MEDHALU is a distinct healthcare response hallucination benchmark, with span
annotations; endpoint and source comparability need verification
(https://netsys.surrey.ac.uk/datasets/medhalu/).
Med-HALT is a distinct hallucination test suite, not a verified same-endpoint
calibration cohort (https://aclanthology.org/2023.conll-1.21/).
No rows or test answers downloaded. No candidate qualified. Before acquisition
verify rights, source identifiers, independent components, exact label semantics,
paired response/context availability and eligible count for prospective power.
Do not pool incompatible endpoints, invent labels or claim clinical validity.
