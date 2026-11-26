# MASTER 01F-F custody recovery v1

CPU-only offline, selected-only provisioning for the exploratory 256/128
cohort. No dataset acquisition, Kaggle submission, inference or test-answer
access. Public CI contains only invented fixtures and ephemeral synthetic keys.

## Trust boundary

The independently authorized data custodian must provide a physically isolated
covariate-only export and subsequently the requested selected-only answer files.
The historical aggregate does not authenticate row provenance. No mixed corpus
or test-answer columns may be mounted. Neither the GitHub connection nor a
researcher-generated self-signed receipt establishes independent provenance.
The custodian signs attestations outside this runner. The operator must verify
identity, authority, source rights and the SHA256 of the custodian's Ed25519 DER
public key through an independent trusted channel. The signature establishes
binding to that key, not the truth of the attested facts. Private key never
enters this runner. No automatic creation of real trust anchors.

The trust directory contains custodian.der (44-byte Ed25519 SubjectPublicKeyInfo),
and phase.json/phase.sig pairs (raw 64-byte detached Ed25519 signatures over the
exact JSON bytes). Use the common fields schema=1, phase, checkpoint_sha256 and
mapping_sha256 (historical pins). Additional exact fields:

* snapshot: snapshot_sha256, source_release_verified=true,
  covariates_only=true, processing_rights_verified=true.
* answers: snapshot_sha256, request_sha256, index_sha256,
  selected_only=true, processing_rights_verified=true.
* approval: snapshot_receipt_sha256, answers_receipt_sha256,
  bundle_manifest_sha256, approved_for_readonly_preflight=true.

The custodian must review the projected bundle before signing approval.
Do not reuse approval from another receipt pair. Store custody materials only
in a permitted private environment; no raw data or private receipts in Git.

## Operator sequence (only after authorized exports and verified environment)

Use an authenticated checkout of this recovery version, a trusted Python
interpreter and OpenSSL installation supporting Ed25519. Use offline,
CPU-only execution with CUDA_VISIBLE_DEVICES empty, HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1, no network access or GPU allocation. Environment
checks are not a sandbox. All input mounts must be operator-controlled,
read-only and disjoint, with no symlink/hardlink entries or concurrent writers.
Use separate owner-only directories (0700) for request and projected bundle.
Keep trust, covariates, answers, request and destination directories separate.

Invoke scripts/e1_master01f_f_custody_v1.py with operation request, provision
or preflight, --trust TRUST --custodian-key-sha256 independently-approved-pin
and --authorized-private-processing only after fresh rights approval.

1. request: --covariates COV --out REQUEST_DIR/request.json
   returns a fixed success marker; send request privately to custodian.
2. provision: --covariates COV --answers ANSWERS --request-file REQUEST_DIR/request.json
   --out BUNDLE_DIR/bundle. Verify signed index/request binding before opening
   selected answers. Existing 01F-E validation reconstructs historical mapping,
   selects deterministically and verifies exact membership and file hashes.
   This is not final approval.
3. preflight: after custodian signs approval, --bundle BUNDLE_DIR/bundle
   --code-root D_CHECKOUT --code-manifest CODE_HASH_DIR/d_source_hashes.json
   --assets ASSETS. Uses hardened 01F-F launcher, historical exact code pin
   and actual immutable D preflight. No selected answer contents opened.

The runner emits only fixed markers and does not print private IDs, hashes,
paths or exception text during gated operations. Input trust and infrastructure
must be established outside the runner. Neither synthetic signature checks
nor a synthetic projection establishes actual provenance, actual model asset
availability or actual private preflight success.
