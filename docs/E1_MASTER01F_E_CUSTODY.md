# MASTER 01F-E v1 — private custody bridge and package qualification

Public code and synthetic fixtures only. No actual research results or private
hashes in this repository. Dedicated private GitHub evidence archive only;
never Google Drive or public Actions artifacts for results. Verify repository
visibility, ownership, access and authorized collaborators before archiving
original ZIP and receipt unchanged. Do not treat a public branch as private.

## Custodian boundary

The historical mixed parquet/official answer loaders are not used. This
implementation cannot safely derive a trusted projection from a corpus
containing test answer columns. It requires two independently pinned exports
from an authorized custodian. No new GPU, data acquisition, calibration, or
test-answer access in this phase.

1. Covariate-only directory with exactly covariates.json: schema=1,
   records containing all historical original splits, each record exactly
   key,source,question,context,config,category,difficulty; mapping containing
   original retained [config,index,partition] rows. All fields must be vetted
   upstream as covariates only, not answers embedded in text. The custodian
   verifies historical source releases and exposure inventory and supplies
   its SHA256 independently. Code validates mapping digest against original
   checkpoint pin, original/retained counts and computes source components
   jointly including excluded and holdout covariates. No test answers.
2. request(covariate_dir,pin) yields only private selected train/dev keys,
   one parent per component, 256/128 fixed original hash order. Custodian
   supplies physically isolated selected-only answer files (not a mixed
   parquet or full split export). Answer index has schema=1, snapshot_sha256,
   request_sha256 (canonical bytes of request), files map of parent.json
   to SHA256. Each answer file has exactly parent,key,partition,answers.
   Answer index digest must be authenticated independently.
3. provision(...,authorized=True) validates both pinned inputs, rejects any
   extra file/symlink, wrong membership, overlap or content mismatch, and
   writes a new private directory with 0600 files and 0700 directory. Caller
   exclusively controls parent directory. No overwrite, public logging or
   output of individual keys. It validates the produced bundle through the
   actual D loader. The returned manifest hash is not independent evidence
   of provenance: the custodian must review the bundle and approve its
   digest via a separate trusted channel. Do not place raw rows in Git.

## Package and read-only preflight

Use exact D commit 4ca7c69f7e00feea4d3d5dcd102c38f563abb64c in a separate
full checkout. E source is not copied into D checkout. CI independently
compares Git archive bytes and checked-out Git object bytes, produces
code-only SHA256 manifest. Review/authenticate manifest externally.
Call e1_master01f_e_package_v1.preflight(D_root, verified_code_hashes,
projected_bundle, pinned_model_assets, approved_projected_manifest_sha256)
from a private offline CPU environment. It checks D bytes before importing
D modules, disables GPU visibility, strips GPU authorization and invokes
only the original read-only operation. D preflight hashes model assets and
reads manifest/stat metadata only, not answer bytes. Requires actual
projected bundle, model assets, and trusted pins; no actual preflight
or experiment has been performed by this code. The launcher uses an
isolated subprocess but the operator must authenticate checkout and
execution environment before running any Python and maintain private
visibility. No remote submission workflow or GPU authorization.
