# MASTER 01D V2 — benchmark-only integrity and split-freeze preparation
2026-11-26. Computational protocol, not a clinical validation, signed
scientific approval, actual split freeze, or permission to access data.

## Historical evidence
Base PR #23 main 7ccadc9516f9b72b9d026a9518994cc8a127c78a.
V3 post-merge workflow 45774803078 and repository CI 45774802938 passed.
MASTER 01C run 45766728678, artifact 15092732808. The ZIP was downloaded
again and its SHA256 independently recomputed:
0e7fdaad8cf81c38a2fe19c782790c7b726fee0965edaf96e375d343b50a5736.
Canonical JSON digest verified:
83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d.
Archive expires Dec 10 2026; preserve canonical-equivalent sanitized JSON
at configs/master01c-verified-aggregate-2026-11-26.json (different formatting
from ZIP; canonical digest unchanged). No row data, IDs or answers included.
Original candidate split 7579/947/948; retained 7427/890/887, paired
examples 18408; holdout 500 unchanged. 270 quarantined. Historical
provisional mapping 025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb.
Zero residual screened flags is not zero true leakage. Same aggregate
does not prove individual pair identity. No new GPU or real-data job.

## A. Scope and governance
No new human annotators, clinical reviewers or adjudicators. Use native
benchmark roles/reference scores, not newly clinically verified labels.
Existing MASTER 01D human review protocol remains historical and unapproved.
Academic/institutional permissions, data licenses and authorized private
processing still apply. This preparation code does not authenticate rights,
provide cloud access or authorize a data job. No original readiness gates,
registry, restricted MedVAL rules or scientific approval flags changed.
E1 approval config deliberately absent. Future benchmark-only execution
gate requires separate prospective adoption and actual evidence; do not
simply populate the old clinical approval flags.

## B. Pinned candidate reconstruction
Verify the exact canonical historical aggregate and package code SHA256.
In a separately authorized private execution, reproduce source inputs
from pinned 01C routines, verify official test file bytes and ID set
without using labels for tuning, and reproduce encoder-screen aggregate
evidence with the original pins. Use historical quarantine with supplied
original normalizer, then separately check connected source/question/context
proxies with NFKC/casefold/word normalization across ALL original rows
(including excluded rows). Preserve holdout and exact retained mapping and
balance; no selective reinstatement, reassignment or removal after results.
New proxy conflict => block, prospectively investigate and version rather
than silently change mapping. Full semantic, source provenance and
pretraining contamination remain unverified. Original 01C used holdout
covariates for screening: disclose transductive screening.
The pure prepare_pinned_candidate function accepts private reconstructed
records, flags, prior screen aggregate components and official test bytes.
It checks consistency; cannot authenticate that externally supplied objects
originate from the same GPU run. Pinned acquisition/inference adapter and
private job controller still need implementation/authorization. The existing
01C adapter is not modified or run, and no reviewer-access attestation is
faked. Never send private outputs to logs, chat, GitHub or Telegram.

Train/dev-only role diagnostics accept only paired answers from retained
train/dev keys. Count identical normalized paired roles, relative token
length (whitespace after normalization), and exact context-copy matches.
These are shortcut diagnostics, not truth labels or a complete artifact
audit. No test/holdout answers allowed in this interface. Later pilot:
question-only, answer-only, style/lexical and evidence-removal controls;
prospective outcome thresholds and model protocol needed. Do not filter
test based on pilot or test results. Identical train/dev roles require
investigation before freeze; no automatic relabeling.

## C. Candidate closed-QA cohort
Preferred candidate BioASQ 14b yes/no only; exact release, rights and
eligible source count still unknown (V3 records different official totals).
PubMedQA yes/no/maybe and MedQA/MedMCQA MCQ are separate tasks, no pooling
as if iid. MEDEC-MS error labels are separate, MedVAL prohibited on Kaggle.
Rights/underlying content permissions and reference/evidence access
must be verified for each specific release before download. Source aliases
must be canonical and externally verified; unknown provenance excluded.
No new human reference labels; gold unavailable at feature extraction.

Given an authorized private metadata inventory, plan connects source aliases,
normalized questions/contexts, and supplied semantic screening links, including
all 01C originals and reserved holdout as protected covariates (not the
universe of previously excluded rows). Additional known exclusions and
training sources must be supplied as protected where applicable. Quarantine
candidate components intersecting prior protected data; remove development
members connected to official test, preserve original official test membership
and report excluded test subset. Select one deterministic prelabel representative
per component. Candidate development group allocation: approximate 50% train,
20% tune, 30% calibration using fixed seed 20261126 and stable group hashes;
no exact sample-size guarantee, no outcome-based balancing. This is proposed,
not an approved final assignment. Counts and exclusions reconcile; private
mapping digest reproducible. Need additional semantic search and independent
release provenance verification before treating the candidate as isolated.
Source grouping does not establish iid sampling or transportability.

For eligible calibration count n, report V2 hypothetical scenarios for K=1,3,12
policies, caps .10/.05/.50, favorable rates .03/.01 and tight .07/.03,
coverage .75, original correctness .70. If n>2000 only evaluate 2000 in
current numerical planning function; do not assert actual power. No policy
selected automatically; training/tune-only protocol selection and actual
pilot feasibility required before calibration labels. No eligible groups
means blocked; no borrowing reserved tests or mixed task labels. No public
release of raw mappings, source identifiers or small-cell details.

## D. Freeze preparation, not freeze
Independent reproducible mapping/hash and provenance, benchmark-role scope,
train/dev pilot, pinned models and locked analysis, approved private storage
and actual rights remain required. Persist immutable manifests only in an
authorized private store after separate approval. Current gate_report always
reports not approved, no clinical review, no final freeze or execution.
No approval file created. No actual model inference, source-group inventory,
clinical or scientific effectiveness results. Novelty and external generalization
unproven. Synthetic CI tests the code and historical sanitized checkpoint,
not real-data validation or clinical safety.
