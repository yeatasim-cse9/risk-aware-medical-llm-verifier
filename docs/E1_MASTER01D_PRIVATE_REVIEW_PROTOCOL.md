# MASTER 01D — private scientific adjudication and final split freeze preparation

Status: proposed scientific protocol and software preparation; **no reviewer
environment, reviewer identities, signatures, additional GPU run, human
adjudication, clinical certification, final freeze or E1 experiment approved**.

## Evidence checkpoint (not a new experiment)
- [Verified MASTER 01C](E1_MASTER01C_VERIFIED_RESULTS_2026-11-26.md), run 45766728678,
  artifact 15092732808, canonical SHA256
  83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d.
- Provisional manifest digest
  025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb.
  7427/890/887 retained candidate parents; 18408 paired answers; official 500
  unchanged. 270 quarantined; zero remaining screened pairs and source-proxy
  conflicts do not prove absence of semantic leakage. All seven scientific
  approval gates remain missing.
- PR #19 merged at a7506f1b0864a9d2edfc3817b872641e28356420. No restricted
  MedVAL data may be used in this phase or transferred to Kaggle/GitHub/chat.

## A. Institutional prerequisites — before reconstruction
Supervisor/institution approves the actual processing rights (including
underlying abstract rights, reviewer access and retention), qualified panel,
private environment, data custodian, review plan and a fresh one-run GPU
authorization. Prior 01C authorization was consumed. Do not submit the
completed 01C slug or reuse a prior authorization. Kaggle privacy alone
does not authorize human reviewer access.

The institution chooses a compliant access-controlled private environment
with individual authenticated accounts, least privilege, encrypted storage
and transport, access and change audit trails, secure backup/deletion,
egress restrictions and explicit retention/revocation rules. Configure these
outside GitHub. An approved runtime may use the pinned private GPU adapter;
the review service may be a different authorized environment, but no transfer
is permitted without explicit authorization. No public notebook, Drive link,
chat upload, Telegram content or GitHub issue may carry rows or review IDs.
No user-local Python is required. The repository does not provision that
environment, implement its access control or provide a reviewer UI.

A custodian provisions an independently authenticated trust registry (Ed25519
public keys and qualified institutional identities) and conveys its SHA256
over an authenticated out-of-band channel. Private signing keys stay with
their owners; never commit keys, signatures, receipts or reviewer identifiers.
A valid signature proves key possession only, not qualification or truth.
Two distinct qualified reviewers and a third independent adjudicator per
task; same people may serve both tasks if qualified. Independent scientific
approver is separate from reviewers, custodian and scientific lead. Custodian
does not review. Conflicts of interest are disclosed. Institutional
authentication and externally enforced one-time authorization consumption
are mandatory; code cannot enforce cross-process single use.

## B. Reconstruction and semantic review
Use exact source hashes from verified 01C and unchanged 01B screening
protocol. Require old mapping, reconstructed retained mapping, counts, pins
and 771 queue entries to match. No claim of authentic original pair IDs.
Review all 171 reconstructed flagged pairs plus the fixed 600 unflagged
sample (100 uniform without replacement per partition-pair stratum). Blinded
packets expose question, evidence and source identity only; hide split,
flag, score, paired roles and other reviewers' answers. Custodian keeps
private key mapping; a random >=32-byte nonce binds session and IDs.

Rubric: same_source (confirmed shared article), near_duplicate (equivalent
information need/evidence), topic_only, clinically_distinct (material PICO
or clinical context difference), unresolved. Require evidence locator and
rationale for every judgment. Source provenance review must separately
assess article identities, inherited abstract provenance/permissions and
context sufficiency, record coverage and limitations; proxy linkage alone
is not full article certification. No claim of exhaustive provenance
verification without corresponding evidence.

Independent initial reviews locked before adjudication. Third reviewer
adjudicates all disagreements and unresolved judgments. Flagged candidate
endpoints remain quarantined even if judged nonduplicates: no selective
reinstatement or outcome-driven split changes. Any confirmed or unresolved
unflagged possible duplicate triggers investigation and a new prespecified
remediation version; no silent override of this mapping. Official holdout
never changes; official answers/labels never enter review or tuning.

Report flagged census dispositions and raw agreement, separate unflagged
strata denominators, confirmed positives and unresolved cases, conservative
upper bound including unresolved. Exact finite-population hypergeometric
intervals (Bonferroni across six strata per endpoint); no IID pair
assumption, no global claim of complete clearance. Small samples give
wide bounds even if no misses found. Kappa is descriptive on reviewed
items, not an unweighted estimator for the entire pair population.

## C. Clinical validation — proposed allocation, requires sign-off
Target population is the **retained 9204 parents**, not excluded candidates
or official holdout. Stratify by split, native source configuration,
category and difficulty. Sample min(N,12) parents without replacement per
observed stratum (seed 20261125+2000+ordered stratum index); census strata
with <=12 parents. Actual sample count unknown until private reconstruction.
This cap is a proposed resource-conscious design, not a power-based
guarantee or approved sample. Before seeing judgments, supervisor may
prospectively amend and re-version the design based on target precision
and reviewer capacity. No selective expansion after seeing results.
Both benchmark answer roles for every sampled parent; blinded individual
answer packets; separate role key with custodian. No final-test clinical
judgments or metrics shared with model developers before analysis lock.
No use of official holdout answers/labels.

Qualified clinical reviewers inspect substantive claims against supplied
evidence, not model scores or role designation. Record source locator and
reasoning, including contradictory or absent evidence and clinically
material limitations. Labels: supported (all material claims supported),
contradicted (at least one material claim directly contradicted),
insufficient_evidence (no demonstrated contradiction but material support
missing), ambiguous (meaning/evidence too ambiguous), not_applicable.
Potential impact: none, minor, potentially_major, uncertain; hypothetical
clinical relevance, not measured patient harm. Prefer explicit material
claim rationale; do not equate unsupported with contradicted or hallucinated
benchmark role with medical error. Lock independent submissions; third
qualified reviewer adjudicates label or impact disagreements and unresolved
cases. Record final uncertain judgments rather than forcing binary labels.

Primary descriptive endpoint: strict paired role concordance (Ground Truth
supported AND Hallucinated Answer contradicted). Secondary endpoints:
confirmed role discordance (Ground Truth contradicted OR Hallucinated
Answer supported), uncertain parent (either answer insufficient/ambiguous/
not applicable), potentially major impact in either answer, and both
answer labels agreed by initial reviewers. Endpoints can overlap; no forced
sum-to-one. Report per split, parent as sampling unit, inverse stratum
sampling fractions through population weighting, conservative 95% intervals
by summing exact hypergeometric bounds with Bonferroni across strata within
each endpoint. Not simultaneous across endpoints, not superpopulation
generalization, not a precise estimate of rare harm. Cohen kappa and raw
agreement are descriptive; design-aware agreement intervals are reported
separately. No fabricated adjudication, physician gold labels or clinical
factuality certification of all 18408 answers from a sample. Independent
scientist must approve the intended benchmark-role task and limitations;
claim-level clinical factuality requires separate validated reference labels.

## D. Freeze preparation and analysis lock
All seven gates must have actual signed evidence: rights, provenance
review, semantic disposition, clinical-label review, independent scientific
approval, locked analysis protocol, private manifest review. Pre-access
receipts also bind private environment, review plan and one new GPU run.
Each receipt is bound to checkpoint, mapping, source implementation digest,
environment, nonce/session, analysis protocol and benchmark-only scope.
Reviewer submissions are signed over exact packet and judgment digests.
Custodian recomputes analysis from authenticated submissions. Signatures
and hashes are structural verification, not clinical/scientific approval.
The analysis protocol must preregister task definition, baselines, primary
metrics, seed/model/threshold selection on train/dev only, calibration,
risk-coverage, paired source-group-aware uncertainty and multiplicity,
locked test evaluation and external generalization before access to
test judgments/model results. It is not yet locked by this implementation.

No automatic final freeze. A successful signed check means the custodian
may perform a separately documented manual immutable manifest freeze in
the authorized private store after independent approval. Retain provenance,
review receipts and private manifests under approved retention rules.
Unresolved missed pairs block this mapping; do not move test records to
training. Clinical sample approval must specify benchmark-only scope and
acknowledge uncertain and discordant cases; do not claim all labels validated.
A different clinical gold task needs separate scientific approval.
No e1-approved-protocol.json, registry readiness change or E1 model job is
created; later E1 execution requires fresh authorization.

## E. Engineering implementation and safe operation
- e1_master01d_review.py: checkpoint-bound preparation, review coverage,
  finite-population interval inversion, weighted estimates and uncertainty.
- e1_master01d_gate.py: signature verification, independent role separation,
  exact subject/evidence binding, manual freeze eligibility only.
- e1_master01d_session.py: pinned source verification, signed pre-access
  gate, private session preparation, authenticated analysis and manual gate.
- test_e1_master01d.py and e1-master01d-synthetic.yml: invented inputs only,
  actual cryptography dependency, negative tests and exact interval coverage.
  GitHub Actions does not access medical data, reviewer identities or secrets.
- No real-data workflow dispatch is provided for 01D until the authorized
  private reviewer environment is arranged. The manual dispatch of the
  synthetic workflow runs tests only. No automated checkpoint/resume,
  private artifact persistence, authenticated reviewer UI or actual final
  freeze has been deployed. No clinical results from this phase yet.
