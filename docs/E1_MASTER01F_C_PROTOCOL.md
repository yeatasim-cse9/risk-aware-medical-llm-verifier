# MASTER 01F-C v1 — prospective qualification, not approval

Public source code and hypothetical planning only. No empirical research
results, private IDs, source memberships, actual timings, or private hashes
are added here. No dataset access, new GPU, submission path or final test.
The actual V4 artifact and original runtime log must remain private.

## Exposure audit and revised interpretation

Review the historical 01/01B/01C screening, 01D V3 shortcut audit, V3 and
recovery failure stages, V4 input acquisition and pilot, and 01F/01F-B
development. Distinguish covariate exposure, label-informed cohort-level
development, model fitting, evaluation and aggregate adaptation.
The 01D V3 shortcut audit consumed paired answer roles for all retained
train/dev parents; V4 also materialized train/dev answer roles before
sampling. Therefore excluding only the V4 fitted source groups is
insufficient for claiming fully untouched confirmatory calibration.
All retained train/dev groups are conservatively development-exposed.
A label-blind source split of the existing train pool remains suitable
only for explicitly exploratory development calibration. It cannot
retroactively undo aggregate label exposure. Existing dev is observed.

Historical source/parquet readers could materialize whole answer columns,
and official ground-truth file bytes were historically used for integrity
and ID checks. Do not claim test-answer bytes were never loaded historically.
This phase does not read any test answers. Future controller must enforce
actual column/row projection and isolate test labels behind a custodian;
preserve historical mapping without copying unsafe access patterns.

A fresh candidate requires independently authenticated joint source
components across all protected originals (including quarantine and
holdout covariates), an explicit inventory of all other exposures, known
provenance, same target task and valid role/label semantics. Unknown
provenance and all overlapping groups are excluded. A metadata-only
structural candidate check cannot prove iid sampling, completeness,
provenance, rights, or clinical validity. It never authorizes execution.
Private in-memory group sets are not output artifacts; the report exposes
only a count band and target feasibility, with strict recomputation and
no approval fields true. The target count is a prespecified design input,
not a private count. No public export approved.

Prior PubMedQA inventory concerns yes/no/maybe and different labels;
it cannot simply be pooled as independent paired MedHallu role examples.
External cohort acquisition, comparable label construction, source
screening and rights require separate design and authorization. No
human reviewer is presumed available or fabricated.

## Calibration power

Use 01F-B's six fixed thresholds, two one-sided binomial bounds per
threshold, family alpha .05, candidate released-group role error cap
.10 and group coverage floor .50. For a single fixed threshold under
iid group representatives, M~Binomial(n,c) and E|M~Binomial(M,r).
Integrate the probability of passing the risk bound over denominators
that pass the coverage bound. No product-of-marginal-powers shortcut.
A zero-error released denominator of 53 is necessary for the risk
bound alone, not a total sample size. Hypothetical alternatives:
r=.03,.05,.07, c=.75; additionally c=.60 at r=.03. Plan at 200,300,
500,750,1000,1500,2000 independent groups. A candidate 500-group target
has roughly 80% joint power at r=.05,c=.75, but near-boundary risk
requires much more. Do not use this as an empirical rate estimate.
If insufficient eligible groups, no confirmatory calibration.

For later AP superiority, use a prespecified meaningful effect and
paired source-level pseudovalue SD estimated only on permitted fitting
data; the aggregate V4 CI does not supply that SD. Normal approximation
sensitivity for effect .02, two-sided alpha .05, 80% power at hypothetical
SD .15/.25/.35 gives 442/1227/2404 groups. No final confirmatory sample
size locked; adjust for multiple primary hypotheses and validate by
simulation. Do not use test outcomes to choose sample size.

## Compute plan

Privately verify the V4 original ZIP and receipt, then supply the actual
stage times and train/dev parent counts to compute_plan. Current collection
uses 2 answers x 3 conditions x 16 slots per parent, 512 padded positions
per forward example, one cuda:0. Use the larger measured inference seconds
per parent, add measured pre/post overhead, add a provisional 30% margin.
No guarantee for larger optimization/bootstrap costs, context distribution,
queue, network, runtime drift, quota rounding or GPU availability.
Do not infer speedup from the unused second T4. Compute plan is separate
from approval and quota. Report only within private scope.

Next: protocol remediation and independent cohort feasibility; if desired,
separately design an explicitly exploratory train/selection-only hybrid
pilot with no confirmatory calibration or test use. Before any actual
private experiment implement authenticated exposure and input provenance,
strict answer access policy, bounded compute and output validation,
versioned controller, fresh rights/quota authorization and read-only
preflight. No change to historical code or 01F-B scientific algorithms.
