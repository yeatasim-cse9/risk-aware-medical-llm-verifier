# MASTER 01F-R — prospective alternative protocol v1
Status: engineering proposal for supervisor approval; no data acquisition,
test access, execution authorization, clinical certification or empirical claims.

## Decision and history
Retain the thesis title provisionally. Preserve all historical MedHallu V4
results privately, including negative baseline comparisons, and all prior
exposure limitations. Do not resume the mixed-corpus historical path without
its original custody gate. Retire the blocked expansion as the critical path,
not as failed scientific evidence. Do not reassign historical test to train.
01F-R is a separate prospective study, not continuation of the same sample.

## Detection track: prioritized candidate, not approved
MedHal (LAMA-WeST/medhal-lf_qafil_bal), documented native train/validation/test,
is a feasibility candidate. Consult Mehenni and Zouaq, MedHal: A Synthetic
Dataset for Medical Hallucination Detection, AACL 2026
(https://aclanthology.org/2026.aacl-long.208/) and its dataset card
(https://huggingface.co/datasets/LAMA-WeST/medhal-lf_qafil_bal).
The card documents context, statement, label, source and inner_id; label
polarity, underlying source licenses, context semantics, immutable revision,
physical shard inventory, and group counts must be established before use.
No test rows or viewer samples are needed in metadata qualification.
Do not assume that separate dataset names imply independent source articles.
Exclude restricted clinical-note sources, unknown provenance, and QA rows
whose evidence is absent or merely the target explanation; no explanation
feature. Qualify a fixed evidence-bearing task/source subset before any fit.
If QA evidence cannot qualify, statement verification is a narrower endpoint;
require explicit supervisor approval, not silent substitution for QA.
MEDHALU (https://netsys.surrey.ac.uk/datasets/medhalu/) is secondary external
feasibility only; no pooling or label conversion without a separate protocol.

Use the upstream native train split for all fitting and selection. Group by
verified underlying source identity and connected exact/near-duplicate
proxies; inner_id plus source alone is insufficient if articles repeat across
tasks. Choose a deterministic label-blind group allocation within train.
A provisional 256/128 group pilot is for feasibility only, not power.
Native validation is reserved for calibration, not tuning or method choice.
Native test is sealed until model, thresholds, endpoints, sample size and
analysis code are frozen. Only native train bytes may be downloaded during
development. File inventory must distinguish physical splits and reject
aliases, extra/mixed files and unpinned paths. The metadata gate in this PR
does not download files or certify actual server behavior. Later acquisition
must hash actual permitted train bytes and enforce an isolated private mount.
Before calibration/final evaluation, source disjointness across partitions
requires separate non-answer provenance inventory or a publisher-supported
group separation guarantee sufficient for the intended claim; if unavailable
do not claim independently verified disjointness or confirmatory calibration.
Audit all other exposures; model pretraining contamination remains unknown.
No claim of independence from historical MedHallu without lineage evidence.
No need for a bespoke human signer if publisher-native separation, provenance,
rights and reproducible access controls meet these requirements. This is a
prospective change requiring approval, not a bypass of historical custody.

## Methods and endpoints
Adapt the frozen DeBERTa candidate's evidence/statement features after runtime
qualification. Compare length, style, statement-only, evidence overlap, full
lexical, raw NLI contradiction/nonentailment, neural and hybrid on identical
membership. Question-only/QA-overlap apply only if explicit valid questions
exist; otherwise record nonapplicability and do not compare numerical results
to historical paired-role metrics. Fit/calibrate no head on test. Use
full/removed/permuted evidence, matched compute and frozen/retrained removal,
with source-group permutations avoiding same group. Predefine token-budget
eligibility from permitted development data and report exclusions/coverage.
Primary proposed detection endpoint: AP of benchmark hallucinated statement
label, paired hybrid minus strongest lexical baseline selected on development.
Report Brier, AUROC, precision/recall, F1 and coverage descriptively; compare
methods with common group bootstrap draws. Paired ranking only for genuinely
paired labels. No forced superiority. Freeze exact model, metric implementation,
contrast family and sample-size justification before final test.
Plan meaningful AP difference .02; use permitted fitting-group pseudovalue
variance and group-resampling simulation, not historical pilot CI as variance.
Use two-sided 95% group-bootstrap intervals, at least 2000 replicates proposed;
Holm-adjust any additional confirmatory contrasts or allocate alpha prospectively.
No test-based changes or optional stopping.

## Risk track
New endpoint: among accepted independently sampled source representatives,
probability of accepting a benchmark-labeled hallucinated statement. One
representative per group selected without labels/scores; do not mix with
historical paired-role risk. Fixed thresholds .05,.10,.20,.30,.40,.50;
cap .10, coverage floor .50, family alpha .05, exact one-sided binomial
bounds with Bonferroni across 12 bounds. Only claim conditional statistical
control under justified iid representative sampling and frozen scores; if
no policy meets bounds report NO_POLICY_MEETS_BOUNDS. Existing 01F-C power
math applies under the new Bernoulli representative assumptions, not as
empirical evidence. Initial planning 500 groups for hypothetical 5% risk and
75% coverage; near-boundary scenarios may require 1500 or more. Recalculate
and approve sample size before calibration. No clinical safety claim.

## Correction track: objective prospective structured task
ClinicalTrials.gov public study records (official API docs:
https://clinicaltrials.gov/data-api/api ; terms:
https://clinicaltrials.gov/about-site/terms-conditions) are a feasibility
candidate for structured evidence-grounded trial fact questions. Verify
terms, rights, snapshot timestamp, source fields and identifiers first.
No patient data or treatment advice. Use factual fields such as study type,
enrollment count and recruitment status, with explicit as-of date and
enrollment type; no inference of efficacy. Prespecify eligible fields and
sampling frame; freeze development study IDs and use a separately acquired
later or source-disjoint evaluation set after prompts, verifier and scorer
lock. No claim of pretraining independence. Preserve source values privately
under applicable terms. Gold values are registry-reported facts, not clinical
truth. Generate actual LLM responses with frozen generator revision and prompt;
use the same initial response across all intervention arms, identical evidence,
and comparable compute or report compute differences. Compare unmodified,
evidence-only generation, unconditional revision, verifier-triggered revision,
abstention-only and deterministic source lookup as a strong baseline.
Require exact typed answer format (one requested value or explicit abstention)
and a source citation; use independently specified exact matching against
registry field, not the verifier or same LLM as judge. Wrong values or citations
are unsupported; malformed/extra prose is invalid and separately reported.
Primary proposed correction endpoints: wrong-to-correct transition among
initial errors and paired net valid-correct response change across all items.
Report correct-to-wrong harm, abstention, malformed response, residual error
among released and unconditional counts. Abstention is not correction.
Use study-group bootstrap, power from permitted development data, freeze
sample and multiplicity plan before evaluation. Structured task results do
not establish open-ended clinical hallucination correction. If this narrow
task cannot support the approved title/objectives, revise scope/title
transparently; never invent gold or use circular self-judging.

## Gates and remaining roadmap
R1 supervisor approval of revised protocol and rights review; R2 metadata-only
native file/revision qualification; R3 authorized private CPU train-only
acquisition and group/label/feature audit; R4 immutable candidate and private
preflight; R5 separately authorized exploratory GPU pilot; R6 calibration
qualification and locked policy; R7 separately authorized locked test;
R8 prospective correction experiment; R9 robustness/statistics; R10 thesis
and paper. Each empirical stage independently verifies original outputs and
archives authorized evidence in private GitHub; never public Actions results
or Google Drive. Do not submit/retry GPU without fresh one-run approval.
A synthetic test or metadata plan cannot certify rights, independent source
provenance, actual physical separation, model performance or readiness.
If a gate cannot be met, stop that claim and disclose it. Supervisor approval
is outstanding; no automatic execution or data download is enabled.
