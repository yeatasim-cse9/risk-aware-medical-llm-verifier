# MASTER 01F-R4 — results-first source decision (proposed)

Status: decision record, not approval or execution. No new dataset acquired,
no protected answers read, no experiment submitted, no scientific result
claimed. Historical MedHallu evidence and R2/R3 infrastructure remain intact.

## Decision

Do not make the entire thesis dependent on resolving mixed-source MedHal
rights. Prioritize the already proposed R1 structured factual correction
track using the ClinicalTrials.gov official API, subject to explicit review
and authorization. MedHal remains the intended broader detection candidate,
on hold unless all five native train shards are authorized or a physically
separate permitted evidence-bearing subset is provided. No full-shard
download followed by filtering. No historical test reassignment.

This is a sequencing and scope decision, not an assertion that registry
facts replace medical QA or that the original paper objectives are met.
ClinicalTrials.gov registry-field evidence is narrower than clinical
reasoning, open-ended medical hallucination or patient-level safety.
A strong result on this task alone cannot substantiate those claims.
Retain the thesis title provisionally; supervisor must approve the scope.

## Public source review (documentation only)

* ClinicalTrials.gov Terms and Conditions:
  https://clinicaltrials.gov/terms-conditions
  Data are publicly available without charge, but third-party copyright
  may apply. Review the complete current terms, attribution, processed date,
  and modification disclosure obligations before use. Do not treat all
  registry content as public domain.
* Official download/API guidance:
  https://clinicaltrials.gov/data-api/how-download-study-records
  https://clinicaltrials.gov/data-api/api
  https://clinicaltrials.gov/data-about-studies/csv-download
  https://clinicaltrials.gov/policy/protocol-definitions
  Study identifiers, recruitment status, study type and enrollment fields
  are documented. Enrollment requires distinguishing actual and estimated.
* NCI text reuse policy:
  https://www.cancer.gov/policies/copyright-reuse
  https://www.cancer.gov/publications/pdq/information-summaries
  NCI text is generally reusable unless otherwise indicated; third-party
  text, images and marks excluded; modified excerpts are not NCI PDQ
  summaries. A future clinical-text extension requires separately validated
  claim labels and source-level splits; not an immediate acquisition.
* CliniFact, Scientific Data 12:86 (2025):
  https://doi.org/10.1038/s41597-025-04417-x
  Published native split CSVs and evidence-linked clinical claims are
  promising, but underlying abstract rights and source overlap unresolved.
* PMC OA policy: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
  Article-level CC0/CC BY selection can reduce rights ambiguity; approved
  retrieval mechanisms and individual licenses still required. No acquisition.
* MedHal source documentation:
  https://aclanthology.org/2026.aacl-long.208/
  Prior verified metadata and rights review in R2/R3. No source permission
  assumed from the top-level license. No publisher contact made.

## First actual study: narrow registry-fact QA

After real supervisor approval, rights review, and fresh explicit CPU
authorization, collect a development-only sample in private Kaggle.
Restrict API field projection to NCT identifiers, first/last posted dates,
study type, overall recruitment status, enrollment count and enrollment
type, and minimal non-answer linkage identifiers needed for duplicate
grouping. No contacts, locations, narrative clinical notes, results text,
documents, patient data or other unreviewed fields. Verify exact endpoint,
returned schema, API version and source processed timestamp before bulk
requests. No blanket legal approval is implied.

Prespecify a fixed sampling frame, inclusion rules and deterministic
source-group selection before accessing answer-bearing development records.
Use identifiers and linkage metadata only for allocation, if separately
permitted. Resolve aliases and linked/duplicate studies; unknown links
remain a limitation. If the API cannot physically isolate development
requests, do not proceed. Request projected answer-bearing fields only
for selected development IDs. No calibration or test fields during
development. Avoid broad downloads and shared caches. Freeze manifest,
as-of timestamp, source record hashes and eligibility privately. Later
calibration and test are independently scoped; only after frozen model,
prompts, scorer, sample size and separate authorization. No claim of
pretraining independence or IID validity merely from distinct NCT IDs.
Never access historical protected answers.

Provisional development feasibility target: 128 distinct study groups,
up to three eligible questions per group (study type, status, enrollment
count/type). Not a powered final sample, no observed count. One locked
generator and matched initial responses across intervention arms. No
fabricated predictions or forced errors counted as real hallucinations.
If natural initial error count is too small, report and reassess design
prospectively; do not induce errors post hoc to improve results.
Counterfactual synthetic perturbations, if separately used, must be a
distinct stress-test, never pooled with naturally generated responses.

Independent exact typed registry-field scorer plus required correct NCT
citation, frozen before scoring. A correct registry value is not a
clinical truth guarantee. Compare original, evidence-only, unconditional
revision, verifier-triggered revision, abstention-only and deterministic
lookup. Evaluate wrong-to-correct and net valid-correct response change,
correct-to-wrong, abstention, invalid format, residual error and coverage.
No correction credit for abstaining. Report compute differences, paired
study-group uncertainty, development-only tuning, and multiplicity.
Fit and select only on eligible development groups; no confirmatory
threshold or risk guarantee on reused development groups. Later sample
size, confidence bounds and final endpoints require prospective lock.

## Operational stop/go

STOP today: actual source-specific permission/terms review and supervisor
scope approval not recorded. No new Kaggle submission or data download.
No additional controller/PR series is required solely to add tests.
Next step is one explicit rights/scope decision, then a minimal private
CPU data acquisition and baseline feasibility run (after runtime and
model rights verification). Reuse existing infrastructure where suitable;
do not repurpose MedHal-specific readers for registry rows. New code only
to support the actual approved experiment, with synthetic regression.
GPU authorization remains a separate fresh, exact-one decision.
Archive only authorized artifacts privately; public repository contains
code/protocol and public metadata only. No Google Drive or local PC
research execution. Preserve negative results and historical limitations.
