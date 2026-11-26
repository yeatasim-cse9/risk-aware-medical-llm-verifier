# MASTER 01F-B v1 — candidate protocol, not scientific approval

This public repository contains only code, invented nonmedical fixtures,
and design. Do not commit actual pilot aggregates, per-row results,
private group identifiers, model heads, calibration predictions, or
private data. The prior V4 experiment is exploratory, and the development
partition has already been observed. No numerical superiority, clinical
grounding, correction efficacy or final scientific claim is approved.

## Controls and common comparisons

Positive class is the benchmark hallucinated-answer role, not clinician
adjudicated correctness. Retain the original lexical baseline (including
question-answer overlap) unchanged. New evidence-only features are
answer-context token-set recall and Jaccard; question-answer overlap
is a separate shortcut control. Length, style, answer-only, question-only,
question+answer and full lexical baselines are retained. Token overlap
cannot resolve semantic scope, negation, role swaps or numeric equivalence;
invented fixtures demonstrate this limitation, not measured clinical results.

Use identical fitting and observed development rows for all models.
Full, removed and same-partition cross-source permuted conditions use
the same answer and question. Permutation is label-blind and deterministic;
it does not prove donor irrelevance. Removed frozen and removed retrained
are distinct interventions. In exact normalized word-count tie subsets,
report descriptive metrics only if sufficiently large; no small-cell export.
No length-matched selection or resampling based on outcome.

The neural candidate uses six original V4 NLI summary features.
Hybrid adds two evidence-only features; hybrid-legacy adds the original
four lexical features to expose shortcut sensitivity. Each has the same
train-only standardization and logistic head optimizer. No tuning on
calibration. The fixed development suite has 16 paired contrasts, with
AP, paired ranking and Brier differences using common source-group
bootstrap draws. 95% percentile intervals are descriptive and unadjusted;
they do not establish superiority or equivalence. Do not select a winner
automatically from this suite. A later independent confirmation must
prespecify endpoints, multiplicity strategy, sample size, and candidate.

## Partition and exposure safeguards

Plan metadata accepts only parent, source_group and original train/dev
partition. Original dev becomes selection, never calibration. Reconstruct
the exact prior pilot's 128 training source groups via original deterministic
sample order, exclude them from calibration, and require an explicit
additional previously exposed training-group inventory. A group is never
split between fit, selection, calibration. Reject parent/group overlap,
unknown exposed groups and insufficient eligible groups. No test answer
or holdout metadata is accepted by the planning API. Fit/score require
exact planned membership; calibration uses one label-blind representative
per group. Metadata/source-component membership and exposure completeness
must be authenticated upstream. The historical lineage audit used
covariates across splits; the code does not establish iid sampling,
unobserved exposure, independent clinical labels or a frozen final split.
The Plan type is an in-memory contract, not a cryptographic provenance proof.

## Calibration candidate

Freeze one full-condition model, six thresholds (0.05, 0.10, 0.20, 0.30,
0.40, 0.50), alpha and risk/coverage targets before calibration outcomes
are inspected. Lower score means lower estimated hallucinated-role risk.
At each threshold release answers with score <= threshold. Group endpoint:
among representative groups with any released answer, probability of at
least one released hallucinated-role answer. Coverage endpoint: fraction
of representative groups with any release. Individual answer coverage
is descriptive. Two one-sided Clopper-Pearson bounds per threshold;
Bonferroni across the 12 bounds. Choose maximum observed group coverage
among passing thresholds, otherwise no policy. Default planning targets:
risk <= 0.10, coverage >= 0.50, alpha=0.05. These are candidate research
targets, not clinical deployment requirements or guarantees. With zero
errors, 53 released independent groups are needed for the risk upper
bound alone; coverage and nonzero errors impose additional requirements.
No revision generator, correction labels or accept/revise/abstain
clinical certification. Calibration family must remain fixed and not
be rerun to adapt to calibration outcomes. The pure API cannot enforce
cross-run researcher behavior.

## Integration boundary

No remote workflows, secrets, data downloading or GPU submission in this
phase. The feature collection adapter only calls an explicitly authorized
caller-supplied engine, with original V4 condition contract. Before a real
run authenticate the immutable model, source-component mapping, exposure
inventory, exact row/condition provenance and resource budget. The adapter
does not independently authenticate supplied neural arrays. Re-run both
partition input-budget audits before private NLI inference in any future
controller. No actual fit/calibration counts or empirical metrics produced.
A future private controller, privacy-safe output schema, scientific
protocol review, fresh rights/quota/one-run authorization and separate
read-only preflight are required before real processing.
