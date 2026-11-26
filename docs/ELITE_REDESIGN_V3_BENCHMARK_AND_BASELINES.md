# ELITE RESEARCH REDESIGN V3 — prior-art stress test and benchmark feasibility
2026-11-26. Engineering proposal; not a completed systematic review or approval.
Base: PR #22, 7522b235f4fa2fd6fa9ae16cd2fbcc410b031543.
MASTER 01C mapping, holdout, and existing readiness blocks unchanged.
No new human annotation, GPU execution, private dataset access or label review.

## Independent literature audit
A Consensus Pro literature thread was retrieved in completed state:
https://consensus.app/search/research-brief-for-computational-thesis-methodolog/UzgXRqd3TpOZ2ElwLdS6Eg/
Its automated search reports 8573 retrieved, 64 eligible, 20 included; not
independently verified record counts or a PRISMA flow. Do not redistribute
the proprietary synthesis in the public repository. The full report remains
in Consensus. The following is our independent assessment of primary
records, not a copy of that synthesis. Search date 2026-11-26; public web
queries included arxiv IDs below, official BioASQ data/license, MedMCQA
and MedQA licenses, PubMedQA and MEDEC repositories, ACL Confidence/Critique.
No claim of exhaustive search or full reproduction.

| Competitor | Primary record | Verified overlap / caveat |
|---|---|---|
| Yang et al., Confidence v.s. Critique, ACL 2025 | https://aclanthology.org/2025.acl-long.203/ | Separates repair and preservation; transition metrics are prior art |
| Zhang, risk-aware intrinsic revisions, Sep 2026 preprint | https://arxiv.org/abs/2609.35832 | Selective invocation based on initial-response signals; compare runtime gating separately from post-generation selection; primary HTML unavailable |
| Liu & Meng, feedback control, Apr 2026 preprint | https://arxiv.org/abs/2604.22273 ; version 1 https://arxiv.org/html/2604.22273v1 | Two-state transition dynamics and verify-first intervention; aggregate repair/preservation baseline relevant |
| Li & Yu, HC-ROCP, Nov 2026 preprint | https://arxiv.org/abs/2611.12662 | Harm-constrained actions and disjoint multiplicity-adjusted calibration; abstract checked, primary HTML unavailable; no assertion that its clinical harm target equals our correctness transition |
| Kamoi et al., TACL 2024 | https://aclanthology.org/2024.tacl-1.78/ | Self-correction evaluation can be unfair; external feedback is important |
| Learn Then Test, prior V2 | https://arxiv.org/abs/2110.01052 | Established held-out risk calibration; IUT/Holm not novel |

Critical result: joint softmax is NOT a uniquely expressive representation.
For binary correctness, p0, pr and h=P(Y0=1,Yr=0) completely identify the
four-state distribution. h must lie in [max(0,p0-pr),min(p0,1-pr)].
Conditional factorization P(Y0)P(Yr|Y0,x) is equally expressive in principle.
Therefore no theoretical superiority claim over a coherent direct-harm
estimator or conditional model. The testable hypothesis is better
out-of-sample estimation (proper joint log loss and harm Brier) and
matched-budget intervention outcomes from shared evidence contrasts and
joint training. If these comparators equal or beat PIRC, the proposed
novelty is unsupported. The same-marginal counterexample is motivation,
not a theorem of empirical benefit. Compare no-contrast and no-perturbation,
same-marginal independence (diagnostic only), independently trained marginals,
direct harm with Frechet clipping, conditional factorization and constant
training transition rates. No model-dependent labels or expert certification.

## Data rights audit: public metadata only
| Dataset | Official evidence | Decision |
|---|---|---|
| BioASQ 14b | https://participants-area.bioasq.org/datasets/ lists CC BY 2.5 and 5729 questions for training 14b; https://participants-area.bioasq.org/general_information/Task14b/ describes 5389 development questions | Different counts may reflect different release snapshots. Must identify exact release and count yes/no only; confirm rights/underlying article and snippet terms, registration, gold-test availability, cross-source overlaps; no assumption all are independent. Preferred candidate. |
| PubMedQA | https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE MIT repo; https://aclanthology.org/D19-1259/ expert labels, yes/no/maybe | Preserve official 500 holdout; other labeled rows limited, rights to source abstracts separate. No conclusion/long answer in features. |
| MedMCQA | https://github.com/medmcqa/medmcqa MIT repo, exam content | Candidate exact MCQ task; underlying exam rights and evidence provenance unresolved, official test label access check. No rationale/explanation fields used as evidence. |
| MedQA | https://github.com/jind11/MedQA/blob/master/LICENSE MIT repo | Candidate separate MCQ task, source exam/textbook terms pending. Not an interchangeable biomedical evidence QA population. |
| MEDEC-MS | https://github.com/abachaa/MEDEC lists 2189 MS train, 574 MS validation, 597 MS test | Existing error flags/spans/corrections, not closed-QA intervention correctness. Rights unresolved; no UW transfer. |
| MedVAL | existing restricted terms | Prohibited on Kaggle/GitHub/chat; no processing. |

Repository software licenses are not proof of rights to every underlying
item. No dataset bytes downloaded, no access approval, no actual eligible
count, no independent source group verified. No benchmark substitution yet.

## V3 operational contract
A single reference-scored QA question, fixed original and fixed revision.
Strict parser for yes/no(/maybe) or A-D/A-E; invalid output cannot be
released. Gold comparison for scoring only, never features/prompts.
Input IDs are private, not exported; aligned_extract enforces identical
ordered evidence IDs for both candidates and no extra label field. For a
closed-answer claim, upstream inference must convert the question and
candidate into a deterministic answer-conditioned hypothesis; exact
template, evidence segmentation/retrieval, ordering, truncation policy,
confidence definition, NLI checkpoint, label mapping, tokenizer and hashes
remain unimplemented. Never treat a yes/no question itself as an assertion.
For MCQ use question and selected option text, never gold explanations.
Unit-level NLI posteriors must be finite, normalized, aligned. Support and
contradiction are independent maxima across units, not complementary;
coverage is fraction of units with entailment >= .5, not evidence recall.
Sensitivity is max support minus max support after deleting the first
fixed answer-blind unit, zero if negative. This is score-aggregation
ablation, not a counterfactual generation test or a fresh NLI pass.
Empty evidence sets give zero evidence features and explicit missing flag.
No claim that NLI scores measure truth. Feature code consumes precomputed
numbers, does not yet perform transformer inference or data processing.
Upstream must preserve IDs/content binding and prohibit arbitrary order.
Confidence and feature calibration not established.

Trainable comparators share V2 features, same candidates, train targets
and tuning-only model selection; parameter counts differ and must be
reported. Direct harm model projects predicted h to feasible marginal
bounds; conditional heads trained on corresponding original-correct and
original-wrong changed subsets, with no actual correctness input at
inference. Unchanged outputs have zero off-diagonal probability.
Every method applies the same invalid-output guard before calibration.
Always accept/revise and marginal selector baselines remain. Closest
published full implementations (pre-generation selective invocation,
verify-first prompting, HC-ROCP and others) are not yet reproduced.
Use actual generator/token/NLI and latency costs; no claim of matched
runtime solely from matched input features.

## Calibration feasibility
Use group identities derived from connected original source IDs, article
links, normalized text and near-duplicate controls across datasets.
Metadata inventory function accepts one representative per group and
rejects repeated group IDs; cannot detect unrecognized shared sources.
Reference labels and answer roles remain separate. Count eligible groups
after rights and private overlap audit, before using model labels. Fixed
K=12 V2 policy family is provisional; assess K=1,3,12 for hypothetical
power with the same caps, not post-calibration family selection. V2 caps
R=.10,I=.05,C=.50 are illustrative benchmark tolerances, not clinical
thresholds. Compare favorable (.03,.01) and tight (.07,.03) rates; true
coverage .75, original-correct .70. No actual efficacy evidence. One
preselected independent source representative is needed for the stated
binomial assumptions. No pooling mismatched question types or tasks
without a separately defined target distribution and valid sampling.
Insufficient data -> prospective smaller family or descriptive-only
analysis, no post hoc relaxation, no use of reserved test.

## Next gate and limitations
Need private, separately authorized release inventory and source-disjoint
cohort construction, actual model assets and feature inference, closest
published method reproduction, locked train/tune/calibration/test policy,
benchmark-only readiness path, preregistration and then real runs.
Synthetic CI is engineering evidence only. Keep all 01C quarantine and
human-dependent 01D historical evidence; no fake reviews or gate changes.
No Top 1% acceptance or demonstrated novelty claim.
