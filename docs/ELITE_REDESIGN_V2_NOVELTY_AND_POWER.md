# MASTER RESEARCH REDESIGN V2 — novelty and mathematical protocol
Date: 2026-11-26. Proposed computational study, not a completed systematic
review, preregistration, scientific result, or approval to execute.

## Checkpoint and constraints
Base main 19b71d6a0fe9336524632e511dc346369fe0c5ed (PR #21).
Verified MASTER 01C canonical aggregate 83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d;
provisional retained mapping 025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb.
7427/890/887 parents, 18408 paired roles; separate official 500 holdout.
No change to existing approval gates, registry, data rights or holdout. No
new human annotation, physician recruitment or human adjudication; existing
reference labels only. Academic approvals and relevant data rights remain.
No real E1–E7 results; no real-data submission in this phase. The old 01D
human-review path remains historical, not falsely completed.

## Search log (bounded targeted search, not systematic completeness)
Public web index, two retrieval modes, 2026-11-26. Searched 20 strings below,
with English-language method/benchmark terms. Prioritize primary publisher,
ACL Anthology, PMLR, arXiv abstracts/HTML and PubMed records. Include works
that directly address paired revision outcomes, calibrated selective
prediction/routing, evidence verification, or reference-scored medical tasks;
retain methodological and search-reporting references separately. Deduplicate
by DOI/arXiv/venue identity. Exclude secondary mirrors where a primary record
exists, nonmethodological listings, unrelated tasks with no transferable
method. Search results are ranked/truncated, no exhaustive hit counts or
formal dual-screening flow can be established. No fabricated PRISMA flow.
Some primary full-text opens failed; compare abstract/available HTML only
where that is all accessible. No competing implementation reproduced yet.

Exact search strings (quotes included where used):
1. selective generation revise abstain risk control revision introduced errors conformal LLM factuality calibration
2. LLM self correction preserving correct answers wrong to correct correct to wrong intervention calibration medical question answering
3. learning to defer revise accept abstain language model selective prediction risk control paired correction
4. learn then test risk control multiple risks Angelopoulos exact binomial
5. medical hallucination MedHallu MedVAL MEDEC benchmark medical error correction reference labels
6. selective self correction language models correction risk original revised answer accept abstain calibration "revision"
7. conformal factuality selective generation CAP abstention self correction evidence language models
8. medical question answering evidence perturbation counterfactual evidence sensitivity hallucination verifier revision
9. "LEC" "Linear Expectation Constraints" selection conditioned risk control Wang 2026 routing constraints
10. "When Should LLMs Trust Their Own Revisions" selective revision preservation gates 2609.35832
11. language model selective revision accept abstain paired correctness conformal repair risk control 2026
12. PubMedQA original 500 test labeled 1000 split Jin BioASQ 14b yes no benchmark
13. PubMedQA A Dataset for Biomedical Research Question Answering EMNLP 2019 Jin 500 test 500 training
14. Holm 1979 simple sequentially rejective multiple test procedure intersection union tests Berger Hsu 1996
15. LLM joint correctness transition prediction revision paired risk conditional calibration evidence verification selective acceptance original revised answer joint probabilities
16. PRISMA S search reporting systematic reviews literature search 2021 TRIPOD LLM
17. LLM self correction predicting correction success regression joint original revised correctness verifier selection "revision" "risk"
18. LLM revision acceptance policy evidence grounded medical hallucination detection correction risk paired transitions 2025 2026
19. site:aclanthology.org selective self correction LLM confidence correct answer preservation verifier revise
20. site:pubmed.ncbi.nlm.nih.gov medical large language model evidence hallucination correction selective prediction calibration

## Nearest-work matrix (selected primary records, not exhaustive)
| Work and source | Relevant verified contribution | Consequence |
|---|---|---|
| [Confidence v.s. Critique, ACL 2025](https://aclanthology.org/2025.acl-long.203/), [preprint text](https://arxiv.org/abs/2412.19513) | Decomposes correctness transitions and conditional preservation/repair; training method | Four states and transition rates not novel |
| [Zhang, September 2026 preprint](https://arxiv.org/abs/2609.35832) | Tracks revision recovery/harm, compares selective invocation and keep/revise, learned gating | Must compare selective revision, distinguish pre-generation gating from post-generation selection |
| [LEC, ICML 2026](https://proceedings.mlr.press/v306/wang26a.html) | Linear expectation constraints for selection-conditioned risk and two-model routing | Risk linearization/routing not novel; full text access failed in this audit |
| [SC-LEC, November 2026 preprint](https://arxiv.org/abs/2611.21384) | Authors identify counterexample to prior empirical threshold selection, propose stability certificates | Do not assume same-sample selected thresholds automatically valid; compare corrected methods |
| [Learn then Test](https://arxiv.org/abs/2110.01052), [published record](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk) | Held-out risk calibration through multiple testing | Statistical framework is established |
| [CAP, ACML 2025](https://proceedings.mlr.press/v304/tayebati26a.html) | Context-adaptive abstention | Compare calibrated abstention and adaptive uncertainty |
| [ProCo, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.714/) | Key-condition verification for self-correction | Verification-guided revision not novel |
| [Self-correction critical survey, TACL 2024](https://aclanthology.org/2024.tacl-1.78/) | Reviews conditions for self-correction and negative findings | Include external-evidence and matched candidate baselines |
| [Retrieve-Then-Verify, JMIR AI 2026](https://ai.jmir.org/2026/1/e93761) | Evidence support/contradiction assessment for medical information | Retrieval and evidence checking not novel |
| [Medical atomic fact checking, 2026](https://pubmed.ncbi.nlm.nih.gov/42770666/) | Guideline-based atomic claim verification/correction | Claim decomposition plus correction not novel |
| [Calibrated medical hidden-state intervention, 2026](https://pubmed.ncbi.nlm.nih.gov/42447946/) | Real-time medical detection/intervention across models and datasets | Strong competitor; distinguish token vs answer-level targets |
| [MedVAL, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-03084-5) | Synthetic evaluator training, physician-annotated evaluation and MEDEC transfer | Independent pre-existing clinical labels essential for clinical claims; restricted dataset remains blocked |
| [MedHallu, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.143/) | Medical hallucination benchmark | Answer roles do not certify arbitrary revised text |
| [MEDEC, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1159/) | Medical error detection/correction benchmark | Alternative reference task conditional on rights and compatibility |
| [PubMedQA, EMNLP 2019](https://aclanthology.org/D19-1259/) | Context excludes conclusion; closed yes/no/maybe gold | Reference-scored intervention feasible in principle, not free-text factuality |
| [Counterfactual Answer Sensitivity, November 2026 preprint](https://arxiv.org/abs/2611.29592) | Evidence sensitivity in medical VLM evaluation | Perturbation sensitivity itself not novel |
| [Healthcare hallucination systematic review, 2026](https://pubmed.ncbi.nlm.nih.gov/42251377) | Reviews mitigation through April 2025 | Useful citation seed, not exhaustive for late 2026 |

Proposed narrow contribution: jointly estimate paired original/revision
correctness from directional evidence contrasts, then select accept/revise/
abstain under explicit preservation and release-risk constraints. Test
whether joint dependence adds value beyond independent marginal predictors
and strong selective correction/calibration baselines. This intersection
is a hypothesis, not a claim that no prior work exists. Further search
using native PubMed, ACL, OpenAlex/Scopus/Web of Science if accessible,
backward/forward chaining, full-text algorithm/implementation comparison,
and updated deduplicated screening logs required before novelty is certified.
PRISMA-S reporting reference: https://doi.org/10.1186/s13643-020-01542-z.
No invented record counts or independent human screening.

## Mathematical specification (closed reference-scored QA only)
For source representative x with fixed original and proposed revision, let
Y0,Yr be exact benchmark correctness. Learn four-state softmax probabilities
p00,p01,p10,p11 on training pairs using categorical cross-entropy plus
L2 regularization (unpenalized intercept). When parsed answers are unchanged,
mask 01/10 as structurally impossible. Candidate features are original/revised
NLI support, contradiction, evidence coverage, support drop under an evidence
perturbation, confidence, answer-change and evidence-available indicators,
five directional differences and product of support drops. All scaled [0,1].
No feature may use gold answers, hidden test labels, original article
conclusions excluded by dataset, or annotation role. Missing evidence is
explicit and scores zero rather than invented support. Exact extractors,
perturbation recipe, models and input hash pins still need implementation and
validation; current code consumes numeric features only. Source-group splits,
parse semantics, and same fixed candidate set across methods are mandatory.

p0=p10+p11, pr=p01+p11. For action A/R/S (accept/revise/abstain), conditional
correct release is p0/pr/0, released-error mass is 1-p0/1-pr/0, induced error
mass is 0/p10/0 and coverage is 1/1/0. For global objective maximize expected
correct releases, constraints:
 E[released_error-r*release]<=0,
 E[introduced_error-i*Y0]<=0,
 E[c-release]<=0.
The Lagrangian (lambda,mu,nu nonnegative) has pointwise scores
 accept=p0-lambda*(1-p0-r)+nu;
 revise=pr-lambda*(1-pr-r)-mu*p10+nu;
 abstain=0.
Action-independent constants mu*i*E[Y0]-nu*c cancel. Choose largest score;
ties prefer abstain then accept. This is a candidate family generation
heuristic, not proof of optimum over all constrained policies, and ordinary
Lagrangian algebra is not a novel theorem. The empirical calibration selection
maximizes observed coverage among certified candidates with correct-release
tie-break; this deliberately differs from candidate generation objective.
Generate at most K=12 fixed policy candidates via train/tune only, record
parameters before calibration. No grid actually frozen yet; no optimizing
calibration features, models, policies or caps after seeing labels. Full
post-generation costs charged even when accepting/abstaining.

Same-marginal counterexample: x=(.05,.25,.05,.65), y=(0,.30,.10,.60)
have p0=.70 and pr=.90. With lambda=nu=0, mu=3, revise score .75 vs .60,
accept .70 both. Hence marginal probabilities alone cannot identify
preservation-sensitive action for all joint distributions. Test empirically
against independent-product p10=p0*(1-pr), direct binary harm predictor,
marginal two-head model, no-contrast and no-perturbation ablations, and
matched-budget selective correction/abstention baselines. This elementary
example does not demonstrate effectiveness or unique literature novelty.

## Calibration improvement: intersection-union tests + Holm
For fixed policy j, define R=Pr(error|released), I=Pr(released error|Y0=1),
C=Pr(released). Require R<r, I<i, C>c as conservative feasibility criteria
(boundary is null); planning targets r=.10,i=.05,c=.50, alpha=.05.
These are illustrative benchmark caps, not clinical-risk thresholds.
Use lower-tail exact binomial p-values at r and i conditional on positive
released and original-correct denominators, and upper-tail at c on n.
Empty risk denominator yields p=1. Under iid source representatives and
fixed policy independent of calibration, the conditional binomial models hold.
For union null H_j=(R>=r OR I>=i OR C<=c), p_j=max(pR,pI,pC) is valid:
if any component null true, Pr(p_j<=t)<=Pr(p_true<=t)<=t.
Apply Holm step-down over K fixed policies. By strong FWER, probability of
declaring any infeasible policy feasible <=alpha, no inter-policy independence
needed. This improves over alpha/(3K) feasibility testing without claiming
simultaneous endpoint-specific confidence intervals. Retain original
Bonferroni intervals if numerical simultaneous bounds needed. Established
intersection-union and Holm methods, not a new statistical theorem.
Reference: Holm, Scand J Stat 6:65–70 (1979), https://ndlsearch.ndl.go.jp/books/R100000136-I1572543024862166272.
The numerical recurrence is tested but not interval arithmetic. No claim
of distribution-free safety under arbitrary source dependence, finite
nonrandom benchmark, transductive filtering, or domain shift. No passing
policy -> report NO_CERTIFIED_POLICY, no post hoc target relaxation.
Independent fixed test assessment reports all actions and denominators;
selection guarantee conditional on stipulated sampling model, not deployment.

## Prospective power / feasibility
No pilot labels have been read. All rates are hypothetical scenarios.
K=12, alpha=.05; first Holm threshold alpha/K; per-policy marginal power
integrates binomial conditional sample counts over release coverage and
original-correct prevalence. Use union bound max(0,pR+pI+pC-2) for
simultaneous pass lower bound at first-step threshold; min marginal powers
upper bound for that policy at that threshold. Not product (correlated
outcomes). Later Holm steps can increase actual family success. Zero-error
minimum denominators computed from (1-cap)^n<=delta; nonzero error requires
more. Exact conditional one-sided McNemar power integrates discordant count
and favorable directions; plan alpha=.025 as prospective conservative
comparison allocation, not final approved multiplicity family. Correct-release
endpoint counts abstentions as failures. Use one representative per source
group chosen before labels for exact tests; report clustered full-cohort
analysis separately. No 18408-row independent-sample inflation.

E1 AP (non-interpolated average precision) primary from PR21: paired
source-group jackknife pseudovalue SD on train-only pilot, then approximate
normal sample size at effect .02 with sensitivity over SD and alpha. This is
a planning approximation, not a powered claim. Final pilot resampling and
cluster-aware bootstrap simulation needed; unequal group sizes and rare
labels may invalidate approximations. Do not use test to estimate variance
or expand after results. The official PubMedQA test 500 is not calibration;
other 500 labeled examples are maximum pre-overlap train+calibration
population, not 500 guaranteed independent groups. MedHallu dev 890 is not
a closed-QA revision calibration set. A sufficiently large authorized
source-disjoint reference-scored calibration cohort is a serious feasibility
blocker. Reduce K prospectively or obtain more eligible groups; if insufficient,
report descriptive results only or revise protocol before test. No use of
restricted MedVAL. No new real data execution authorized.

## Engineering and remaining gates
Added pirc_v2_method.py (joint softmax and IUT/Holm), pirc_v2_power.py
(hypothetical feasibility, paired power, grouped AP planning),
test_pirc_v2.py and synthetic CI. Model extraction, matched baseline training,
private split freeze, rights and eligible calibration groups, exact protocol
lock, full literature implementation comparison and real experiments remain
pending. A 10/10 score is a target, not a verified result. No E1 readiness
config or scientific approval is created.
