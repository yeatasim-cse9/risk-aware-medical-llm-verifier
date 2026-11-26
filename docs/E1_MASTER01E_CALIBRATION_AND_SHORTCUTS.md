# MASTER 01E — calibration remediation and shortcut-resistant E1
Status: engineering preparation, no new real-data execution or final protocol approval.
2026-11-26. Base PR #26, main 011bc679cc877a3acb549650d27d25ef780b0865.
Historical artifact SHA256 6ae3d0e47326dc52498e9b43b22fd1a96eb49aa15e45643b0da213db6fd64af5,
canonical a8e159311f8829a5b5ece7e12e0ffd3a76812f20dd3f066c9d508bf23304ace8.
Both recomputed from original ZIP. 500/500 PubMedQA nonholdout candidates
excluded. Train length paired ranking 0.751918675104349; dev 0.7589887640449439.
No test inspection or alterations.

## Targeted literature and official metadata (2026-11-26)
- https://participants-area.bioasq.org/datasets/ lists 5729 Training 14b
  questions, registration required, CC BY 2.5 and NLM conditions.
- https://participants-area.bioasq.org/general_information/Task14b/ lists
  5389 development questions and four test batches; Phase B uses expert
  selected articles/snippets. Counts differ; exact release bytes not accessed.
- https://www.nature.com/articles/s41597-023-02068-4 (Scientific Data 2023)
  describes existing biomedical expert benchmark annotations.
- https://www.nlm.nih.gov/databases/download.html and
  https://www.ncbi.nlm.nih.gov/home/about/policies/ state underlying
  abstracts may be copyright protected. Dataset license alone does not
  establish processing rights for all content.
- https://aclanthology.org/2025.emnlp-main.143/ MedHallu: benchmark
  generated hallucinated answers paired with PubMedQA-derived references.
- https://aclanthology.org/N18-2017/ and https://aclanthology.org/S18-2023/
  demonstrate hypothesis-only annotation artifacts; https://aclanthology.org/P19-1084/
  studies mitigation. Answer-only control is essential, not novel.
- https://aclanthology.org/2025.findings-emnlp.1035/ evaluates generalization
  of hallucination metrics across models and corpora.
- https://arxiv.org/abs/2605.17028 (preprint) addresses benchmark construction
  artifacts; https://arxiv.org/abs/2610.08026 (preprint) distinguishes reference
  faithfulness and factual correctness. Not peer-reviewed confirmation.
Search scope targeted public metadata and abstracts; no complete systematic
review or full reproduction. A secondary index reports 1541 yes/no training
items (https://www.paperpeel.com/paper/2609.39975); primary arXiv record
unavailable in this audit. Treat only as hypothetical planning scenario,
not verified eligible sample count. At 30% expected allocation, 1541
fully eligible independent groups would give about 462 calibration groups
before any exclusions; actual allocation is hash-based, not exact quota.

## Two distinct tracks
Track A E1: MedHallu native paired answer-role classification, not clinical
truth, not PIRC intervention. Retain the MASTER 01C mapping and quarantine.
No new split, no test score, no E1 readiness change. Track B closed-QA PIRC
requires new source-disjoint reference-scored yes/no calibration and separate
test population. PubMedQA blocked under existing protected-overlap rule.
Do not recycle its reserved holdout. BioASQ preferred but access unverified;
MedQA/MedMCQA separately defined task only, no pooling, MEDEC separate error
task; restricted MedVAL remains prohibited.

## Private BioASQ adapter
No downloader, login, credentials or real data in this change. A separately
authorized runtime must obtain official release through registered account
and confirm exact file version, SHA256, expected count, rights including
snippets, approved private Kaggle transfer/storage and retention. No data
upload to chat or GitHub. Require protected reconstruction with all 01C
originals/holdout plus other known exclusions and prior training sources;
provided protected inventory is not authenticated by pure function.
Decode pinned official JSON yesno only, canonical PubMed URLs from both
documents/snippets, native reference availability yes/no, no gold value
or ideal answer returned. Unknown URL/schema fails closed; handle verified
alternate official format prospectively. Other question types counted but
not used as bridge rows; extra cross-type/semantic audit still required.
Combine candidate and protected proxies, group and choose label-blind
representatives, report only counts and hypothetical power (K=1,3,12).
Optional official gold test file for overlap screening; without it no
external test-ready claim. Test membership never reassigned to train.
Source groups are proxies, not iid proof; pretraining contamination remains
unknown. Gold snippet setting is evidence-provided QA, not retrieval success.
No authorization/approval booleans become true. No final freeze.

## Proposed E1 comparator family
Positive = hallucinated role. Same questions/context for both answers;
no other answer, category, difficulty or ground-truth metadata in input.
Length (normalized token count), style (length/diversity/digits/negation/
punctuation), question-only, answer-only hashed unigram/bigram, question+
answer, evidence lexical overlap, full lexical. Logistic regression with
train-only standardization, fixed hash dimension 256, seed-independent
feature hashing, shared training budgets; report dimensions and runtime.
Question-only paired ranking must be 0.5. Evidence removal from frozen
model and deterministic same-partition cross-source evidence permutation
are sensitivity controls, not label-preserving counterfactuals or proof of
causal grounding. Separately trained question+answer comparison is needed.
Do not rewrite/truncate answers to match lengths or drop pairs based on
test outcomes. Report prospective length-gap strata, including ties, with
denominators and uncertainty. Implement subgroup analysis before test lock.
Current CPU prototype is not a strong neural evidence verifier; actual
NLI/generator/model asset pinning, strong published baselines, fair token/
inference cost comparison, calibration/threshold fitting remain pending.

Proposed primary endpoint AP (positive hallucinated role, balanced 50%
benchmark prevalence), secondary paired ranking half credit ties, Brier;
same parent/source-aligned cluster bootstrap difference with deterministic
seed, 10000 replicates for final protocol. No clinical prevalence claim.
Tune on dev only, strongest baseline selected before test, multiplicity
hierarchy and effect/power plan locked before test, no retuning after.
Prototype percentile bootstrap requires independent source groups and
regularity, not exact finite-sample confidence guarantee. Freeze a single
protocol and evaluate test once after separately authorized pilot/freeze.
No claim of empirical superiority, actual model performance or novelty.
No change to old E1 readiness, historical evidence or registry. No GPU run.
