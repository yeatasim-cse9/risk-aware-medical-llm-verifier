# MASTER 01E V2 — neural evidence verification and private pilot preparation
2026-11-26. Base PR #27 f35220287c7eed41f15fa0b7732817831ec291ff.
Post-merge CI 45779254148 and 45779254224 passed. Preserve 01C mapping,
01D V3 aggregate, original holdout and existing E1 authorization gate.
No new dataset/model acquisition, human annotation, GPU job or real pilot.

## Prior art and candidate models (public metadata only)
SummaC, TACL 2022 (https://aclanthology.org/2022.tacl-1.10/) uses NLI
segmentation and aggregation for inconsistency detection; AlignScore, ACL
2023 (https://aclanthology.org/2023.acl-long.634/) uses unified text alignment.
NLI aggregation is an established comparator, not PIRC novelty.
Candidate cross-encoder/nli-deberta-v3-base (model card and config page
https://huggingface.co/cross-encoder/nli-deberta-v3-base,
https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/main/config.json)
lists Apache-2.0 and safetensors. Alternative MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
(https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
Neither is certified clinical. Raw config/API endpoints not accessible during
metadata audit, so exact revisions/file hashes are NOT invented. Future
reviewed manifest must pin revision, SHA256 of each required file, license
and model provenance. Candidate smoke test before data acquisition.
The loader only loads safetensors and local files, disables remote code,
requires manifest digest and exact file set, checks label names and load
mismatches. Caller must establish origin/rights and isolate verified assets
against mutation. No downloader or network action in this code.
CPU synthetic runtime pins are not the approved future Kaggle runtime.

## Inference and comparators
E1 positive = native hallucinated answer role. NLI premise = evidence
window; hypothesis = entire answer text alone. Do not include gold label,
other answer, question-as-assertion or annotation attributes. This is
whole-answer evidence alignment; short/context-dependent answers may be
ambiguous, multi-claim answers may exceed NLI capability. Not atomic
claim verification, and no assertion of medical correctness.
Token window 192, overlap 48, max 16 windows, sequence length 512.
Entire supplied context and hypothesis covered, no silent truncation.
Over-budget input blocks pilot (no automatic exclusion), investigate and
version prospectively. Candidate tokenizer/actual model smoke still needed.
Real neural forward pass through torch, softmax by verified model label
names; aggregate max entailment/contradiction, means, min entailment,
fraction entailment >=.5. These are scores, not calibrated clinical risks.
Train-only 6-feature logistic head, plus raw contradiction/non-entailment.
Same model and fixed 16x512 padded positions per answer in all conditions;
full supplied evidence, empty premise, cross-source permuted evidence.
Ignore dummy slots in full/permuted aggregation; first empty-premise
posterior for removal. Separate trained removal head and frozen full head
under removal/permutation. Equal padded positions/forward examples not
identical FLOPs, active tokens or wall-clock. Count active tokens and real
windows; record total elapsed seconds. Peak memory/per-condition runtime
instrumentation and independently reproduced strong published comparator
still required. Cheap lexical baselines not artificially inflated to neural
budget; compare accuracy-cost separately. Shuffled evidence sensitivity
not a proof of causality or label-preserving counterfactual.

## Private pilot (proposed, exploratory only)
No final test access. Private caller reconstructs historical pinned records,
flags and mapping using existing verified workflow, authenticates the
source/answer binding and supplies train/dev native answers only.
prepare_verified_inputs verifies historical evidence, source files,
retention/quarantine/balance and mapping, rejects any test answers.
It does not itself download data or repeat encoder inference, authenticate
caller-supplied records or establish rights. run requires authorized true
and full train/dev counts, then chooses fixed hash-ranked one representative
per source proxy group (seed 20261126), 128 train and 64 dev. This sample
size is an engineering/compute smoke decision, not power justification.
No selection by role outcomes or length. Source-group proxy independence
not established. Pilot fits train-only, all comparisons exploratory.
13 score conditions: 7 existing lexical/shortcut modes plus full neural,
removed frozen, permuted frozen, removed trained, raw contradiction and
raw non-entailment. AP primary descriptive, paired ranking, Brier,
1000-replicate aligned group-bootstrap AP difference (no multiplicity
claim). Length-gap strata fixed in advance: GT shorter, tie, GT longer 1–5,
GT longer >=6 normalized words. Metrics suppressed for <10 parents in
private report, but public egress policy not yet implemented.
No threshold or strongest comparator locked, no final test run or
confirmatory inference. Need train-only variance pilot and power review,
dev-only tuning and calibration appropriate to E1, fixed primary contrast,
locked protocol and source-group analysis, before test access. Closed-QA
PIRC intervention remains blocked with zero eligible PubMedQA groups;
BioASQ access not confirmed. No clinical or intervention claims from E1.
No public artifact or Kaggle dispatcher for pilot; separate private
controller, strict aggregate allowlist, model asset manifest, GPU preflight,
fresh dataset and model rights plus one-run authorization still required.
No changes to existing approvals, historical evidence or experiment registry.
