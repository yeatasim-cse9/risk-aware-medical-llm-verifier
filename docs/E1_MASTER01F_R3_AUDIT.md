# MASTER 01F-R3 — rights and train-only audit
Status: proposed, no actual rights or supervisor approval, no rows acquired.
Exact candidate revision and publisher metadata pin from R2 retained.

## Rights decision
MedHal Apache-2.0 declaration does not settle underlying rights. SumPubMed
declares MIT (https://github.com/vgupta123/sumpubmed); PMC article licenses
vary (https://pmc.ncbi.nlm.nih.gov/tools/openftlist/). MedMCQA has differing
repo and historical data declarations (https://github.com/medmcqa/medmcqa,
https://huggingface.co/datasets/openlifescienceai/medmcqa/blob/6e0171200c3ad193c0a37bd3e935ce16cf9b978a/dataset_infos.json).
MedQA exam and textbook provenance requires review
(https://github.com/jind11/MedQA). ACM declares MIT but derives from
PMC-Patients, whose data declares CC BY-NC-SA 4.0
(https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes/blob/main/README.md,
https://github.com/zhao-zy15/PMC-Patients). Potential license conflict is
not adjudicated here. MedNLI excluded according to current card; its
credentialed DUA is not a cloud processing permission
(https://physionet.org/content/mednli/1.0.0/).
No mixed shard acquisition until all contents are approved. Prefer an
authorized physically separate SumPubMed subset if whole-shard rights
cannot be established; requires a new reviewed manifest and controller.
Request publisher clarification on exact release underlying licenses,
provenance/group inventory, and label polarity. Do not send raw examples.

## Semantics and private audit
Current publisher documentation describes boolean factuality, true=factual
as the natural reading; immutable README observation is a documentation
check, not empirical label validation or legal approval. Require explicit
review before audit; factuality is not equivalent to evidence entailment.
Prescreen SumPubMed only, nonempty context/statement/inner_id, 30k chars.
Eligibility is structural, not proof of evidence sufficiency. Inspect
permitted development examples for semantics later. Exact proxy connected
components on source+inner_id, normalized context and statement; conservative
merging may reduce sample size. No claims of complete paraphrase detection,
article disjointness, cross-benchmark separation or heldout independence.
Require a source identifier mapping and separate provenance-only protected
split inventory before any confirmatory evaluation. One deterministic
representative/group, 256 fit and 128 selection, no label-based resampling.
If either class absent, fail. Fit-only orientation of statement/context
length and lexical overlap; synthetic flag diagnostic only, never feature.
Tied AP uses benchmark nonfactual positive. No model fitting or calibration.
No explanation column read. Actual pyarrow/Kaggle compatibility unverified.
Full train bytes verified before any row read, exact mount inventory; operator
must use exclusive read-only private mount, trusted pyarrow, private
no-accelerator Kaggle, no network during audit. Environment flags not a
sandbox. Private ephemeral report only, no public output, no raw text or IDs.
No execution until fresh explicit private CPU approval and rights receipt
bound to R2 metadata. The label confirmation flag requires separate human
review, does not independently certify correctness. Do not make up receipts.
This phase intentionally avoids another GPU controller.
