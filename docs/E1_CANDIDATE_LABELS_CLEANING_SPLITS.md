# E1 MedHallu: candidate binary roles, quarantine, proxy splits (DESIGN ONLY)

**Status (2026-11-25):** CODE COMPLETE FOR SYNTHETIC VALIDATION. **NOT E1 TRAINING OR APPROVED EVALUATION.**

## Evidence, not assumptions

Verified GitHub Actions run: https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45590480345

The independently verified private Kaggle artifact has two source configurations, `pqa_labeled` (1,000) and `pqa_artificial` (9,000). They have 10,000 rows and 9,999 unique normalized questions, one repeated-question group, one same-answer-role collision, and zero identical record pairs. Record-level identities, actual native category frequencies, original PubMedQA IDs and private data contents are **not** in the approved aggregate artifact. Exact revision and two raw-file SHA256s are preserved without source text in `configs/medhallu-e1-prior-aggregate-evidence.json`.

The official MedHallu dataset card identifies the two **answer roles**, `Ground Truth` and `Hallucinated Answer`, for a common `Question`, and notes derivation from PubMedQA. This is grounds to DESIGN a **candidate** two-class role contrast:
- Role **0** = `Ground Truth` answer **as supplied by the benchmark**.
- Role **1** = `Hallucinated Answer` **as supplied by the benchmark**.

**Do not relabel `Difficulty Level` or `Category of Hallucination` as binary or as the proposed evidence-grounded three-way labels.** Category semantics/values and diagnostic trust in the ground-truth answers still need review. These role assignments are NOT physician-adjudicated claim-level gold labels or proof of medical correctness.

Official sources: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu and https://medhallu.github.io/

## Implemented provisional cleaning / split contract

`src/research_data/e1_candidate_splits.py` has `candidate_plan(rows_by_config, seed=20261125)`. This is a pure in-memory design method; it does not download datasets, generate training artifacts, use GPUs, or run an LLM.

1. Require both native six-column configurations and validate structure with the existing `audit_native_rows` checker.
2. **Quarantine** any row where normalized `Ground Truth` and `Hallucinated Answer` are equal. Never automatically assign contradictory role labels to the same text.
3. **Quarantine the entire repeated-question group**, not just an arbitrarily chosen duplicate; neither copy is silently deleted or allowed into train/dev/test.
4. Group remaining parent rows by exact normalized `Question` and identical nonempty `Knowledge` content (conservative proxy for shared source). Both answer roles from a parent stay in the same group and hence in the same partition.
5. Order groups by deterministic SHA256 of a fixed seed and group identity, then propose an **80/10/10 by-group-count** train/dev/test allocation. Proportions of *rows* may differ when groups contain multiple rows. All outputs remain **candidate only**.
6. Return only a private in-memory `(config, row_index) → split` mapping; a separate, strictly allowlisted aggregate summary can be exported. The mapping, text, question fingerprints, source identifiers, category values and raw records must **never** be sent to GitHub, Telegram or the public repository. The aggregate includes a stable digest for future reproducibility checks, not the private assignments.
7. Fail closed when too few groups are independent; reject any fabricated `approved=true` signal or unexpected text fields.

This grouping reduces a class of obvious leakage but **does not establish independent PubMedQA article/question provenance**, semantic paraphrase nonoverlap, temporal independence, model-pretraining noncontamination or generalization. In particular, group by `Knowledge` fingerprint is only an exact-context proxy; original PMID/article IDs are not present in the six-column MedHallu schema.

## Still blocked before freezing E1 science

- Identify verified PubMedQA source article/PMID IDs in an approved private environment and inspect question/abstract provenance. Do not invent IDs or treat proxy hashes as certified lineage.
- Scientifically adjudicate the 1 repeated question group and the 1 conflicting-answer record privately. Approve quarantine versus amended labeling; don't silently discard real data.
- Review native answer-role assumptions, hallucination types/difficulty labels, synthetic-vs-human source configurations, and any manual labeling requirements. Define if the correct answer role is clinically validated or only dataset-provided.
- Run the planner on the exact two SHA256-pinned files **in a distinct, approved private CPU notebook**, with independent verification of a sanitized count-only report. This has **not been run** in this change.
- Evaluate near-duplicate text and source-lineage overlap and freeze *private* split-ID manifests. Do not tune hyperparameters/calibration on the held-out final test.
- Preregister baseline model, prompts/evaluator, metrics (including macro-F1 and paired confidence intervals), seeds, corpus licenses and data rights. Only an authorized, independently reviewed E1 protocol can change the existing `BLOCKED` status.

### Executed vs not executed

- ✅ Verified 10k-row native **structural audit** was completed in a previous private Kaggle run.
- ✅ Candidate label/quarantine/split **algorithm** and synthetic-only tests are implemented and can run as GitHub CI.
- ❌ No private candidate-split generation on real MedHallu rows has run in this phase.
- ❌ No E1 baseline model training, inference or measured hallucination-detection performance has run.
