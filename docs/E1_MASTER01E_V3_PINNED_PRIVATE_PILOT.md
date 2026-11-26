# MASTER 01E V3 — pinned neural assets and private pilot controller
2026-11-26. Base merged PR #28 410917c67f2b9257e3239b3fb76187c4d8c133e7.
Dedicated CI 45779660263: 5/5 synthetic tests including actual random CPU
transformer forward pass; all three pre-merge CI passed. No pretrained
performance or real-data evidence from that run.

## Public model qualification (not medical-data processing)
Selected candidate cross-encoder/nli-deberta-v3-base, immutable revision
6c749ce3425cd33b46d187e45b92bbf96ee12ec7 (2025-04-11 commit).
Primary records:
https://huggingface.co/cross-encoder/nli-deberta-v3-base/commit/6c749ce3425cd33b46d187e45b92bbf96ee12ec7
https://huggingface.co/cross-encoder/nli-deberta-v3-base/tree/6c749ce3425cd33b46d187e45b92bbf96ee12ec7
https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/main/model.safetensors
https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/main/spm.model
https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/main/config.json
Published Apache-2.0, general NLI (SNLI/MNLI), not a clinical verifier.
The model card config names contradiction=0, entailment=1, neutral=2.
Published large-file SHA256 pointers pinned independently:
model.safetensors d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa
spm.model c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd.
Immutable revision tree lists both files; exact-revision API metadata must
also match before proceeding. Qualification job accesses only public
model metadata and four small JSON files, verifies Git blob IDs and
computes SHA256 of those files. It does not download large model assets,
medical datasets or use GPU. A candidate manifest must be reviewed and
committed separately; a missing manifest blocks preflight/submission.
No automatic trust of a moving branch or alternate model. A published
pointer is not independent verification of downloaded bytes. During a
separately authorized pilot, download the six allowlisted files from the
exact revision, verify every byte hash, use slow tokenizer (SentencePiece)
and offline safetensors only, no remote executable code. Fail closed on
dependency or model compatibility drift. New model smoke test uses invented
text before any medical dataset acquisition.

## Private pilot scope
Reconstruct the historical pinned MedHallu/PubMedQA lineage and encoder
screen using unchanged 01C routines, verify quarantine, mapping, balance,
official holdout file and identity. Validate prior audited module hashes
against permanent MASTER 01D V3 archive. Prior private reconstruction
reads pinned native data including test covariates and native answer fields
for integrity/schema; no test answers are passed to the pilot classifier,
no test inference or scoring, and no test-based tuning or new split.
Read only retained train/dev answer pairs for pilot inputs. Reuse V2
deterministic group sampling 128 train and 64 dev; 384 paired answers,
13 score conditions, 18,432 padded NLI forward examples, 1000 group
bootstrap replicates. Entire selected context/answer coverage within fixed
budget; any over-budget case aborts, no automatic removal or retry.
Same 16x512 padded positions per answer/condition, actual compute cost
can differ; record active tokens, total elapsed and GPU peak allocated/
reserved memory for pilot. No new human labels, retrieval or PIRC
intervention; no clinical truth certification. The pilot is exploratory,
not a confirmatory E1 run. No strongest model selected or protocol locked.
Length subgroup analysis computed in private memory only; all subgroup
metrics and counts omitted from public aggregate, not merely suppressed
small cells. No private row-level outputs, model predictions, answers,
IDs, source mappings, weights, or credentials uploaded. Public output
allowlist validates numeric ranges, AP contrasts, group counts, budgets
and hashes; cannot independently reperform private scoring. No actual
split freeze, E1 authorization or legacy approval gate change.

## Manual workflow (only after reviewed manifest and CI verified)
E1 MASTER 01E V3 Private Pilot Controller. First read-only preflight
on main with all three confirmations false; verifies model manifest,
prior audit completed, new private notebook absent and quota endpoint
accessible, not remaining GPU hours. No dataset or GPU submission.
After fresh confirmation of pinned dataset processing rights, model
rights, actual remaining quota and exactly one pilot authorization,
dispatch submit once on main with all three confirmations true.
Do not reuse MASTER 01D V3 authorization. New slug
yeatasimbillah/medical-llm-e1-master01e-v3-pilot-gpu-v1.
First-attempt guard, concurrency, duplicate check immediately before
push; not a distributed transaction against external manual pushes.
Read-only status or verify on uncertain failure; verify requires original
submission commit; no rerun, rename, deletion or retry. Telegram prefix
E1 MASTER 01E V3, completion only after validated aggregate upload.
Controller monitor 240 minutes, workflow timeout 270 minutes.
Temporary private scratch cleanup attempted on normal exit or exception;
forced termination may leave private files. No data transfer to public
GitHub/chat/Telegram. No pretrained weights or raw data in CI. Model-only
metadata qualification is not GPU smoke. BioASQ rights and release access
remain pending, zero eligible PubMedQA calibration groups unchanged.
