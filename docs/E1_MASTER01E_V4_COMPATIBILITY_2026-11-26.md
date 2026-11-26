# MASTER 01E V4 compatibility remediation — 2026-11-26

Recovery run 45788350458, job 139800887041, main 71e9dbea32fa9f82c9cfb270daab7598895e1685.
User runtime log reached GPU_SMOKE at 39.6 s and emitted ATTRIBUTE_ERROR,
UNCLASSIFIED at 39.7 s. It passed runtime pins, GPU setup, pinned asset
acquisition and model load. No historical reconstruction or private pilot
was reached; no validated aggregate. Both previous GPU authorizations consumed.

Leading cause: Transformers v5 removed/moved legacy tokenizer methods;
the actual DeBERTa tokenizer is tokenizers-backed. Original inference uses
build_inputs_with_special_tokens and create_token_type_ids_from_sequences.
Exact failing attribute not logged. See upstream MIGRATION_GUIDE_V5.md and
DeBERTa-v2 tokenizer template. Prior CI used a BERT fixture, not this pinned
pretrained tokenizer/runtime. This was a coverage gap.

Candidate V4 uses public tokenizer call for raw IDs, explicitly frames
the same CLS A SEP B SEP template including empty premises, verifies
actual public API IDs, masks and segment IDs on invented strings before
inference, fails closed on discrepancy, keeps model, asset hashes, context
windows, overlap, slots, hypothesis limits, inference and aggregation unchanged.
No silent truncation, package upgrades, model substitution, CPU fallback
in a real GPU pilot, or scientific approval. Tests compare unchanged AST.

First run ONLY public synthetic CPU diagnostic in new private Kaggle notebook:
accelerator None, Internet enabled, no data inputs or secrets. It verifies all
six public asset bytes, prints runtime versions and legacy method availability,
checks actual pair framing and full/removed/permuted synthetic CPU inference.
A CPU runtime difference is explicit; CPU success is not GPU qualification.
No real medical data accessed. No new GPU authorized. Keep both failed
notebooks unchanged.

Next gate: inspect actual CPU evidence, fix any remaining incompatibility.
Only then integrate versioned V4 private controller with exact source pins,
pretrained GPU smoke before historical reconstruction, train/dev input-budget
validation before pilot inference, fine-grained failure telemetry, strict CI,
read-only preflight and fresh rights/quota/one-run authorization. This PR
does not provide a private submission workflow and must not be treated as one.
