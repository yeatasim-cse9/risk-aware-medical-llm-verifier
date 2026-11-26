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

## User-reported actual Kaggle CPU diagnostic (2026-11-26)

Source commit b6400b297d0d6b524fbd9663221e7d4fa9ab49dc.
All six public model asset hashes passed; actual pinned weights loaded.
Reported versions: torch 2.11.0+cpu, transformers 5.20.2, huggingface_hub
1.33.0, numpy 2.1.3, pyarrow 23.0.1. Only torch CPU/CUDA build differs
from the expected GPU version among these five pins.

Exact markers from user-provided output:
- MASTER01EV4_CPU_TOKENIZER=DebertaV2Tokenizer
- MASTER01EV4_CPU_LEGACY_ENCODE=TRUE
- MASTER01EV4_CPU_LEGACY_BUILD_INPUTS_WITH_SPECIAL_TOKENS=FALSE
- MASTER01EV4_CPU_LEGACY_CREATE_TOKEN_TYPE_IDS_FROM_SEQUENCES=FALSE
- MASTER01EV4_CPU_OLD_PATH_FAILED_AT=build_inputs_with_special_tokens
- MASTER01EV4_CPU_OLD_PATH_FAILURE_TYPE=AttributeError
- MASTER01EV4_CPU_OLD_PATH_MISSING_ATTRIBUTE=build_inputs_with_special_tokens
- MASTER01EV4_CPU_SYNTHETIC_MODEL_SMOKE_PASSED
- MASTER01EV4_CPU_DIAGNOSTIC=PASS_RUNTIME_DIFFERS
- MASTER01EV4_CPU_MEDICAL_DATA_USED=FALSE
- MASTER01EV4_BOOTSTRAP_COMPLETE

This reproduces a concrete old-engine failure and validates the corrected
engine on CPU. It is user-provided evidence, not independently fetched
Kaggle output. Original GPU exception attribute was not recorded; strong
causal match, not proof of the exact original traceback.

## Next gate: actual GPU qualification, no medical data

The new public synthetic GPU diagnostic requires a fresh, separately
authorized one-run approval. It enforces all five exact runtime pins,
CUDA availability, deterministic setup, model byte pins, public tokenizer
pair contract, multiwindow and answer budget boundaries, full/removed/
permuted forward checks and repeatability. No private data access or
pilot submission. Report sanitized stage markers and peak memory.
Do not manually install packages or retry a failure.

Only after the GPU diagnostic passes: integrate the versioned V4 private
controller with exact source pins, GPU smoke before historical
reconstruction, both train/dev sampled input-budget checks before any
pilot inference, fine-grained failure telemetry, strict CI, read-only
preflight and separate fresh private data/model rights, quota and
one-pilot authorization. This PR does not provide a private submission
workflow and must not be treated as one. No scientific or clinical approval.
