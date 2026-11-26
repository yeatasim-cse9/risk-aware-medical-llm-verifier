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

## User-reported exact-runtime GPU qualification (2026-11-26)

Qualification source commit: 4e579148961df905721b2601200a73a9c8e83de5.
User reported WINDOW_BOUNDARY, FULL_FORWARD, REMOVED_FORWARD,
PERMUTED_FORWARD and REPEAT_FULL, then:

- MASTER01EV4_GPU_PEAK_ALLOCATED_BYTES=1149010944
- MASTER01EV4_GPU_PEAK_RESERVED_BYTES=1226833920
- MASTER01EV4_GPU_SYNTHETIC_MODEL_SMOKE_PASSED
- MASTER01EV4_GPU_DIAGNOSTIC=EXACT_RUNTIME_PASS
- MASTER01EV4_GPU_MEDICAL_DATA_USED=FALSE
- MASTER01EV4_GPU_BOOTSTRAP_COMPLETE

The reported exact-runtime marker is generated only after the pinned
version checks and model hash checks. GPU T4 x2 selection was shown in
the notebook screenshot; actual device/version marker lines were not
included in the pasted excerpt. The evidence is user-reported, not
independently fetched from Kaggle. Both CPU and GPU evidence are archived
in configs/master01e-v4-public-qualification-2026-11-26.json.
Authorization for that public GPU diagnostic is consumed.

## Final private controller integration

Separate V4 private runner, budget auditor, diagnostics, staged pilot and
controller. Original 18 module bytes and historical checkpoint/archive,
model manifest and previous controller remain pinned. The 5 new runtime
modules are bound by SHA256 in the V4 envelope. The public qualified
engine and two diagnostic source files are fixed to the qualification
commit; evidence bytes are fixed to their archival commit. Original
V3 validator remains the inner aggregate validator. No clinical
validation or scientific approval is implied.

The controller requests NvidiaTeslaT4 (T4 x2), asserts two Tesla T4
devices at runtime and uses cuda:0 only; no TPU or parallel GPU rewrite.
It repeats public deterministic GPU smoke before historical data access.
The historical pipeline is unchanged and uses GPU encoder screening for
mapping reconstruction before the NLI input audit. The audit checks
both exact deterministic train/dev pilot samples (128 and 64 parents)
before any private NLI forward: both answer roles, full/removed/permuted
evidence, every real and padded slot, complete context token coverage,
vocab range, fixed-length tensor and mask/segment contracts, baseline
feature finiteness and existing same-partition permutation. It computes
expected aggregate costs privately, then reconciles train and dev
actual inference costs before head fitting. Overflow fails closed;
no row dropping, truncation, resampling, fallback or budget change.
The staged pilot uses the original training, metrics, sampling,
bootstrap and baseline functions with only stage markers and exact
cost checks added. No test answers or final scientific evaluation.
Private diagnostic output contains fixed stages and allowlisted codes;
no raw exceptions, private text, row IDs or small-cell results.

CI checks synthetic behavior, AST equivalence, original metrics,
unauthorized wrapper failure, tamper and duplicate protection, and
independently compares all embedded bytes and hashes in the actual
code-only package against Git commit contents. The new slug is
yeatasimbillah/medical-llm-e1-master01e-v4-private-pilot-gpu-v1.
Both old failed notebooks must remain in error state.

After merge AND verified main CI: Actions > E1 MASTER 01E V4 Private Pilot
Controller > Run workflow > main > operation preflight; all confirmation
checkboxes false; expected_submission_commit blank. Expected marker
MASTER01EV4_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED.
Read-only preflight does not determine actual remaining GPU quota or
inspect private input lengths. Real-data budget audit executes inside
the separately authorized future private pilot, before private NLI
inference. New explicit dataset/model rights and GPU quota confirmation
and exactly-one-run authorization required. No automatic retry or new
GPU submission in this implementation phase. No claim that unknown
private-data or infrastructure failures have been eliminated.
