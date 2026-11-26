# MASTER 01E V3 recovery — read-only forensics (2026-11-26)

Failed run: 45783030732, remote job 139782968940, main 2f62bbbb7f4b08350317a9c6d9ec810493ca23a8.
Private notebook: medical-llm-e1-master01e-v3-pilot-gpu-v1. Preserve it.
User-supplied sanitized Kaggle log: 0.8 s MASTER01EV3_PRIVATE_PILOT_STARTED;
41.6 s MASTER01EV3_FAILED_CODE=SAFE_FAILURE_REQUIRES_REVIEW.
HTML conversion warnings followed the marker; causality not established.
GitHub polling observed queued then error, not running. The runtime log establishes
Python wrapper entry, not GPU availability, inference, or scientific completion.
The exact exception/stage is unknown; generic marker loses both non-allowlisted
ValueErrors and other exception types. No validated neural aggregate exists.
Prior one-run authorization was consumed; no reuse.

This candidate adds fixed-stage markers to a separate copy of the original private
algorithm and a strict exception-class/code allowlist. No raw exception strings,
tracebacks, private inputs, identifiers, or model output are emitted. The
original code, model manifest, prior evidence and failed notebook are unchanged.
Synthetic tests compare the original and recovery run AST after removing the
additional authorization gate and fixed markers.

Not an executable Kaggle controller. Before any future submission, integrate a
versioned wrapper that constructs the recorder on a saved descriptor BEFORE
suppression, marks bootstrap/import, validation and output stages, catches
exceptions inside the safe boundary and reports only recorder.failure(exc).
Distinguish hard termination with last observed stage, not fabricated exceptions.
Bind recovery code hashes, verify historical pins and model manifest, preserve
duplicate guards, independently validate aggregates, run strict synthetic CI,
read-only preflight and obtain fresh rights/quota/one-run authorization.
No change to sampling, data, token limits, model or scientific claims is proposed.
