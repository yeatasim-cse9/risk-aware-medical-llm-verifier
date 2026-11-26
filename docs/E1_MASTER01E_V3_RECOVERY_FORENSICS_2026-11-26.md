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

## Versioned controller integration

The separate recovery controller and manual-only GitHub workflow now package the
unchanged pinned original 18 modules plus two recovery modules. The original
model manifest digest is fixed at a1d1677a97ca4a7b6a57c0f0e369c54a53eca014bf5daf1791122d9d7642b1e8.
The original scientific algorithm and historical artifacts remain unchanged.
The recovery envelope binds the old aggregate, exact submission commit and
recovery runner, diagnostics, controller and workflow file hashes; the inner
aggregate still passes the original strict validator. The wrapper emits
fixed-vocabulary stages through a saved descriptor while suppressing raw
stdout/stderr through interpreter exit. A hard kill may leave no final failure
marker. Failure codes classify exact allowlisted types and tokens only.

Recovery notebook: yeatasimbillah/medical-llm-e1-master01e-v3-recovery-gpu-v1.
It is distinct from the failed notebook and cannot overwrite it. The preflight
requires the prior audit complete, old V3 notebook error, new slug absent,
successful quota endpoint, immutable code and manifest pins.
It does NOT certify remaining GPU hours. No dataset or weights are downloaded
by preflight. Public model qualification checks published LFS hashes and small
config bytes; actual large asset bytes are verified in a separately authorized
runtime, not certified by CI.

After merge and verification of all final CI jobs: open Actions > E1 MASTER 01E V3
Recovery Private Pilot Controller > Run workflow > main; operation preflight;
leave ALL three confirmation checkboxes false and expected_submission_commit
blank. Require MASTER01EV3R_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED. Do not
select submit or rerun a previous submission. Fresh dataset/model rights,
quota and exactly-one-run explicit authorization are separately required for
any future submit. No automatic recovery execution. No new inference results.

No change to sampling, data, token limits, model or scientific claims is proposed.
