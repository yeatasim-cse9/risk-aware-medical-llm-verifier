# MASTER 01F-F — read-only preflight hardening

This is a synthetic/code-only qualification increment. It does not authorize or
perform actual private processing, calibration, test-answer access, model
inference, or GPU submission.

Use a clean, separately authenticated immutable D worktree. The stricter
launcher pins the exact previously published code-only JSON bytes, requires all
84 module hashes, rejects every additional entry in scripts (including caches,
native extensions and symlinks), and checks committed module bytes before the
child process. The isolated child receives a minimal environment and only the
preflight operation. Require its exact two-line success protocol; no output
from private data or private paths should be published. The interpreter, Git,
installed dependencies, filesystem and exclusive control of the read-only
mounts remain operator trust boundaries. This is not a sandbox and does not
prevent concurrent mutation by a privileged actor.

Dedicated CI exercises the actual immutable controller against a deliberately
empty synthetic mount; it must reach PREFLIGHT and fail closed. This is a
negative integration test, not proof of successful actual private preflight
or runtime/model compatibility. No real data or model weights are downloaded.

## Required custody inputs (not generated here)

An authorized custodian must provide an independently authenticated
covariates.json (original all-split covariates only, including excluded source
records, and retained mapping). The 01F-E bridge derives a private deterministic
256/128 selection request. The custodian must provide only those selected
train/dev answer files plus an independently authenticated index bound to that
request and snapshot. Never mount mixed answer columns or test answers. The
projected bundle manifest digest must be approved through a separate trusted
channel. The historical aggregate and mapping digest alone cannot authenticate
covariate text or original answer provenance. Do not store raw rows in Git.

The private operator must provision the exact six pinned public model files,
verify actual asset bytes and dependency compatibility in the intended CPU
environment, and execute the launcher only after input custody is established.
The metadata-only preflight reads the bundle manifest and stats selected files,
not their contents; actual byte validation is a separate authorized preparation
and execution gate. No synthetic marker may substitute for actual success.
