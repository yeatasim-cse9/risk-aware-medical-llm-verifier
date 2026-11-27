# R9C operational integration (no science changes)
Use the previously preflighted private notebook and verify the exact R9B source
hash before packaging. Only the four explicit authorization assignments change
for GPU mode; the immutable science plan remains unchanged. No private results,
raw logs, credentials or result artifacts in public Actions.
The new Actions secret THESIS_PRIVATE_ARCHIVE_READ_TOKEN must be a separate
fine-grained token limited to Contents read and Metadata read on the private
archive. Never put the Kaggle write token in Actions. Existing Kaggle and Telegram
secrets are reused. First run readiness preflight (read-only); then, with CPU
approval, qualify the same notebook via API CPU execution and validate a fixed
secret-access marker. This tests actual attachment persistence, not an assumption.
Do not dispatch the GPU workflow until that marker and supervisor/rights/quota
approval are confirmed. The separate GPU workflow is one-shot: prior dispatches,
even failed ones, lock further submissions; no workflow reruns, auto retries, or
manual concurrent edits. Readiness status and verify remain available. A failure
or ambiguous push requires reconciliation; existing stage records are never
overwritten. This integration does not implement mid-stage recovery. A terminal
Kaggle success alone is not scientific approval; verify private chain first.
