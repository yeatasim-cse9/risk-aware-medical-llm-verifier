# MASTER 01B runtime recovery v2 — 2026-11-25

V1 GitHub run 45616754590: CI and submission succeeded; Kaggle status
queued, running, then error. User-supplied safe runtime log reports successful
source reconstruction, MiniLM model weight loading 103/103, then ENCODER_MINILM /
SAFE_FAILURE_REQUIRES_REVIEW at 84.7s. No v1 verified aggregate exists.
Exact original exception is unknown. A plausible failure is legacy fast-tokenizer
API compatibility: Transformers v5 migration guide documents movement of
build_inputs_with_special_tokens out of the tokenizer base API
(https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md).
Do not assert it as the proven root cause.

V2 preserves both model pins, the provisional mapping, threshold grid, 256-token
windows (254 BERT payload tokens plus CLS and SEP), 64-token overlap, exhaustive
pair coverage, and scientific gates. Explicit BERT framing is validated against
the public tokenizer interface on synthetic text. Padding uses explicit IDs and
attention masks. Both exact pinned model/tokenizer pairs must pass a synthetic
single- and multi-window forward/pairwise GPU smoke check BEFORE medical input
downloads. New safe stages isolate load, tokenization, forward and comparisons;
only allowlisted error kind/code is logged, never raw exceptions or data. CI
includes a locally constructed tokenizer and randomly initialized tiny BERT with
Transformers 5.0.0, no data or pretrained weights.

One new v2 private Kaggle slug is permitted only when remote v1 status is error,
v2 is missing, quota endpoint responds, and fresh rights + GPU authorization is
given. Preserve v1 for provenance; no auto retries. Same GitHub workflow, v2
aggregate filename. No final split approval, clinical adjudication, or model
training. The recovery changes are prospective until the authorized v2 run
passes; the original root cause and GPU runtime compatibility remain unconfirmed.
