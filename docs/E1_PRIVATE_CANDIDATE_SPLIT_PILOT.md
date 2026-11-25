# MedHallu E1 private candidate split pilot — execution gate

Status: **Kaggle-ready code is synthetic-tested, but the real candidate split notebook is NOT yet submitted.** This is a preparation stage, NOT a claim of an approved E1 scientific experiment.

## Frozen upstream evidence
- Official source: `UTAustin-AIHealth/MedHallu` at exact Hub revision `515060458a945c633debc6fd5baac7764416b724`.
- The private Kaggle native-row audit (version 2) produced independently verified GitHub Run [#45590480345](https://github.com/yeatasim-cse9/risk-aware-medical-llm-verifier/actions/runs/45590480345), aggregate artifact 15010069180.
- Both exact Parquet filenames, SHA256 file-byte digests, file sizes, native-row counts and the original aggregate digest are in `configs/medhallu-e1-prior-aggregate-evidence.json`. These are data-integrity anchors and reveal **no source text**.
- Independently verified findings: 10,000 source rows, 9,999 distinct normalized questions, one repeated-question group, one equal reference/hallucinated-answer pair.

## What the new code does in private Kaggle

`scripts/e1_candidate_private_kaggle.py` creates a **private CPU-only notebook**, no GPU/TPU, no external LLM/API calls, with internet allowed solely to acquire a pinned public Hugging Face dataset snapshot. The notebook:
1. Creates a private Kaggle temporary directory and downloads exactly the two approved Parquet files under an immutable Hub source revision.
2. **Blocks all changed source file sizes, byte SHA256s, source row counts and native column sets** before making candidate splits.
3. Independently recomputes the original four aggregate anomaly counts and checks against the prior verified report.
4. Calls the reviewed `candidate_plan(..., seed=20261125)` on in-memory private rows. Quarantines contradictory role pairs and all rows in a repeated normalized-question group, and proposes deterministic 80/10/10 proxy-group train/dev/test partitions. Both candidate answer roles stay together.
5. Deletes local row and assignment references, writes only a sanitized aggregate count/digest and explicit **not approved** flags to `/kaggle/working/e1_candidate_split_aggregate.json`. No source text, category values, question identifiers, per-record assignments, or original Parquet bytes are in GitHub Actions or Telegram output.
6. Catches failures and prints only a safe **stage** and sanitized constant error identifier, never exception/traceback text.

**This is NOT equivalent to a frozen data split or a clinical gold standard.** The private mapping is intentionally *not exported*, so final approved split IDs still need a separate, secure lineage-reviewed protocol before E1 inference or training. The binary contrast is based on benchmark-provided `Ground Truth` vs `Hallucinated Answer` roles; it is NOT the proposal's physician-graded or evidence-grounded three-way claim-level labels.

## GitHub operation (no manual Kaggle paste)

Workflow: [MedHallu E1 Candidate Private Split Audit Controller](../.github/workflows/e1-candidate-private-kaggle.yml)

- `prepare` (default): run only synthetic tests and build/upload a **code-only** package. This also occurs automatically on a code change merged into `main`; no Kaggle submission.
- `preflight`: read-only Kaggle notebook status and quota-API access. No notebook/data execution.
- `submit`: requires **both** checkbox attestations that inherited PubMed abstract rights for private Kaggle analysis have been reviewed and **this one CPU run** has been approved. Rejects existing notebook slug. Prints the exact Git source commit SHA.
- `status`: read-only remote state. GitHub green check confirms only GitHub workflow success, not Kaggle runtime completion.
- `verify`: after Kaggle state `complete`, supply the **exact Git SHA from the submit operation**, checks it belongs to trusted `main` history, retrieves only saved output and validates strict schema, frozen source file SHA256s, source module digests, summary counters, negative scientific-approval flags and report integrity SHA256. Only the verified sanitized JSON is uploaded as a short-lived artifact.
- No `retry_failed` or automatic re-submission is available for this job; failures require a reviewed code fix and a new explicitly approved notebook slug/version.

### Remaining blockers
- Independently verify original PubMedQA article/PMID lineage and semantic duplicate leakage; identical `Knowledge` is only a proxy.
- Review/adjudicate the one duplicate question group and equal-answer collision.
- Validate candidate binary role semantics and confirm final benchmark label mapping and source-specific sampling.
- Approve private frozen split IDs, test isolation, model checkpoints, baseline metrics, reproducibility and rights per E1 experiment protocol.
- Do **not** modify E1 registry status `planned` or mark `leakage_safe_splits_frozen` true at this stage.

The experiment E1 remains **BLOCKED** until independently approved protocol and evidence exist.
