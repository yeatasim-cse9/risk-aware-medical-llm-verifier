# Hybrid E1 Label + Source-Lineage Screen (Manual Private Kaggle)

**Stage:** manual private CPU-only diagnostic, not a clinical gold standard and not model training. The source code for the single bootstrap Kaggle cell lives at `scripts/e1_hybrid_private_label_lineage.py`, with synthetic-only tests in `tests/test_e1_hybrid_private_label_lineage.py`.

## User workflow (no local Python)

1. Review that MedHallu/PubMedQA terms and inherited PubMed-abstract cloud-processing rights are still approved. Create a **PRIVATE** Kaggle notebook, set accelerator **None (CPU)** and Internet **On**.
2. Paste **one pinned loader cell** supplied with the reviewed PR merge's exact 40-character Git SHA and the SHA256 of the script's raw UTF-8 bytes. The loader MUST validate script SHA256 before `exec`, and set `E1_PINNED_GITHUB_COMMIT` to that same exact source Git SHA. Never run `main` or a mutable branch URL without pinning.
3. Select **Run All** once. The Python loader gets two reviewed modules from the same immutable Git commit, then downloads only two approved MedHallu Parquet files into private, temporary Kaggle storage; their exact byte SHA256s and sizes must match the independently verified earlier audit.
4. The script recomputes the source anomaly counts, reruns the fixed candidate split design, **requires the previously verified assignment SHA256**, prints only strictly screened short category labels/counts inside the private notebook, performs a limited rare-5-word-gram cross-split overlap screen and emits the sanitized aggregate `/kaggle/working/e1_hybrid_label_lineage_screen_v1.json`.
5. Send the output line `E1_HYBRID_AUDIT_COMPLETE_PRIVATE_CPU_ONLY` and the count `E1_HYBRID_NEAR_DUPLICATE_FLAGS=<integer>` here, or copy the **aggregate JSON only**. Never send screenshot/cell content with source questions, evidence, article text, raw rows, credentials or per-row split IDs.

## Scientific honesty and privacy

- This runs on the same fixed 10k MedHallu data revision used in the prior private audit. `Ground Truth` vs `Hallucinated Answer` is a **provisional benchmark role**, not a clinically adjudicated hallucination label. Native `Category of Hallucination` and `Difficulty Level` are independently counted for label review, not silently remapped into three evidence-grounded claim labels.
- The six native MedHallu fields contain **no explicit PMID/source article identifier**. Cross-partition matching by exact Knowledge or repeated Question plus bounded n-gram screening CANNOT prove original PubMedQA article lineage, near paraphrase coverage, or semantic leakage isolation.
- The bounded 5-word-gram candidate screen ignores many pairs (including short questions and frequent grams) and is **not an exhaustive semantic duplicate search**. Even `flags=0` cannot prove split independence. The report has explicit false flags for final lineage and split approvals.
- Only the aggregate JSON (counts and permitted SHA digests, with negative-readiness flags) is saved in `/kaggle/working`. No source Parquet, biomedical text, native category names, question fingerprints or row-level mapping is uploaded to GitHub or Telegram.
- Kaggle logs can contain stage/allowlisted short category names inside the **private notebook**. No raw exceptions, source data or remote HTTP messages are deliberately emitted. This is a research preparation check; E1 remains `BLOCKED`.
- Nothing automatically retries or submits another Kaggle run. If a failure occurs, send only the two constant `E1_HYBRID_FAILED_STAGE` / `E1_HYBRID_FAILED_CODE` lines.

## Limitations of the manual approach

Because this is run manually and the prior candidate split mapping is ephemeral, the reported assignment digest does not itself produce a retained private split-ID manifest. A later fully approved, source-lineage-aware experiment protocol must freeze split IDs in a controlled private environment with independently verified immutable data files and model/checkpoint sources. Also, GitHub workflow-level Telegram notifications do not automatically cover manual Kaggle notebook runtime without a separately registered monitoring adapter.
