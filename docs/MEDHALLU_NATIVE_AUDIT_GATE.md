# MedHallu native-row schema and leakage audit: preparation gate

Status: **AUDITOR BUILT AND SYNTHETIC-TESTED; REAL ROW AUDIT NOT YET AUTHORIZED OR RUN.**

## Source provenance examined on 2026-11-25
- Official dataset: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
- Public metadata audit previously observed upstream repository revision `515060458a945c633debc6fd5baac7764416b724`, which is NOT a verified SHA256 of downloaded Parquet bytes.
- Published configurations: `pqa_labeled` (declared 1,000) and `pqa_artificial` (declared 9,000), each exposing a `train` split. These are source configurations, not leakage-controlled train/dev/test sets.
- Six published native columns: `Question`, `Knowledge`, `Ground Truth`, `Difficulty Level`, `Hallucinated Answer`, `Category of Hallucination`.
- The official card text states an MIT license for MedHallu. However, the queried Hub metadata API did not declare a structured license field, and the underlying PubMedQA/article-abstract licensing and downstream redistribution terms have NOT been independently cleared. Do not flip `access_verified` or approve Kaggle transfer based solely on the card.

## Delivered code
- `src/research_data/medhallu_native_audit.py`: `audit_native_rows(rows_by_config)`, `verify_report()`, `write_aggregate()`. The adapter must receive *already-authorized* in-memory private rows, of both required source configurations, with exact column keys and validated structural types.
- Checks exact six-field schema, nonblank question/answer, `Knowledge` list type, duplicated normalized questions within/across configurations, identical pairs, and cases where normalized ground truth equals the hallucinated answer.
- Outputs **only sanitized aggregate counts and negative readiness flags**. Does not export native category values, sample text, source identifiers, hashed row IDs or paired examples. It never calls the internet or downloads any files.
- `tests/test_medhallu_native_audit.py` uses only synthetic invented examples. GitHub workflow `MedHallu Native Row Audit Preparation - Synthetic Only` runs the tests automatically without Kaggle credentials, GPU, real rows or medical results.

## Interpretation limitations
- Collision/duplicate checks can flag potential contamination but do NOT certify no question-level leakage, semantic paraphrase overlap or PubMedQA ancestry contamination.
- The original dataset contains paired ground-truth and hallucinated responses, but these are NOT automatically three-class evidence-grounded claim annotations. A pre-registered evaluation mapping and held-out grouped splits are mandatory.
- Verifying that each input contains a field named `Category of Hallucination` does not verify allowed category values or human-adjudicated labels. This is not an E1 medical model run.
- The independent E1 readiness gate remains **BLOCKED**; no fabricated `configs/e1-approved-protocol.json` or misleading dataset/source SHA are permitted.

## Next protected execution phase
1. Independently review MedHallu, PubMedQA and source abstract terms and document permissible private Kaggle transfer.
2. Freeze exact official revision and download file-byte SHA256, confirm the expected two source configurations and 10k rows in the approved private Kaggle environment.
3. Run this native audit using a distinct, private, explicitly user-approved CPU-only Kaggle adapter. Never put real row files in a public GitHub artifact or Telegram text. Only aggregated, reviewed metadata may be promoted.
4. Establish source/question lineage IDs, audit overlaps and commit only sanitized split counts/hashes (not rows); preregister baseline, label mapping and uncertainty evaluation.
5. Only then request scientific E1 submission approval.

This stage has no automated Kaggle push because the source rights and row-level handling have not yet passed the required gate.
