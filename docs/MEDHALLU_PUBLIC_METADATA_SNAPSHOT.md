# MedHallu metadata-only source snapshot (E1 preparation)

Run `MedHallu Public Metadata Snapshot - No Dataset Rows` in GitHub Actions. On the first merge, its narrow `push` trigger runs automatically. No user laptop or Kaggle code paste is needed.

## What the job observes
- Official Hugging Face MedHallu repository ID and current Hub dataset revision SHA.
- Public dataset-viewer `/splits`, `/size` and `/info` responses for `pqa_labeled` and `pqa_artificial`.
- Declared license metadata, published config/split names, aggregate row counts, native column names.
- Integrity hash of the safe metadata report and the exact GitHub code commit that produced it.

**Scope:** public metadata only. It downloads no dataset rows, Parquet, original clinical text, pretrained weights, or restricted MedVAL data. The workflow has no access to Kaggle, Telegram or paid model API secrets. The verified manifest is kept as a 30-day GitHub Actions artifact, not committed as a dataset.

**Crucial limits:** Hub HEAD revision is version metadata, not a pin of downloaded data bytes. Dataset-viewer endpoints may lag behind Hub HEAD; their records are **not cryptographically bound** to the Hub SHA. A card's MIT license declaration must still be reviewed with underlying PubMedQA terms and original licensing. Native column names do not prove the values, labels, label mapping, grouping IDs, or leakage-safe train/dev/test. Never claim the E1 baseline ran from this snapshot.

Follow-up once this metadata-only job succeeds: inspect actual native fields and licensing; design an authorized row-level audit in an approved private Kaggle CPU environment, with source/file SHA verification before any E1 training. E1 remains BLOCKED by `scripts/e1_readiness.py` until all scientific and legal gates clear.

Public source: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
