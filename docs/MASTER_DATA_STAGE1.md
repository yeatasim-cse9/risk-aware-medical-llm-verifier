# Stage 1: MC01 Dataset Manager + MC02 Data Audit

These are the first **research master code modules**, not E1–E7 experiments.
No biomedical metric or patient data is produced by the synthetic CI unit tests.

## Operating boundary

- GitHub: code, safe configs, tests, provenance documents and metadata-only manifests.
- Kaggle: permitted private data and CPU/GPU compute; no MedVAL.
- MedVAL v1.0.1: restricted and NOT permitted in Kaggle, GitHub, Telegram or third-party APIs. Its benchmark-native schema and physician-gold mapping are not approved.
- BioASQ: registration, terms and a frozen exact release required before ingesting rows.
- MedHallu: public metadata-only audit first; manual license/provenance review and a row-level schema audit in a private Kaggle job before persistent datasets or scientific train/dev/test splits.
- Metadata source: https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu

## What the codes do now

**MC01** contacts only allowlisted public Hugging Face Hub/dataset-viewer *metadata* endpoints. It writes an integrity-checksummed manifest of the immutable remote SHA, native field names, split names/counts, and declared license. It does **not** fetch source rows, Parquet files, native label values or inferred three-class gold.

**MC02** accepts *already normalized, authorized* JSONL rows with:
\`{"id":"nonempty","split":"train|dev|test","text":"nonempty","question_id":"optional","source_id":"optional","group_id":"optional"}\`.
It reports aggregate exact/near textual and lineage-group overlap across splits. Reports **never include raw text, sample IDs or matched pairs**. If too many candidate pairs are generated, it fails closed. Near overlap does not exclude paraphrases or model-pretraining contamination.

The metadata and data-audit stages are distinct. Do not use MC02 on restricted MedVAL in Kaggle. It is not an approved native MedHallu dataset loader, and no train/validation/test split has yet been created.

## Developer/CI commands (NOT tasks for user to run locally)

\`\`\`
python -m unittest discover -s tests -p test_master_data_modules.py -v
python scripts/build_medhallu_audit.py --commit <exact-40-char-SHA> --out build/medhallu-audit
\`\`\`

## Kaggle-only execution sequence

1. GitHub Actions: **MedHallu Metadata Audit Controller → prepare** (code-only package; no Kaggle quota).
2. Manual launch from Actions: **submit** only after reviewing CPU quota and setting \`confirm_cpu_quota=true\`. The run is a **private CPU notebook with internet enabled** only to request public Hugging Face *metadata*. It does not download dataset rows.
3. **status**, then **verify** using the exact submitted Git SHA. Verification downloads only the non-sensitive metadata manifest.
4. Review observed native columns, declared license, data lineage and absence of verified native labels; approve a later version-pinned authorized row audit before training E1.

## Reproducibility limitations

- Source SHA is provenance of public metadata, not a hash of every Parquet file.
- The original source publishes train splits; no scientific held-out test exists merely from their names.
- Public dataset labels may have synthetic biases; PubMedQA lineage is important.
- No test performance, confidence interval or clinical-safety result exists at this stage.
