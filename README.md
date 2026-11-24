# Risk-Aware Medical LLM Verifier

**Thesis:** Risk-Aware Evidence-Grounded Verification for Detecting and Reducing Hallucinations in Medical LLM Responses

Official proposal-aligned, reproducible research on claim-level hallucination verification, evidence retrieval, NLI, evidence-aware LLM evaluation, calibration, clinical risk rules, correction, abstention, and cross-dataset generalization.

## Research experiments

| ID | Purpose | Dataset | Planned environment |
|---|---|---|---|
| E1 | Baseline hallucination detection | MedHallu | Kaggle after audit |
| E2 | Retrieval-free judge | MedVAL-Bench v1.0.1 | Approved restricted environment |
| E3 | Evidence retrieval + NLI | MedVAL-Bench v1.0.1 | Approved restricted environment |
| E4 | Proposed hybrid verifier | MedVAL-Bench v1.0.1 | Approved restricted environment |
| E5 | External generalization | BioASQ 14b | Kaggle, license permitting |
| E6 | Correction / hallucination reduction | BioASQ responses | Kaggle, license permitting |
| E7 | Ablation | Appropriate frozen tests | Dataset-specific approved environment |

Evidence sources: **PubMed 2026**, **DailyMed**, with versioned provenance. E2–E4 use physician annotations for their applicable tasks, not as assumed claim-level three-class gold labels.

## Current status

- ✅ Public GitHub repository and initial manual GitHub Actions CPU smoke test.
- ✅ Human-guided Kaggle CPU checkpoint and cross-notebook resume demonstration (separate UI notebooks).
- ✅ Version-controlled **synthetic** CPU pilot runner, integrity checks, unit tests, a credential-free Kaggle submission builder, and manual Kaggle controller workflows.
- ⏳ Kaggle API token, Telegram bot secrets, API launch/monitoring tests, automatic session chaining, real E1–E7 datasets/models and scientific experiments are **not deployed**.

## Repository map

- `src/research_automation/pilot.py`: deterministic 100-task synthetic CPU pilot, atomic checkpoint writes and validation.
- `scripts/build_kernel.py`: package pinned Git commit into a private CPU-only Kaggle script.
- `scripts/verify_checkpoint.py`: sanity and integrity checks on returned synthetic output.
- `tests/`: local and GitHub Actions unit tests.
- `.github/workflows/research-ci.yml`: automatic basic CI on push / PR.
- `.github/workflows/kaggle-cpu-pilot.yml`: **manual** prepare / submit / status / collect for synthetic-only experiment.
- `configs/experiment-registry.json`: proposal-derived E1–E7 plan.
- `docs/`: security rules, reproducibility contract, operational instructions.

## Free local testing

Requires Python 3.11+; no external packages or API keys:

```powershell
python -m unittest discover -s tests -v
```

Credential-free Kaggle preview:

```powershell
python scripts/build_kernel.py --commit (git rev-parse HEAD) --slot a --out build/kaggle
```

For the safe GitHub Actions procedure see [GitHub → Kaggle Runbook](docs/GITHUB_KAGGLE_RUNBOOK.md).

## Critical safety

This is a public code repository. **Never commit or upload credentialed PhysioNet / MedVAL data, patient text, credentials, or unauthorized data copies.** Kaggle and GitHub Actions are not an approved MedVAL environment by default. See [Security](docs/SECURITY.md) and [Reproducibility](docs/REPRODUCIBILITY.md).

Academic research only — **not clinical advice**.
