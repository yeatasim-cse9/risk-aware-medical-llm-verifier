# Risk-Aware Medical LLM Verifier

**Thesis:** Risk-Aware Evidence-Grounded Verification for Detecting and Reducing Hallucinations in Medical LLM Responses

Research repository for a reproducible framework to detect and reduce hallucinations in medical LLM responses through evidence-grounded, claim-level verification and clinical risk-aware decisions.

## Research goals

- Extract atomic medical claims from LLM-generated responses.
- Retrieve traceable evidence from PubMed and DailyMed.
- Compare NLI, evidence-aware LLM judges, and hybrid verification.
- Distinguish **Supported**, **Contradicted**, and **Insufficient Evidence** claims.
- Evaluate evidence sufficiency, confidence calibration, and risk-aware abstention.
- Assess evidence-grounded revision and preservation of correct claims.
- Measure cross-dataset generalization and statistical reliability.

## Planned datasets

| Dataset | Role |
| --- | --- |
| MedHallu | Training and development |
| MedVAL-Bench v1.0.1 | Restricted physician-annotated primary gold test |
| BioASQ 14b | External generalization |
| PubMed 2026 | Biomedical evidence corpus |
| DailyMed | Drug-label evidence corpus |

## Planned experiments

E1: Baseline detection (MedHallu); E2: Retrieval-free judge (MedVAL); E3: Retrieval + NLI (MedVAL); E4: Hybrid verifier (MedVAL); E5: External generalization (BioASQ); E6: Hallucination reduction (BioASQ); E7: Ablation studies.

## Reproducibility and automation

Planned infrastructure: GitHub, GitHub Actions, Kaggle free-tier compute, checkpoint-based resume, and Telegram notifications. Experiment runs will record dataset/model versions, Git commits, seeds, configuration, metrics, and checkpoint metadata. **Infrastructure is not yet deployed or validated.**

## Data protection

**Never commit API tokens, credentials, restricted MedVAL data, private clinical text, or unauthorized dataset copies.** MedVAL processing will occur only under applicable PhysioNet access and data-use terms in an approved environment.

## Disclaimer

This is academic research, not a clinical decision support tool or personalized medical advice.
