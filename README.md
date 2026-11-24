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
- ✅ GitHub Actions → Kaggle API authentication; live preflight; private CPU notebook **Version 1** successfully launched remotely; saved 40/100 synthetic task checkpoint returned and validated by GitHub Actions.
- ✅ Remote **three-session** synthetic checkpoint continuation verified: slot A v1 `40/100` → slot B v1 `80/100` → slot A v2 `100/100`, all checkpoint outputs independently downloaded and integrity validated using pinned Git commit `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`.
- ✅ Kaggle → GitHub → Telegram verified synthetic checkpoint notification, and a **scheduled synthetic-only monitor** with a persistent, non-sensitive ledger; duplicate 100/100 notification suppression verified in GitHub Actions run `45479386760`.
- ✅ Step 10.4A–C read-only Kaggle A/B resume guards verified: trusted checkpoints, a CPU quota-unknown safety stop, cooldown/backoff policy tests, duplicate-reservation prevention and failure-recovery state-machine simulations; `kaggle quota` accelerator query succeeded without using GPU time.
- ✅ Private, offline Kaggle GPU smoke package built and unit-tested; GPU quota endpoint and empty separate GPU notebook slot confirmed read-only in GitHub Actions run `45483180421`. GPU submission still requires manual approval.
- ✅ First isolated private Kaggle GPU smoke test executed remotely and independently verified: CUDA matrix result, SHA-256 checkpoint integrity and pinned code provenance in GitHub Actions run `45484283016`.
- ✅ Step 10.5B tiny, untrained Transformer GPU inference and restorable weights code prepared with no medical data or external downloads. Read-only GPU notebook preflight and all 8 new safety tests verified in run `45485423732`. GPU inference job **not yet launched**.
- ✅ First actual CUDA inference on an **untrained synthetic Transformer**, plus saved/reloaded model state and independent checkpoint integrity/provenance verification, confirmed in GitHub Actions run `45485964322`.
- ✅ Step 10.5C source-linked **model-card license metadata** shortlist: MiniLM retrieval (Apache-2.0), MiniLM NLI (Apache-2.0), Microsoft BiomedBERT (MIT); all 8 offline audit safety tests and GitHub run `45486737054` passed. **No model weights acquired or model licenses fully cleared; no GPU use**.
- ✅ Step 10.5D MiniLM retrieval model **immutable Hugging Face revision**, reported safetensors size and metadata SHA-256 pinned by read-only public API workflow `45487590017`. Actual byte hashes/VRAM fit not measured; no weights downloaded or GPU used.
- ⏳ **Actual unattended multi-session dispatch**, live CPU quota proof, durable submitted-intent persistence, pretrained medical-model evaluation and E1–E7 scientific experiments are **not yet deployed or verified**. GitHub cron timing is best-effort.

## Repository map

- `src/research_automation/pilot.py`: deterministic 100-task synthetic CPU pilot, atomic checkpoint writes and validation.
- `scripts/build_kernel.py`: package pinned Git commit into a private CPU-only Kaggle script.
- `scripts/verify_checkpoint.py`: sanity and integrity checks on returned synthetic output.
- `tests/`: local and GitHub Actions unit tests.
- `.github/workflows/research-ci.yml`: automatic basic CI on push / PR.
- `.github/workflows/kaggle-cpu-pilot.yml`: **manual** prepare / auth-check / preflight / submit / status / collect for synthetic-only experiment.
- `.github/workflows/kaggle-readonly-preflight.yml`: read-only live Kaggle preflight on matching `main` source changes; no notebook submission.
- `.github/workflows/synthetic-kaggle-telegram-monitor.yml`: six-hour UTC scheduled synthetic-only checkpoint monitor with persistent duplicate-alert suppression; no notebook submission.
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
